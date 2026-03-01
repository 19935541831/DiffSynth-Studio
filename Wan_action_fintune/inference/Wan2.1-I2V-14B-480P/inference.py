#!/usr/bin/env python3
"""
Inference script for validating trained Wan video model with action_condition.
Reads a CSV file with prompt, input_image path, and action_seq path.
Generates videos for each row and saves them.
"""

import argparse
import os
import csv
import glob
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict


def infer_dit_dim(model_dir: str) -> int | None:
    """Infer dit_dim from config.json"""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("dim")


ACTION_ENCODER_PREFIX = "pipe.action_encoder."


def load_action_encoder(pipe, action_encoder_path):
    """Load action encoder weights from a dedicated file"""
    print(f"Loading action_encoder from {action_encoder_path}")
    state_dict = load_state_dict(action_encoder_path, torch_dtype=pipe.torch_dtype, device=pipe.device)
    pipe.action_encoder.load_state_dict(state_dict, strict=False)
    print("Action encoder loaded successfully.")


def load_lora_weights(pipe, lora_dir, epoch=None):
    """Load LoRA weights from checkpoint directory"""
    if epoch is None:
        safetensors_files = glob.glob(os.path.join(lora_dir, "epoch-*.safetensors"))
        if not safetensors_files:
            print(f"No LoRA checkpoint found in {lora_dir}")
            return
        epochs = [int(f.split("epoch-")[1].split(".safetensors")[0]) for f in safetensors_files]
        epoch = max(epochs)

    lora_path = os.path.join(lora_dir, f"epoch-{epoch}.safetensors")
    if not os.path.exists(lora_path):
        print(f"LoRA checkpoint not found: {lora_path}")
        return

    print(f"Loading LoRA weights from {lora_path}")
    pipe.load_lora(pipe.dit, lora_path, alpha=1)
    print("LoRA weights loaded successfully.")


def load_combined_checkpoint(pipe, checkpoint_path):
    """Load a combined checkpoint that contains both LoRA and action encoder weights.

    The checkpoint is expected to use the naming convention produced by the training
    script (--remove_prefix_in_ckpt "pipe.dit."):
      - LoRA keys have NO "pipe.dit." prefix, e.g. "blocks.0.attn.q.lora_A.default.weight"
      - Action encoder keys retain the "pipe.action_encoder." prefix, e.g.
        "pipe.action_encoder.mlp.0.weight"
    """
    print(f"Loading combined checkpoint from {checkpoint_path}")
    full_state_dict = load_state_dict(checkpoint_path, torch_dtype=pipe.torch_dtype, device=pipe.device)

    action_encoder_state_dict = {}
    lora_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith(ACTION_ENCODER_PREFIX):
            new_key = key[len(ACTION_ENCODER_PREFIX):]
            action_encoder_state_dict[new_key] = value
        else:
            lora_state_dict[key] = value

    if action_encoder_state_dict:
        print(f"  -> {len(action_encoder_state_dict)} action encoder tensors found, loading...")
        missing, unexpected = pipe.action_encoder.load_state_dict(action_encoder_state_dict, strict=False)
        if missing:
            print(f"     Missing keys: {missing}")
        if unexpected:
            print(f"     Unexpected keys: {unexpected}")
        print("  Action encoder loaded successfully.")
    else:
        print("  Warning: no action encoder weights found in checkpoint.")

    if lora_state_dict:
        print(f"  -> {len(lora_state_dict)} LoRA tensors found, loading...")
        pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=1)
        print("  LoRA weights loaded successfully.")
    else:
        print("  Warning: no LoRA weights found in checkpoint.")


def load_action_seq(action_seq_path):
    """Load action sequence from .npy file"""
    action_seq = np.load(action_seq_path)
    # Convert to torch tensor: (T, D) -> torch.Tensor
    action_seq = torch.from_numpy(action_seq).float()
    return action_seq


def load_action_normalization_params(json_path: str) -> dict:
    """Load action normalization params from JSON (e.g. action_normalization.json).

    Expected format (from convert_to_parquet.py):
        arm_indices, arm_min, arm_max - only arm joints are normalized to [-1, 1];
        gripper dimensions are left unchanged.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "arm_indices": np.array(data["arm_indices"], dtype=np.int32),
        "arm_min": np.array(data["arm_min"], dtype=np.float32),
        "arm_max": np.array(data["arm_max"], dtype=np.float32),
    }


def normalize_arm_action(
    action: np.ndarray,
    arm_indices: np.ndarray,
    arm_min: np.ndarray,
    arm_max: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize arm dimensions to [-1, 1], keep gripper dimensions unchanged.

    action: (T, D) array; arm_indices index the last dimension.
    """
    action = action.astype(np.float32, copy=True)
    denom = np.maximum(arm_max - arm_min, eps)
    action_arm = action[:, arm_indices]
    action_arm_norm = 2.0 * (action_arm - arm_min) / denom - 1.0
    action[:, arm_indices] = np.clip(action_arm_norm, -1.0, 1.0)
    return action


def main():
    parser = argparse.ArgumentParser(description="Inference script for Wan video model with action_condition")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file with columns: prompt, input_image, action_seq"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P",
        help="Directory containing base model checkpoints"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to a combined checkpoint file (.safetensors) that contains both "
            "LoRA and action encoder weights. When provided, --lora_dir and the "
            "action_encoder.pth in --model_dir are ignored."
        )
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help=(
            "Directory containing LoRA epoch checkpoints (epoch-N.safetensors). "
            "Used only when --checkpoint_path is not provided."
        )
    )
    parser.add_argument(
        "--lora_epoch",
        type=int,
        default=None,
        help="Specific LoRA epoch to load from --lora_dir (default: latest)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save generated videos"
    )
    parser.add_argument(
        "--joint_dim",
        type=int,
        default=14,
        help="Dimension of action sequence vectors (joint_dim)"
    )
    parser.add_argument(
        "--action_norm_file",
        type=str,
        default=None,
        help=(
            "Path to action normalization JSON (e.g. action_normalization.json from "
            "convert_to_parquet). When specified, arm joint dimensions are normalized to "
            "[-1, 1]; gripper dimensions are left unchanged."
        )
    )
    parser.add_argument(
        "--dit_dim",
        type=int,
        default=None,
        help="Dimension of DiT model (dit_dim). If not provided, will be inferred from config.json"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=17,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Video height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Video width"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: random)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate for output video (default: 30)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=5,
        help="Video quality (1-10, higher is better quality, default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading base model...")
    model_dir = args.model_dir
    
    # Infer dit_dim if not provided
    if args.dit_dim is None:
        args.dit_dim = infer_dit_dim(model_dir)
        if args.dit_dim is None:
            raise ValueError("Cannot infer dit_dim. Please provide --dit_dim explicitly.")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(
                path=glob.glob(os.path.join(model_dir, "diffusion_pytorch_model*.safetensors")),
                skip_download=True
            ),
            ModelConfig(
                path=os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                skip_download=True
            ),
            ModelConfig(
                path=os.path.join(model_dir, "Wan2.1_VAE.pth"),
                skip_download=True
            ),
            ModelConfig(
                path=os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                skip_download=True
            ),
        ],
        tokenizer_config=ModelConfig(
            path=os.path.join(model_dir, "google/umt5-xxl"),
            skip_download=True
        ),
    )
    
    # Manually initialize action encoder
    print("Initializing action encoder...")
    from diffsynth.models.wan_video_action_encoder import WanActionEncoder
    pipe.action_encoder = WanActionEncoder(
        joint_dim=args.joint_dim,
        dit_dim=args.dit_dim
    ).to(device=args.device, dtype=torch.bfloat16)

    # Load weights: prefer combined checkpoint, fall back to separate files
    if args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Combined checkpoint not found: {args.checkpoint_path}")
        load_combined_checkpoint(pipe, args.checkpoint_path)
    else:
        # Load action encoder weights from model_dir
        action_encoder_path = os.path.join(model_dir, "action_encoder.pth")
        if os.path.exists(action_encoder_path):
            load_action_encoder(pipe, action_encoder_path)
        else:
            print(f"Warning: action_encoder.pth not found at {action_encoder_path}")

        # Load LoRA weights if directory provided
        if args.lora_dir is not None and os.path.exists(args.lora_dir):
            load_lora_weights(pipe, args.lora_dir, args.lora_epoch)
        elif args.lora_dir is not None:
            print(f"Warning: lora_dir not found: {args.lora_dir}")
    
    # Load action normalization params if specified
    norm_params = None
    if args.action_norm_file is not None:
        if not os.path.exists(args.action_norm_file):
            raise FileNotFoundError(f"Action normalization file not found: {args.action_norm_file}")
        norm_params = load_action_normalization_params(args.action_norm_file)
        print(f"Loaded action normalization from {args.action_norm_file} (arm dims only)")

    # Read CSV file
    print(f"Reading CSV file: {args.csv_path}")
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Found {len(rows)} rows in CSV file")
    
    # Process each row
    for idx, row in enumerate(tqdm(rows, desc="Generating videos")):
        try:
            prompt = row['prompt']
            input_image_path = row['input_image']
            action_seq_path = row['action_seq']
            
            # Validate paths
            if not os.path.exists(input_image_path):
                print(f"Warning: Input image not found: {input_image_path}, skipping...")
                continue
            
            if not os.path.exists(action_seq_path):
                print(f"Warning: Action sequence not found: {action_seq_path}, skipping...")
                continue
            
            # Load input image (support both image and video files)
            if input_image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                # Extract first frame from video
                video_data = VideoData(input_image_path, height=args.height, width=args.width)
                input_image = video_data[0]
            else:
                # Load image file
                input_image = Image.open(input_image_path).convert("RGB")
            
            # Load action sequence
            action_seq_np = np.load(action_seq_path)
            if norm_params is not None:
                action_seq_np = normalize_arm_action(
                    action_seq_np,
                    norm_params["arm_indices"],
                    norm_params["arm_min"],
                    norm_params["arm_max"],
                )
            action_seq = torch.from_numpy(action_seq_np).float()

            # Validate action sequence shape
            T_action, D_action = action_seq.shape
            if D_action != args.joint_dim:
                print(f"Warning: Action sequence dimension mismatch. Expected {args.joint_dim}, got {D_action}, skipping...")
                continue
            
            if T_action != args.num_frames:
                print(f"Warning: Action sequence length mismatch. Expected {args.num_frames}, got {T_action}, skipping...")
                continue
            
            # Generate video
            print(f"\nGenerating video {idx+1}/{len(rows)}: {prompt[:50]}...")
            video = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                input_image=input_image,
                action_seq=action_seq,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                tiled=False,
            )
            
            # Save video
            output_filename = f"video_{idx:04d}.mp4"
            output_path = os.path.join(args.output_dir, output_filename)
            save_video(video, output_path, fps=args.fps, quality=args.quality)
            print(f"Saved video to: {output_path}")
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nCompleted! Generated videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=2 python /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/Wan2.1-I2V-14B-480P/inference.py \
--csv_path /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/short_video_infer/inference_input.csv \
--model_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P \
--checkpoint_path /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora_1/epoch-4.safetensors \
--output_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/short_video_infer/outputs \
--quality 10 \
--action_norm_file /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/norm/action_normalization.json
"""