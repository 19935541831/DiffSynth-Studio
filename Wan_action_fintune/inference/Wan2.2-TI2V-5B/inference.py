#!/usr/bin/env python3
"""
Inference script for validating trained Wan2.2-TI2V-5B model with action_condition.
Reads a CSV file with prompt, input_image path, and action_seq path.
Generates videos for each row and saves them.
"""

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.core import load_state_dict
from diffsynth.models.wan_video_action_encoder import WanActionEncoder
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import VideoData, save_video

ACTION_ENCODER_PREFIX = "pipe.action_encoder."
DIT_PREFIX = "pipe.dit."
NEGATIVE_PROMPT_DEFAULT = ""


def infer_dit_dim(model_dir: str) -> int | None:
    """Infer dit_dim from config.json."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("dim")


def load_action_encoder(pipe: WanVideoPipeline, action_encoder_path: str) -> None:
    """Load action encoder weights from a dedicated file."""
    print(f"Loading action_encoder from {action_encoder_path}")
    state_dict = load_state_dict(action_encoder_path, torch_dtype=pipe.torch_dtype, device=pipe.device)
    missing, unexpected = pipe.action_encoder.load_state_dict(state_dict, strict=False)
    loaded_count = len(state_dict) - len(unexpected)
    if loaded_count <= 0:
        raise RuntimeError(
            "Failed to load action_encoder: no tensors matched. "
            "Please check checkpoint format and joint_dim/dit_dim settings."
        )
    if missing:
        print(f"  Missing keys in action_encoder: {missing}")
    if unexpected:
        print(f"  Unexpected keys in action_encoder: {unexpected}")
    print("Action encoder loaded successfully.")


def load_lora_weights(pipe: WanVideoPipeline, lora_dir: str, epoch: int | None = None) -> None:
    """Load LoRA weights from checkpoint directory."""
    if epoch is None:
        safetensors_files = glob.glob(os.path.join(lora_dir, "epoch-*.safetensors"))
        if not safetensors_files:
            print(f"No LoRA checkpoint found in {lora_dir}")
            return
        epochs = [int(Path(f).stem.split("epoch-")[1]) for f in safetensors_files]
        epoch = max(epochs)

    lora_path = os.path.join(lora_dir, f"epoch-{epoch}.safetensors")
    if not os.path.exists(lora_path):
        print(f"LoRA checkpoint not found: {lora_path}")
        return

    print(f"Loading LoRA weights from {lora_path}")
    pipe.load_lora(pipe.dit, lora_path, alpha=1)
    print("LoRA weights loaded successfully.")


def _looks_like_lora_key(key: str) -> bool:
    return (".lora_A." in key) or (".lora_B." in key) or ("lora_up" in key) or ("lora_down" in key)


def load_combined_checkpoint(pipe: WanVideoPipeline, checkpoint_path: str) -> None:
    """Load a combined checkpoint containing action encoder + (LoRA or full DiT) weights.

    This function supports two common checkpoint styles:
    1) LoRA checkpoint:
       - action encoder keys: pipe.action_encoder.*
       - LoRA keys: no pipe.dit prefix, e.g. blocks.0.attn.q.lora_A.default.weight
    2) Full DiT checkpoint:
       - action encoder keys: pipe.action_encoder.*
       - DiT keys: with or without pipe.dit prefix
    """
    print(f"Loading combined checkpoint from {checkpoint_path}")
    full_state_dict = load_state_dict(checkpoint_path, torch_dtype=pipe.torch_dtype, device=pipe.device)

    action_encoder_state_dict: dict[str, torch.Tensor] = {}
    dit_or_lora_state_dict: dict[str, torch.Tensor] = {}

    for key, value in full_state_dict.items():
        if key.startswith(ACTION_ENCODER_PREFIX):
            action_encoder_state_dict[key[len(ACTION_ENCODER_PREFIX):]] = value
            continue

        if key.startswith(DIT_PREFIX):
            dit_or_lora_state_dict[key[len(DIT_PREFIX):]] = value
        else:
            dit_or_lora_state_dict[key] = value

    if action_encoder_state_dict:
        print(f"  -> {len(action_encoder_state_dict)} action encoder tensors found, loading...")
        missing, unexpected = pipe.action_encoder.load_state_dict(action_encoder_state_dict, strict=False)
        loaded_count = len(action_encoder_state_dict) - len(unexpected)
        if loaded_count <= 0:
            raise RuntimeError(
                "Action encoder tensors found but none matched model keys. "
                "Please ensure checkpoint matches Wan2.2-TI2V-5B action encoder architecture."
            )
        if missing:
            print(f"     Missing keys: {missing}")
        if unexpected:
            print(f"     Unexpected keys: {unexpected}")
        print("  Action encoder loaded successfully.")
    else:
        raise RuntimeError(
            "No action encoder weights found in checkpoint. "
            "Action-conditioned inference requires action encoder weights."
        )

    if not dit_or_lora_state_dict:
        print("  Warning: no DiT/LoRA weights found in checkpoint.")
        return

    has_lora = any(_looks_like_lora_key(k) for k in dit_or_lora_state_dict)
    if has_lora:
        print(f"  -> {len(dit_or_lora_state_dict)} LoRA tensors found, loading...")
        pipe.load_lora(pipe.dit, state_dict=dit_or_lora_state_dict, alpha=1)
        print("  LoRA weights loaded successfully.")
    else:
        print(f"  -> {len(dit_or_lora_state_dict)} DiT tensors found, loading full DiT state...")
        missing, unexpected = pipe.dit.load_state_dict(dit_or_lora_state_dict, strict=False)
        loaded_count = len(dit_or_lora_state_dict) - len(unexpected)
        if loaded_count <= 0:
            raise RuntimeError(
                "Failed to load DiT weights: no tensors matched. "
                "Checkpoint likely belongs to another model/version."
            )
        if missing:
            print(f"     Missing keys: {missing}")
        if unexpected:
            print(f"     Unexpected keys: {unexpected}")
        print("  DiT weights loaded successfully.")


def load_action_normalization_params(json_path: str) -> dict:
    """Load action normalization params from JSON.

    Expected format:
      arm_indices, arm_min, arm_max
    Only arm joints are normalized to [-1, 1]; gripper dimensions are unchanged.
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
    """Normalize arm dimensions to [-1, 1], keep gripper dimensions unchanged."""
    action = action.astype(np.float32, copy=True)
    denom = np.maximum(arm_max - arm_min, eps)
    action_arm = action[:, arm_indices]
    action_arm_norm = 2.0 * (action_arm - arm_min) / denom - 1.0
    action[:, arm_indices] = np.clip(action_arm_norm, -1.0, 1.0)
    return action


def build_pipeline(args: argparse.Namespace) -> WanVideoPipeline:
    print("Loading base model...")

    if args.dit_dim is None:
        args.dit_dim = infer_dit_dim(args.model_dir)
        if args.dit_dim is None:
            raise ValueError("Cannot infer dit_dim from config.json. Please provide --dit_dim.")

    diffusion_paths = glob.glob(os.path.join(args.model_dir, "diffusion_pytorch_model*.safetensors"))
    if not diffusion_paths:
        raise FileNotFoundError(f"No diffusion model files found in: {args.model_dir}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=[
            ModelConfig(path=diffusion_paths, skip_download=True),
            ModelConfig(path=os.path.join(args.model_dir, "models_t5_umt5-xxl-enc-bf16.pth"), skip_download=True),
            ModelConfig(path=os.path.join(args.model_dir, "Wan2.2_VAE.pth"), skip_download=True),
        ],
        tokenizer_config=ModelConfig(
            path=os.path.join(args.model_dir, "google/umt5-xxl"),
            skip_download=True,
        ),
    )

    print("Initializing action encoder...")
    pipe.action_encoder = WanActionEncoder(
        joint_dim=args.joint_dim,
        dit_dim=args.dit_dim,
    ).to(device=args.device, dtype=torch.bfloat16)

    if args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Combined checkpoint not found: {args.checkpoint_path}")
        load_combined_checkpoint(pipe, args.checkpoint_path)
    else:
        action_encoder_path = os.path.join(args.model_dir, "action_encoder.pth")
        if os.path.exists(action_encoder_path):
            load_action_encoder(pipe, action_encoder_path)
        else:
            raise FileNotFoundError(
                f"action_encoder.pth not found at {action_encoder_path}. "
                "Please provide --checkpoint_path with action encoder weights."
            )

        if args.lora_dir is not None and os.path.exists(args.lora_dir):
            load_lora_weights(pipe, args.lora_dir, args.lora_epoch)
        elif args.lora_dir is not None:
            print(f"Warning: lora_dir not found: {args.lora_dir}")

    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for Wan2.2-TI2V-5B with action_condition")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV path with columns: prompt,input_image,action_seq")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.2-TI2V-5B",
        help="Directory containing base model checkpoints",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to a combined checkpoint (.safetensors) containing action encoder and "
            "LoRA/full DiT weights. If provided, --lora_dir and model_dir/action_encoder.pth are ignored."
        ),
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Directory containing LoRA epoch checkpoints (epoch-N.safetensors)",
    )
    parser.add_argument("--lora_epoch", type=int, default=None, help="Specific LoRA epoch to load (default: latest)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save generated videos")
    parser.add_argument("--joint_dim", type=int, default=14, help="Action sequence vector dimension")
    parser.add_argument(
        "--action_norm_file",
        type=str,
        default=None,
        help="Path to action normalization JSON. Only arm dims will be normalized to [-1, 1].",
    )
    parser.add_argument(
        "--dit_dim",
        type=int,
        default=None,
        help="DiT dimension. If omitted, inferred from model_dir/config.json",
    )
    parser.add_argument("--num_frames", type=int, default=9, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=256, help="Video height")
    parser.add_argument("--width", type=int, default=320, help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate for output video")
    parser.add_argument("--quality", type=int, default=5, help="Video quality (1-10)")
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT_DEFAULT, help="Negative prompt")
    parser.add_argument(
        "--disable_prompt",
        action="store_true",
        help="Ignore prompt in CSV and run with empty prompt (recommended only if training used --disable_prompt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pipe = build_pipeline(args)

    norm_params = None
    if args.action_norm_file is not None:
        if not os.path.exists(args.action_norm_file):
            raise FileNotFoundError(f"Action normalization file not found: {args.action_norm_file}")
        norm_params = load_action_normalization_params(args.action_norm_file)
        print(f"Loaded action normalization from {args.action_norm_file} (arm dims only)")

    print(f"Reading CSV file: {args.csv_path}")
    with open(args.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Found {len(rows)} rows in CSV file")

    for idx, row in enumerate(tqdm(rows, desc="Generating videos")):
        try:
            prompt = row["prompt"]
            input_image_path = row["input_image"]
            action_seq_path = row["action_seq"]

            if not os.path.exists(input_image_path):
                print(f"Warning: input image not found: {input_image_path}, skipping...")
                continue
            if not os.path.exists(action_seq_path):
                print(f"Warning: action sequence not found: {action_seq_path}, skipping...")
                continue

            if input_image_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_data = VideoData(input_image_path, height=args.height, width=args.width)
                input_image = video_data[0]
            else:
                input_image = Image.open(input_image_path).convert("RGB")

            action_seq_np = np.load(action_seq_path)
            if norm_params is not None:
                action_seq_np = normalize_arm_action(
                    action_seq_np,
                    norm_params["arm_indices"],
                    norm_params["arm_min"],
                    norm_params["arm_max"],
                )
            action_seq = torch.from_numpy(action_seq_np).float()

            if action_seq.ndim != 2:
                print(f"Warning: action_seq must have shape (T, D), got {tuple(action_seq.shape)}, skipping...")
                continue

            t_action, d_action = action_seq.shape
            if d_action != args.joint_dim:
                print(f"Warning: action dim mismatch. expected={args.joint_dim}, got={d_action}, skipping...")
                continue
            if t_action != args.num_frames:
                print(f"Warning: action length mismatch. expected={args.num_frames}, got={t_action}, skipping...")
                continue

            print(f"\nGenerating video {idx + 1}/{len(rows)}: {prompt[:50]}...")
            prompt_input = "" if args.disable_prompt else prompt
            video = pipe(
                prompt=prompt_input,
                negative_prompt=args.negative_prompt,
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

            output_path = os.path.join(args.output_dir, f"video_{idx:04d}.mp4")
            save_video(video, output_path, fps=args.fps, quality=args.quality)
            print(f"Saved video to: {output_path}")

        except Exception as err:
            print(f"Error processing row {idx}: {err}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nCompleted! Generated videos saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Example:
CUDA_VISIBLE_DEVICES=0 python /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/Wan2.2-TI2V-5B/inference.py \
  --csv_path /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/short_video_infer/inference_input.csv \
  --model_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.2-TI2V-5B \
  --checkpoint_path /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.2-TI2V-5B_sft_1/epoch-16.safetensors \
  --output_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/short_video_infer/outputs_wan22 \
  --quality 10 \
  --action_norm_file /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/norm/action_normalization.json
"""
