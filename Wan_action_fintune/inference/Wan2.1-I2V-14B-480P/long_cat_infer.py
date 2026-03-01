#!/usr/bin/env python3
"""
Autoregressive long-video inference for Wan action-conditioned I2V.

Given an arbitrary-length action sequence (T, D), this script generates
video chunks with a fixed `chunk_num_frames` and stitches them into one
long video by:
1) using the last frame of previous chunk as next chunk's input_image
2) dropping the first frame of subsequent chunks to avoid duplication
"""

import argparse
import csv
import os
import glob
import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video, VideoData

from inference import (
    infer_dit_dim,
    load_combined_checkpoint,
    load_action_encoder,
    load_lora_weights,
    load_action_normalization_params,
    normalize_arm_action,
)


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Autoregressive inference for arbitrary-length action sequence (CSV batch)."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="CSV path with columns: prompt,input_image,action_seq[,output_name].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated long videos.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P",
        help="Directory containing base model checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Combined checkpoint (.safetensors) containing LoRA + action encoder.",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="LoRA checkpoint directory (used when checkpoint_path is not set).",
    )
    parser.add_argument(
        "--lora_epoch",
        type=int,
        default=None,
        help="Specific LoRA epoch to load from lora_dir (default: latest).",
    )

    parser.add_argument("--joint_dim", type=int, default=14, help="Action dimension D.")
    parser.add_argument(
        "--action_norm_file",
        type=str,
        default=None,
        help="Path to action normalization JSON generated in data preprocessing.",
    )
    parser.add_argument(
        "--dit_dim",
        type=int,
        default=None,
        help="DiT hidden dimension. If omitted, inferred from config.json.",
    )

    parser.add_argument(
        "--chunk_num_frames",
        type=int,
        default=17,
        help="Frames per autoregressive chunk (recommend 4k+1, e.g. 17).",
    )
    parser.add_argument("--height", type=int, default=240, help="Output video height.")
    parser.add_argument("--width", type=int, default=320, help="Output video width.")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Diffusion inference steps."
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed; chunk i uses seed+i for diversity/continuity.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu.")
    parser.add_argument("--fps", type=int, default=30, help="Output FPS.")
    parser.add_argument("--quality", type=int, default=5, help="Video quality 1-10.")
    return parser.parse_args()


def load_input_image(input_path: str, height: int, width: int) -> Image.Image:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image/video not found: {input_path}")
    if input_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        video_data = VideoData(input_path, height=height, width=width)
        return video_data[0]
    return Image.open(input_path).convert("RGB")


def build_pipe(args):
    model_dir = args.model_dir
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
                skip_download=True,
            ),
            ModelConfig(
                path=os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
                skip_download=True,
            ),
            ModelConfig(path=os.path.join(model_dir, "Wan2.1_VAE.pth"), skip_download=True),
            ModelConfig(
                path=os.path.join(model_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
                skip_download=True,
            ),
        ],
        tokenizer_config=ModelConfig(
            path=os.path.join(model_dir, "google/umt5-xxl"),
            skip_download=True,
        ),
    )

    from diffsynth.models.wan_video_action_encoder import WanActionEncoder

    pipe.action_encoder = WanActionEncoder(joint_dim=args.joint_dim, dit_dim=args.dit_dim).to(
        device=args.device,
        dtype=torch.bfloat16,
    )

    if args.checkpoint_path is not None:
        if not os.path.exists(args.checkpoint_path):
            raise FileNotFoundError(f"Combined checkpoint not found: {args.checkpoint_path}")
        load_combined_checkpoint(pipe, args.checkpoint_path)
    else:
        action_encoder_path = os.path.join(model_dir, "action_encoder.pth")
        if os.path.exists(action_encoder_path):
            load_action_encoder(pipe, action_encoder_path)
        else:
            print(f"Warning: action_encoder.pth not found at {action_encoder_path}")

        if args.lora_dir is not None and os.path.exists(args.lora_dir):
            load_lora_weights(pipe, args.lora_dir, args.lora_epoch)
        elif args.lora_dir is not None:
            print(f"Warning: lora_dir not found: {args.lora_dir}")

    return pipe


def generate_long_video(
    pipe,
    prompt: str,
    input_image_path: str,
    action_seq_path: str,
    args,
    row_seed: int,
):
    print(f"Loading action sequence: {action_seq_path}")
    action_seq_np = np.load(action_seq_path).astype(np.float32)
    if action_seq_np.ndim != 2:
        raise ValueError(f"action_seq should be 2D (T, D), got shape={action_seq_np.shape}")
    t_total, d_action = action_seq_np.shape
    if d_action != args.joint_dim:
        raise ValueError(f"Action dimension mismatch: expected {args.joint_dim}, got {d_action}")
    if t_total < 1:
        raise ValueError("action_seq is empty.")

    if args.action_norm_file is not None:
        if not os.path.exists(args.action_norm_file):
            raise FileNotFoundError(f"Action normalization file not found: {args.action_norm_file}")
        norm_params = load_action_normalization_params(args.action_norm_file)
        action_seq_np = normalize_arm_action(
            action_seq_np,
            norm_params["arm_indices"],
            norm_params["arm_min"],
            norm_params["arm_max"],
        )
        print(f"Applied action normalization from: {args.action_norm_file}")

    first_input_image = load_input_image(input_image_path, args.height, args.width)
    current_input_image = first_input_image

    chunk_size = args.chunk_num_frames
    stride = chunk_size - 1
    all_frames = []

    print(f"Start autoregressive generation: T={t_total}, chunk={chunk_size}, stride={stride}")
    start = 0
    chunk_id = 0
    while start < t_total:
        end = min(start + chunk_size, t_total)
        valid_len = end - start
        chunk_actions = action_seq_np[start:end]

        if valid_len < chunk_size:
            pad_len = chunk_size - valid_len
            pad = np.repeat(chunk_actions[-1:, :], repeats=pad_len, axis=0)
            chunk_actions = np.concatenate([chunk_actions, pad], axis=0)

        chunk_actions_t = torch.from_numpy(chunk_actions).float()
        chunk_seed = row_seed

        print(f"[Chunk {chunk_id}] action[{start}:{end}] valid={valid_len} seed={chunk_seed}")
        chunk_video = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            input_image=current_input_image,
            action_seq=chunk_actions_t,
            height=args.height,
            width=args.width,
            num_frames=chunk_size,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            seed=chunk_seed,
            tiled=True,
        )

        if len(chunk_video) < valid_len:
            raise RuntimeError(
                f"Chunk output too short: got {len(chunk_video)}, expected at least {valid_len}"
            )

        if chunk_id == 0:
            all_frames.extend(chunk_video[:valid_len])
        else:
            all_frames.extend(chunk_video[1:valid_len])

        current_input_image = chunk_video[valid_len - 1]
        start += stride
        chunk_id += 1

    # Safety trim, expected exactly t_total frames
    if len(all_frames) > t_total:
        all_frames = all_frames[:t_total]
    return all_frames


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.chunk_num_frames < 2:
        raise ValueError("--chunk_num_frames must be >= 2")

    print("Loading pipeline...")
    pipe = build_pipe(args)

    print(f"Reading CSV: {args.csv_path}")
    with open(args.csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Found {len(rows)} rows.")

    for idx, row in enumerate(rows):
        try:
            prompt = row["prompt"]
            input_image_path = row["input_image"]
            action_seq_path = row["action_seq"]

            if not os.path.exists(input_image_path):
                print(f"[Row {idx}] input_image not found, skip: {input_image_path}")
                continue
            if not os.path.exists(action_seq_path):
                print(f"[Row {idx}] action_seq not found, skip: {action_seq_path}")
                continue

            print(f"\n[Row {idx}] generating long video...")
            row_seed = args.seed + idx if args.seed is not None else None
            frames = generate_long_video(
                pipe=pipe,
                prompt=prompt,
                input_image_path=input_image_path,
                action_seq_path=action_seq_path,
                args=args,
                row_seed=row_seed,
            )

            output_name = row.get("output_name", "").strip()
            if not output_name:
                output_name = f"long_video_{idx:04d}.mp4"
            elif not output_name.lower().endswith(".mp4"):
                output_name = f"{output_name}.mp4"
            output_path = os.path.join(args.output_dir, output_name)

            print(f"[Row {idx}] Saving long video ({len(frames)} frames) -> {output_path}")
            save_video(frames, output_path, fps=args.fps, quality=args.quality)
            print(f"[Row {idx}] Done.")
        except Exception as e:
            print(f"[Row {idx}] Failed: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\nAll done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

"""
python /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/long_cat_infer.py \
--csv_path /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/long_video_infer/long_cat_input.csv \
--model_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P \
--checkpoint_path /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora/epoch-9.safetensors \
--output_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/long_video_infer/outputs \
--quality 10 \
--action_norm_file /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/norm/action_normalization.json
"""