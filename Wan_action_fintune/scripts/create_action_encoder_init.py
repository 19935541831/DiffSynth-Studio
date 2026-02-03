# create_action_encoder_init.py
import argparse
import json
import os
import torch

from diffsynth.models.wan_video_action_encoder import WanActionEncoder


def infer_dit_dim(model_dir: str) -> int | None:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config.get("dim")


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize Wan action encoder.")
    parser.add_argument("--model_dir", type=str, default="/project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P")
    parser.add_argument("--joint_dim", type=int, default=14)
    parser.add_argument("--dit_dim", type=int, default=None)
    parser.add_argument("--out_path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dit_dim = args.dit_dim or infer_dit_dim(args.model_dir)
    if dit_dim is None:
        raise ValueError("Cannot infer dit_dim. Please pass --dit_dim explicitly.")
    out_path = args.out_path or os.path.join(args.model_dir, "action_encoder.pth")

    encoder = WanActionEncoder(joint_dim=args.joint_dim, dit_dim=dit_dim)
    torch.save(encoder.state_dict(), out_path)
    print(f"Saved action_encoder.pth to {out_path} (joint_dim={args.joint_dim}, dit_dim={dit_dim})")


if __name__ == "__main__":
    main()
