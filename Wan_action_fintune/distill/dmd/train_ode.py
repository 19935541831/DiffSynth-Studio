"""
ODE stage training: train causal student on teacher ODE trajectories.
Usage: python -m Wan_action_fintune.distill.dmd.train_ode --config config_ode.yaml
"""

import os
import sys
import argparse
from typing import Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .ode_regression import ODERegression
from .dataset_ode import ODETrajectoryDataset


def build_action_for_student(
    action_seq: torch.Tensor,
    num_latent_frames: int,
    h: int,
    w: int,
    joint_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    action_seq: (B, T_act, joint_dim). Pack to (B, T_latent, 4*joint_dim), expand to (B, T_video, 4*joint_dim).
    """
    B, T_act, D = action_seq.shape
    first = action_seq[:, :1, :].repeat(1, 3, 1)
    packed = torch.cat([first, action_seq], dim=1)
    if packed.shape[1] % 4 != 0:
        packed = packed[:, : (packed.shape[1] // 4) * 4]
    T_latent = packed.shape[1] // 4
    packed = packed.reshape(B, T_latent, 4 * D).to(device=device, dtype=dtype)
    action_expanded = packed.unsqueeze(3).unsqueeze(4).expand(
        B, T_latent, h, w, 4 * D
    ).reshape(B, T_latent * h * w, 4 * D)
    return action_expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_ode.yaml")
    parser.add_argument("--data_dir", type=str, help="ODE trajectory dir (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Checkpoint dir (overrides config)")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--generator_ckpt", type=str)
    parser.add_argument("--num_frame_per_block", type=int, default=1)
    parser.add_argument("--joint_dim", type=int, default=14)
    parser.add_argument("--num_ode_steps", type=int, default=50)
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    if os.path.isfile(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_dir = getattr(args, "model_dir", None)
    ode = ODERegression(args, device)
    ode.generator.train()
    optimizer = torch.optim.AdamW(
        [p for p in ode.generator.parameters() if p.requires_grad],
        lr=getattr(args, "lr", 1e-4),
        betas=(getattr(args, "beta1", 0.9), getattr(args, "beta2", 0.999)),
    )

    data_dir = getattr(args, "data_dir", None)
    if not data_dir or not os.path.isdir(data_dir):
        print("data_dir missing or not a directory; using dummy one batch for sanity check.")
        dataset = None
    else:
        dataset = ODETrajectoryDataset(
            data_dir,
            max_samples=getattr(args, "max_samples", None),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=getattr(args, "batch_size", 2),
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: {
                "ode_latent": torch.stack([t["ode_latent"] for t in x]),
                "prompt": [t["prompt"] for t in x],
                "action_seq": torch.stack([t["action_seq"] for t in x]),
            },
        )
        dataloader = iter(dataloader)

    output_dir = getattr(args, "output_dir", "./ode_ckpts")
    os.makedirs(output_dir, exist_ok=True)
    max_steps = getattr(args, "max_steps", 10_000)
    save_every = getattr(args, "save_every", 1000)
    patch_size = (getattr(ode.generator, "patch_size", (2, 2, 2)) or (2, 2, 2))[:3]
    if isinstance(patch_size, (list, tuple)) and len(patch_size) >= 2:
        ph, pw = patch_size[1], patch_size[2]
    else:
        ph, pw = 2, 2

    for step in range(max_steps):
        if dataset is not None:
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(dataloader)
                batch = next(dataloader)
        else:
            B, T_plus, F, C, H, W = 1, 51, 21, 16, 30, 52
            batch = {
                "ode_latent": torch.randn(B, T_plus, F, C, H, W, device=device, dtype=ode.dtype),
                "prompt": ["a dummy prompt"],
                "action_seq": torch.randn(B, 81, getattr(args, "joint_dim", 14), device=device, dtype=ode.dtype),
            }

        ode_latent = batch["ode_latent"].to(device, dtype=ode.dtype)
        prompts = batch["prompt"]
        action_seq = batch["action_seq"].to(device, dtype=ode.dtype)
        B, _T_plus, F, C, H, W = ode_latent.shape
        h, w = H // ph, W // pw
        with torch.no_grad():
            context = ode.encode_prompts(prompts)
            action = build_action_for_student(
                action_seq,
                F,
                h,
                w,
                getattr(args, "joint_dim", 14),
                device,
                ode.dtype,
            )
        loss, log_dict = ode.generator_loss(
            ode_latent,
            context,
            action,
            clip_feature=None,
            y=None,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ode.generator.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step} loss={loss.item():.6f} {log_dict}")
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"generator_{step+1:06d}.pt")
            torch.save(ode.generator.state_dict(), ckpt_path)
            print(f"Saved {ckpt_path}")

    print("ODE training done.")


if __name__ == "__main__":
    main()
