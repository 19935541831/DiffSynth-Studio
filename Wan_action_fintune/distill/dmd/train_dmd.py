"""
DMD stage training: generator (causal student) + critic (teacher) with backward simulation.
Usage: python -m Wan_action_fintune.distill.dmd.train_dmd --config config_dmd.yaml
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from .dmd import DMD
from .dataset_ode import ODETrajectoryDataset


def run_units_single(
    pipe,
    prompt: str,
    action_seq: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
) -> Dict[str, Any]:
    """Run pipeline units for one sample; return context, action_emb, clip_feature, y."""
    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": 50,
    }
    inputs_nega = {
        "negative_prompt": "",
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": 50,
    }
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": None,
        "camera_control_origin": None,
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": None,
        "rand_device": "cpu",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": 1.0,
        "cfg_merge": False,
        "sigma_shift": 5.0,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": None,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "action_seq": action_seq.unsqueeze(0) if action_seq.dim() == 2 else action_seq,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )
    return {
        "context": inputs_posi.get("context"),
        "action_emb": inputs_shared.get("action_emb"),
        "clip_feature": inputs_shared.get("clip_feature"),
        "y": inputs_shared.get("y"),
    }


def build_conditional_dict(
    pipe,
    prompts: List[str],
    action_seqs: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    joint_dim: int,
    patch_size: tuple,
) -> tuple:
    """
    Run units per sample and stack; build action_for_student (B, T_video, 4*joint_dim).
    Returns conditional_dict, unconditional_dict.
    """
    B = len(prompts)
    contexts, action_embs, clip_features, ys = [], [], [], []
    for i in range(B):
        ac = action_seqs[i].to(pipe.device, dtype=pipe.torch_dtype)
        out = run_units_single(pipe, prompts[i], ac, height, width, num_frames)
        contexts.append(out["context"])
        action_embs.append(out["action_emb"])
        clip_features.append(out["clip_feature"])
        ys.append(out["y"])

    def stack_or_cat(tensors, dim=0):
        if tensors[0] is None:
            return None
        return torch.cat([t.to(device, dtype=dtype) for t in tensors], dim=dim)

    context = stack_or_cat(contexts)
    action_emb = stack_or_cat(action_embs)
    clip_feature = stack_or_cat(clip_features) if clip_features[0] is not None else None
    y = stack_or_cat(ys) if ys[0] is not None else None

    T_latent = (3 + num_frames) // 4
    if (3 + num_frames) % 4 != 0:
        T_latent = (action_emb.shape[1] if action_emb is not None else T_latent)
    ph, pw = (patch_size[1], patch_size[2]) if len(patch_size) >= 2 else (2, 2)
    h, w = height // 16, width // 16
    T_video = T_latent * h * w
    action_packed = []
    for i in range(B):
        ac = action_seqs[i]
        if ac.dim() == 2:
            ac = ac.unsqueeze(0)
        first = ac[:, :1, :].repeat(1, 3, 1)
        packed = torch.cat([first, ac], dim=1)
        if packed.shape[1] % 4 != 0:
            packed = packed[:, : (packed.shape[1] // 4) * 4]
        Tl = packed.shape[1] // 4
        packed = packed.reshape(1, Tl, 4 * joint_dim)
        expanded = packed.expand(1, Tl, h, w, 4 * joint_dim).reshape(1, Tl * h * w, 4 * joint_dim)
        action_packed.append(expanded)
    action_for_student = torch.cat(action_packed, dim=0).to(device, dtype=dtype)

    conditional_dict = {
        "context": context,
        "action_emb": action_emb,
        "clip_feature": clip_feature,
        "y": y,
        "action_for_student": action_for_student,
        "prompts": prompts,
        "action_seqs": action_seqs,
    }
    uncond_context = run_units_single(
        pipe, "", action_seqs[0:1].to(pipe.device), height, width, num_frames
    )["context"]
    if uncond_context is not None:
        uncond_context = uncond_context.expand(B, -1, -1).to(device, dtype=dtype)
    unconditional_dict = {
        "context": uncond_context,
        "action_emb": None,
        "clip_feature": None,
        "y": None,
    }
    return conditional_dict, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_dmd.yaml")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--generator_ckpt", type=str)
    parser.add_argument("--num_frame_per_block", type=int, default=1)
    parser.add_argument("--joint_dim", type=int, default=14)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--image_or_video_shape", type=str, default="1,21,16,30,52")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--backward_simulation", action="store_true", default=True)
    args = parser.parse_args()

    if os.path.isfile(getattr(args, "config", "")):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_dir = getattr(args, "model_dir", None)
    args.denoising_step_list = getattr(
        args, "denoising_step_list",
        list(range(50)),
    )
    args.num_train_timestep = getattr(args, "num_train_timestep", 1000)
    args.real_guidance_scale = getattr(args, "real_guidance_scale", 1.0)

    dmd = DMD(args, device)
    dmd.generator.train()
    gen_opt = torch.optim.AdamW(
        [p for p in dmd.generator.parameters() if p.requires_grad],
        lr=getattr(args, "lr", 1e-4),
        betas=(getattr(args, "beta1", 0.9), getattr(args, "beta2", 0.999)),
    )
    critic_opt = torch.optim.AdamW(
        [p for p in dmd.pipe.dit.parameters() if p.requires_grad],
        lr=getattr(args, "lr", 1e-4),
        betas=(getattr(args, "beta1", 0.9), getattr(args, "beta2", 0.999),
    ) if getattr(args, "train_critic", False) else None

    data_dir = getattr(args, "data_dir", None)
    if data_dir and os.path.isdir(data_dir):
        dataset = ODETrajectoryDataset(data_dir, max_samples=getattr(args, "max_samples"))
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=getattr(args, "batch_size", 2),
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: {
                "prompt": [t["prompt"] for t in x],
                "action_seq": torch.stack([t["action_seq"] for t in x]),
            },
        )
        dataloader = iter(dataloader)
    else:
        dataloader = None

    output_dir = getattr(args, "output_dir", "./dmd_ckpts")
    os.makedirs(output_dir, exist_ok=True)
    max_steps = getattr(args, "max_steps", 5000)
    save_every = getattr(args, "save_every", 500)
    height = getattr(args, "height", 480)
    width = getattr(args, "width", 832)
    num_frames = getattr(args, "num_frames", 81)
    shape_str = getattr(args, "image_or_video_shape", "1,21,16,30,52")
    image_or_video_shape = [int(x) for x in shape_str.split(",")]
    patch_size = getattr(dmd.pipe.dit, "patch_size", (2, 2, 2)) or (2, 2, 2)
    joint_dim = getattr(args, "joint_dim", 14)

    for step in range(max_steps):
        if dataloader is not None:
            try:
                batch = next(dataloader)
            except StopIteration:
                dataloader = iter(dataloader)
                batch = next(dataloader)
        else:
            B = 1
            batch = {
                "prompt": ["a cat walking"],
                "action_seq": torch.randn(B, num_frames, joint_dim, device=device),
            }

        image_or_video_shape[0] = len(batch["prompt"])
        with torch.no_grad():
            cond, uncond = build_conditional_dict(
                dmd.pipe,
                batch["prompt"],
                batch["action_seq"],
                height,
                width,
                num_frames,
                device,
                dmd.dtype,
                joint_dim,
                patch_size,
            )

        train_gen = (getattr(args, "dfake_gen_update_ratio", 1) and step % getattr(args, "dfake_gen_update_ratio", 1) == 0)
        if train_gen:
            gen_loss, gen_log = dmd.generator_loss(
                image_or_video_shape,
                cond,
                uncond,
                clean_latent=None,
                height=height,
                width=width,
                num_frames=num_frames,
            )
            gen_opt.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(dmd.generator.parameters(), 1.0)
            gen_opt.step()
            if step % 100 == 0:
                print(f"step {step} gen_loss={gen_loss.item():.6f} {gen_log}")

        if critic_opt is not None:
            crit_loss, crit_log = dmd.critic_loss(
                image_or_video_shape,
                cond,
                uncond,
                clean_latent=None,
                height=height,
                width=width,
                num_frames=num_frames,
            )
            critic_opt.zero_grad()
            crit_loss.backward()
            critic_opt.step()
            if step % 100 == 0:
                print(f"step {step} crit_loss={crit_loss.item():.6f}")

        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"generator_{step+1:06d}.pt")
            torch.save(dmd.generator.state_dict(), ckpt_path)
            print(f"Saved {ckpt_path}")

    print("DMD training done.")


if __name__ == "__main__":
    main()
