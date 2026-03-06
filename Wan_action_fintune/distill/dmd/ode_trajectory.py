"""
Generate ODE trajectories using the teacher (WanVideoActionDiT + pipeline).
Each trajectory is [noise, x_1, ..., x_T] (latent at each denoising step).
Saves to .pt files or returns in-memory for ODE regression training.

Usage: run from repo root. Load pipeline with teacher dit, then call
generate_trajectory_for_batch or use the pipeline with trajectory collection.
"""

import os
import sys
from typing import List, Optional, Tuple, Union, Any

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def generate_trajectory_for_sample(
    pipe,
    prompt: str,
    action_seq: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int = 50,
    input_image: Any = None,
    seed: int = 42,
    cfg_scale: float = 1.0,
) -> Tuple[torch.Tensor, str, torch.Tensor]:
    """
    Run teacher pipeline and collect full ODE trajectory by hooking scheduler.step.
    Returns:
        ode_latent: (1, num_steps+1, F, C, H, W) from noise to clean
        prompt: str
        action_seq: (1, T, D) as stored
    """
    trajectory = []

    original_step = pipe.scheduler.step
    def capturing_step(noise_pred, timestep, sample, **kwargs):
        trajectory.append(sample.clone())
        out = original_step(noise_pred, timestep, sample, **kwargs)
        trajectory.append(out.clone())
        return out

    pipe.scheduler.step = capturing_step
    try:
        pipe(
            prompt=prompt,
            action_seq=action_seq,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            input_image=input_image,
            seed=seed,
            cfg_scale=cfg_scale,
        )
    finally:
        pipe.scheduler.step = original_step

    if not trajectory:
        raise RuntimeError("No trajectory collected; pipeline may not have run denoising loop.")
    trajectory_unique = trajectory[::2] + [trajectory[-1]]
    ode_latent = torch.stack(trajectory_unique, dim=1)
    return ode_latent, prompt, action_seq


def generate_ode_trajectories_batch(
    pipe,
    prompts: List[str],
    action_seqs: List[torch.Tensor],
    num_frames: int,
    height: int,
    width: int,
    num_inference_steps: int = 50,
    output_dir: Optional[str] = None,
    input_images: Optional[List] = None,
    start_seed: int = 42,
) -> List[dict]:
    """
    Generate ODE trajectories for a batch. If output_dir is set, save each as output_dir/{i}.pt.
    Returns list of {"ode_latent", "prompt", "action_seq", "seed"}.
    """
    results = []
    for i, (prompt, action_seq) in enumerate(zip(prompts, action_seqs)):
        if action_seq.dim() == 2:
            action_seq = action_seq.unsqueeze(0)
        inp_img = input_images[i] if input_images is not None else None
        ode_latent, _, action_saved = generate_trajectory_for_sample(
            pipe,
            prompt=prompt,
            action_seq=action_seq,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            input_image=inp_img,
            seed=start_seed + i,
        )
        item = {
            "ode_latent": ode_latent.cpu(),
            "prompt": prompt,
            "action_seq": action_saved.cpu(),
            "seed": start_seed + i,
        }
        results.append(item)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(item, os.path.join(output_dir, f"{i:06d}.pt"))
    return results
