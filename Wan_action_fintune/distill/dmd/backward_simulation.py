"""
Action-conditioned backward simulation: given noise, prompt, action,
run teacher (WanVideoActionDiT) multi-step denoise to get trajectory
[B, T_steps+1, F, C, H, W] for DMD.
"""

import os
import sys
from typing import Optional, List

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def inference_with_trajectory(
    pipe,
    noise: torch.Tensor,
    prompt: str,
    action_seq: torch.Tensor,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    num_inference_steps: int = 50,
    denoising_strength: float = 1.0,
    sigma_shift: float = 5.0,
    cfg_scale: float = 1.0,
    switch_DiT_boundary: float = 0.875,
    progress_bar=None,
) -> torch.Tensor:
    """
    Run teacher denoising from given noise with prompt and action_seq.
    Returns trajectory (B, num_steps+1, F, C, H, W): [noise, x_1, ..., x_T].
    """
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": "",
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_shared = {
        "input_image": None,
        "end_image": None,
        "input_video": None,
        "denoising_strength": denoising_strength,
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
        "cfg_scale": cfg_scale,
        "cfg_merge": False,
        "sigma_shift": sigma_shift,
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
        "action_seq": action_seq,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    inputs_shared["latents"] = noise.to(pipe.device, dtype=pipe.torch_dtype)
    trajectory: List[torch.Tensor] = [noise.clone()]

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    steps = pipe.scheduler.timesteps
    if progress_bar is not None:
        steps = progress_bar(steps)

    for progress_id, timestep in enumerate(steps):
        if timestep.item() < switch_DiT_boundary * 1000 and pipe.dit2 is not None and models["dit"] is not pipe.dit2:
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2

        timestep_t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep_t)
        if cfg_scale != 1.0:
            noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep_t)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        inputs_shared["latents"] = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"]
        )
        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        trajectory.append(inputs_shared["latents"].clone())

    return torch.stack(trajectory, dim=1)
