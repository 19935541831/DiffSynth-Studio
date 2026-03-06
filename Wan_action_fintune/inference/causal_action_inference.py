#!/usr/bin/env python3
"""
Causal action-conditioned inference using CausalWanVideoActionDiT with KV cache.
Run from repo root (3.3_diffsynth_act_cond) so that diffsynth imports work.

Usage:
  python Wan_action_fintune/inference/causal_action_inference.py \
    --model_dir <base_wan_dir> \
    --causal_checkpoint <path_to_causal_dit.pt> \
    --prompt "A robot arm picks up a block" \
    --action_seq <path_to_action.npy> \
    --input_image <first_frame.png> \
    --output_path out.mp4
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

# Add repo root for diffsynth imports
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffsynth.models.causal_wan_video_action_dit import CausalWanVideoActionDiT
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.diffusion import FlowMatchScheduler


def _initialize_kv_cache(
    batch_size: int,
    num_blocks: int,
    max_T: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> List[dict]:
    """Allocate per-layer KV cache. Each layer: {"k": (B, max_T, nH, d), "v": (B, max_T, nH, d)}."""
    kv_cache = []
    for _ in range(num_blocks):
        kv_cache.append({
            "k": torch.zeros((batch_size, max_T, num_heads, head_dim), dtype=dtype, device=device),
            "v": torch.zeros((batch_size, max_T, num_heads, head_dim), dtype=dtype, device=device),
        })
    return kv_cache


class CausalActionInferencePipeline:
    """Minimal pipeline for causal action-conditioned video generation with KV cache."""

    def __init__(
        self,
        dit: CausalWanVideoActionDiT,
        vae: torch.nn.Module,
        text_encoder: torch.nn.Module,
        action_encoder: Optional[torch.nn.Module] = None,
        scheduler: FlowMatchScheduler = None,
        tokenizer=None,
        image_encoder: Optional[torch.nn.Module] = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dit = dit.to(device)
        self.vae = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.action_encoder = action_encoder.to(device) if action_encoder is not None else None
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.image_encoder = image_encoder.to(device) if image_encoder is not None else None
        self.device = device or next(dit.parameters()).device
        self.dtype = dtype
        self.num_frame_per_block = getattr(dit, "num_frame_per_block", 1)
        self.num_blocks_layers = len(dit.blocks)

    def _encode_prompt(self, prompts: List[str], tokenizer) -> torch.Tensor:
        """Return context (B, L, text_dim) for dit.text_embedding."""
        ids, mask = tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    def _expand_action_to_tokens(
        self,
        action_seq: torch.Tensor,
        T_tokens: int,
    ) -> torch.Tensor:
        """Expand action (B, T_in, action_dim) to (B, T_tokens, action_dim) by repeating."""
        B, T_in, D = action_seq.shape
        if T_in >= T_tokens:
            return action_seq[:, :T_tokens]
        repeats = (T_tokens + T_in - 1) // T_in
        out = action_seq.repeat(1, repeats, 1)[:, :T_tokens]
        return out

    def inference(
        self,
        noise: torch.Tensor,
        prompts: List[str],
        action_seq: torch.Tensor,
        input_image: Optional[torch.Tensor] = None,
        clip_feature: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        noise: (B, F, C, H, W)
        action_seq: (B, T_frame, action_dim) e.g. (B, 17, 56) for 4*14
        Returns video (B, F, C, H, W) in [0,1].
        """
        self.dit.eval()
        self.vae.eval()
        batch_size, num_frames, C, H, W = noise.shape
        device = noise.device
        dtype = noise.dtype

        with torch.no_grad():
            if self.tokenizer is not None:
                context = self._encode_prompt(prompts, self.tokenizer)
            else:
                raise RuntimeError("tokenizer is required for encoding prompts")
            context = context.to(device=device, dtype=dtype)

        patch_size = self.dit.patch_size
        f_patch = num_frames // patch_size[0]
        h_patch = (H // 8) // patch_size[1]
        w_patch = (W // 8) // patch_size[2]
        T_tokens = f_patch * h_patch * w_patch
        block_T = self.num_frame_per_block * h_patch * w_patch
        num_blocks = (T_tokens + block_T - 1) // block_T

        action_seq = action_seq.to(device=device, dtype=dtype)
        action_emb_expanded = self._expand_action_to_tokens(action_seq, T_tokens)

        max_T = T_tokens
        num_heads = self.dit.num_heads
        head_dim = self.dit.dim // num_heads
        kv_cache = _initialize_kv_cache(
            batch_size, self.num_blocks_layers, max_T, num_heads, head_dim, dtype, device
        )

        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)
        timesteps = self.scheduler.timesteps
        sigmas = self.scheduler.sigmas

        y = None
        if input_image is not None and self.dit.has_image_input:
            y = input_image.to(device=device, dtype=dtype)
        if clip_feature is not None and self.dit.require_clip_embedding:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
        else:
            clip_feature = torch.zeros(
                (batch_size, 257, 1280), device=device, dtype=dtype
            ) if self.dit.require_clip_embedding else None

        output = torch.zeros_like(noise, device=device, dtype=dtype)

        for block_index in range(num_blocks):
            start_f = block_index * self.num_frame_per_block
            end_f = min(start_f + self.num_frame_per_block, f_patch)
            current_start = block_index * block_T
            current_end = min(current_start + block_T, T_tokens)

            n_f_block = end_f - start_f
            noisy_block = noise[:, start_f:end_f]
            if noisy_block.shape[1] < n_f_block:
                pad_f = n_f_block - noisy_block.shape[1]
                noisy_block = torch.cat([
                    noisy_block,
                    torch.zeros(
                        (batch_size, pad_f, C, H, W),
                        device=device, dtype=dtype,
                    ),
                ], dim=1)

            for step_idx, current_timestep in enumerate(timesteps):
                if isinstance(current_timestep, torch.Tensor):
                    t = current_timestep.to(device)
                else:
                    t = torch.tensor(current_timestep, device=device, dtype=torch.long)
                t_batch = t.unsqueeze(0).expand(batch_size)

                pred = self.dit(
                    noisy_block,
                    t_batch,
                    context,
                    action_emb_expanded[:, current_start:current_end],
                    clip_feature=clip_feature,
                    y=y,
                    kv_cache=kv_cache,
                    current_start=current_start,
                    current_end=current_end,
                )

                if step_idx + 1 < len(timesteps):
                    next_t = timesteps[step_idx + 1]
                    if isinstance(next_t, torch.Tensor):
                        next_t = next_t.to(device)
                    else:
                        next_t = torch.tensor(next_t, device=device, dtype=torch.long)
                    sigma = sigmas[step_idx].to(device)
                    sigma_next = sigmas[step_idx + 1].to(device)
                    flow = pred
                    noisy_block = noisy_block + flow * (sigma_next - sigma)
                else:
                    denoised_block = pred

            n_keep = denoised_block.shape[1]
            output[:, start_f : start_f + n_keep] = denoised_block

            t_zero = torch.zeros(batch_size, device=device, dtype=torch.long)
            self.dit(
                denoised_block,
                t_zero,
                context,
                action_emb_expanded[:, current_start:current_end],
                clip_feature=clip_feature,
                y=y,
                kv_cache=kv_cache,
                current_start=current_start,
                current_end=current_end,
            )

        with torch.no_grad():
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)
        return video


def main():
    parser = argparse.ArgumentParser(description="Causal action-conditioned inference with KV cache")
    parser.add_argument("--model_dir", type=str, required=True, help="Base Wan model directory")
    parser.add_argument("--causal_checkpoint", type=str, default=None, help="Causal DiT checkpoint (.pt or .safetensors)")
    parser.add_argument("--prompt", type=str, default="A robot arm moves forward.")
    parser.add_argument("--action_seq", type=str, required=True, help="Path to .npy action (T, D)")
    parser.add_argument("--input_image", type=str, default=None, help="First frame image path (I2V)")
    parser.add_argument("--output_path", type=str, default="causal_out.mp4")
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--num_frame_per_block", type=int, default=1)
    parser.add_argument("--joint_dim", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.core import load_state_dict

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[ModelConfig(args.model_dir)],
    )
    pipe.load_models_to_device(["dit", "vae", "text_encoder", "image_encoder"])

    bl0 = pipe.dit.blocks[0]
    dit_config = {
        "dim": pipe.dit.dim,
        "in_dim": pipe.dit.in_dim,
        "ffn_dim": getattr(bl0, "ffn_dim", 8192),
        "out_dim": getattr(pipe.dit, "out_dim", 16),
        "text_dim": getattr(pipe.dit, "text_dim", 4096),
        "freq_dim": pipe.dit.freq_dim,
        "eps": getattr(bl0.norm1, "eps", 1e-6),
        "patch_size": pipe.dit.patch_size,
        "num_heads": pipe.dit.num_heads,
        "num_layers": len(pipe.dit.blocks),
        "has_image_input": pipe.dit.has_image_input,
        "action_dim": 4 * args.joint_dim,
        "action_embed_hidden": 512,
        "seperated_timestep": True,
        "require_vae_embedding": getattr(pipe.dit, "require_vae_embedding", True),
        "require_clip_embedding": getattr(pipe.dit, "require_clip_embedding", True),
    }

    causal_dit = CausalWanVideoActionDiT(num_frame_per_block=args.num_frame_per_block, **dit_config)
    if args.causal_checkpoint and os.path.isfile(args.causal_checkpoint):
        ckpt = load_state_dict(args.causal_checkpoint, torch_dtype=dtype, device="cpu")
        causal_dit.load_state_dict(ckpt, strict=False)
    else:
        teacher_sd = pipe.dit.state_dict()
        for k, v in teacher_sd.items():
            if k in causal_dit.state_dict() and causal_dit.state_dict()[k].shape == v.shape:
                causal_dit.state_dict()[k].copy_(v)

    action_encoder = pipe.action_encoder
    if action_encoder is None:
        from diffsynth.models.wan_video_action_encoder import WanActionEncoder
        action_encoder = WanActionEncoder(joint_dim=args.joint_dim, dit_dim=pipe.dit.dim)

    scheduler = FlowMatchScheduler("Wan")
    causal_pipe = CausalActionInferencePipeline(
        dit=causal_dit,
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        action_encoder=action_encoder,
        scheduler=scheduler,
        tokenizer=pipe.tokenizer,
        image_encoder=pipe.image_encoder,
        device=device,
        dtype=dtype,
    )

    torch.manual_seed(args.seed)
    num_frames, height, width = args.num_frames, args.height, args.width
    C = 16
    noise = torch.randn(1, num_frames, C, height // 8, width // 8, device=device, dtype=dtype)

    action_arr = np.load(args.action_seq)
    action_seq = torch.from_numpy(action_arr).float().unsqueeze(0)
    joint_dim = action_arr.shape[-1]
    if action_seq.shape[1] != num_frames:
        action_seq = torch.nn.functional.interpolate(
            action_seq.transpose(1, 2),
            size=num_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    if joint_dim == 4 * args.joint_dim:
        packed = action_seq
    else:
        first = action_seq[:, :1].repeat(1, 3, 1)
        action_seq = torch.cat([first, action_seq], dim=1)
        n_pack = action_seq.shape[1] // 4
        packed = action_seq.reshape(1, n_pack, 4 * args.joint_dim)

    input_image_t = None
    clip_feature = None
    if args.input_image and os.path.isfile(args.input_image):
        from PIL import Image
        img = Image.open(args.input_image).convert("RGB")
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        input_image_t = img.unsqueeze(0).to(device=device, dtype=dtype)
        if pipe.image_encoder is not None:
            clip_feature = pipe.image_encoder.encode_image([args.input_image]).to(device=device, dtype=dtype)

    video = causal_pipe.inference(
        noise,
        [args.prompt],
        packed,
        input_image=input_image_t,
        clip_feature=clip_feature,
        num_inference_steps=args.num_inference_steps,
    )
    video_np = video[0].cpu().numpy().transpose(0, 2, 3, 1)
    video_np = (np.clip(video_np, 0, 1) * 255).astype(np.uint8)
    try:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_h, out_w = video_np.shape[1], video_np.shape[2]
        writer = cv2.VideoWriter(args.output_path, fourcc, 8.0, (out_w, out_h))
        for i in range(video_np.shape[0]):
            writer.write(cv2.cvtColor(video_np[i], cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved {args.output_path}")
    except Exception as e:
        print(f"Save video failed: {e}. Saving frames as npy.")
        np.save(args.output_path.replace(".mp4", ".npy"), video_np)


if __name__ == "__main__":
    main()
