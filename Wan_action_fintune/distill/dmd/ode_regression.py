"""
ODE regression: train student (CausalWanVideoActionDiT) to match clean latent
from teacher ODE trajectory. Randomly sample (noisy_latent, timestep) from trajectory,
student predicts x0, MSE loss to clean.
"""

import os
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffsynth.models.causal_wan_video_action_dit import CausalWanVideoActionDiT
from diffsynth.diffusion import FlowMatchScheduler


class ODERegression(nn.Module):
    """
    Train causal student on teacher ODE trajectories.
    Supports causal_video: timestep same within each block.
    """

    def __init__(self, args, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if getattr(args, "mixed_precision", False) else torch.float32
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        from diffsynth.pipelines.wan_video import WanVideoPipeline
        from diffsynth.core import ModelConfig
        model_dir = getattr(args, "model_dir", None)
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self.dtype,
            device=device,
            model_configs=[ModelConfig(model_dir)] if model_dir else [],
        )
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
            "action_dim": getattr(args, "action_dim", 4 * getattr(args, "joint_dim", 14)),
            "action_embed_hidden": 512,
            "seperated_timestep": True,
            "require_vae_embedding": getattr(pipe.dit, "require_vae_embedding", True),
            "require_clip_embedding": getattr(pipe.dit, "require_clip_embedding", True),
        }
        self.generator = CausalWanVideoActionDiT(num_frame_per_block=self.num_frame_per_block, **dit_config)
        if getattr(args, "generator_ckpt", None):
            from diffsynth.core import load_state_dict
            ckpt = load_state_dict(args.generator_ckpt, torch_dtype=self.dtype, device="cpu")
            self.generator.load_state_dict(ckpt, strict=False)
        else:
            own = self.generator.state_dict()
            for k, v in pipe.dit.state_dict().items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v.to(own[k].dtype))
        self.generator = self.generator.to(device)
        if getattr(args, "generator_grad", None):
            for n, p in self.generator.named_parameters():
                p.requires_grad = n in args.generator_grad if isinstance(args.generator_grad, (list, set)) else True

        self.text_encoder = pipe.text_encoder.to(device)
        self.text_encoder.requires_grad_(False)
        self.tokenizer = pipe.tokenizer
        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(getattr(args, "num_ode_steps", 50))
        self.denoising_step_list = torch.arange(
            len(self.scheduler.timesteps),
            device=device,
            dtype=torch.long,
        )
        self.generator_task = getattr(args, "generator_task", "causal_video")

    @torch.no_grad()
    def encode_prompts(self, prompts: list) -> torch.Tensor:
        ids, mask = self.tokenizer(prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        context = self.text_encoder(ids, mask)
        if mask is not None:
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, v in enumerate(seq_lens):
                context[i, v:] = 0
        return context

    def _process_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.generator_task == "bidirectional_video":
            timestep = timestep[:, 0:1].expand_as(timestep)
        elif self.generator_task == "causal_video":
            B, F = timestep.shape
            timestep = timestep.reshape(B, -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(B, -1)
        return timestep

    @torch.no_grad()
    def _prepare_generator_input(
        self,
        ode_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ode_latent: (B, num_steps+1, F, C, H, W).
        Returns noisy_input (B, F, C, H, W), timestep (B, F).
        """
        B, num_steps_plus, F, C, H, W = ode_latent.shape
        num_steps = num_steps_plus - 1
        index = torch.randint(
            0, num_steps_plus,
            (B,),
            device=self.device,
            dtype=torch.long,
        )
        noisy_input = ode_latent[torch.arange(B, device=self.device), index]
        timestep_ids = index.unsqueeze(1).expand(B, F)
        timestep_ids = self._process_timestep(timestep_ids)
        timesteps = self.scheduler.timesteps.to(self.device)
        timestep = timesteps[timestep_ids.clamp(0, len(timesteps) - 1)]
        return noisy_input, timestep

    def generator_loss(
        self,
        ode_latent: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        ode_latent: (B, num_steps+1, F, C, H, W)
        context: (B, L, text_dim)
        action: (B, T_tokens, action_dim)
        """
        target_latent = ode_latent[:, -1]
        noisy_input, timestep = self._prepare_generator_input(ode_latent)
        pred = self.generator(
            noisy_input,
            timestep,
            context,
            action,
            clip_feature=clip_feature,
            y=y,
            kv_cache=None,
        )
        loss = F.mse_loss(pred, target_latent, reduction="mean")
        log_dict = {
            "ode_timestep_mean": timestep.float().mean().detach(),
            "ode_loss": loss.detach(),
        }
        return loss, log_dict
