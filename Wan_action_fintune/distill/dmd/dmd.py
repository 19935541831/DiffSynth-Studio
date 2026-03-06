"""
DMD (Distribution Matching Distillation) for action-conditioned causal student.
Generator = CausalWanVideoActionDiT, real_score = fake_score = teacher (WanVideoActionDiT via pipeline model_fn).
Backward simulation is action-conditioned via backward_simulation.inference_with_trajectory.
"""

import os
import sys
from typing import Tuple, Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from diffsynth.models.causal_wan_video_action_dit import CausalWanVideoActionDiT
from diffsynth.diffusion import FlowMatchScheduler

from . import backward_simulation


class DMD(nn.Module):
    """
    DMD with generator=causal student, real_score=fake_score=teacher (via pipeline model_fn).
    _consistency_backward_simulation uses action-conditioned backward_simulation.
    """

    def __init__(self, args, device: torch.device, pipe=None):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if getattr(args, "mixed_precision", False) else torch.float32
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.real_guidance_scale = getattr(args, "real_guidance_scale", 1.0)
        self.denoising_step_list = getattr(args, "denoising_step_list", None)
        self.num_train_timestep = getattr(args, "num_train_timestep", 1000)
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.generator_task_type = getattr(args, "generator_task_type", "causal_video")
        self.real_task_type = getattr(args, "real_task_type", "causal_video")
        self.fake_task_type = getattr(args, "fake_task_type", "causal_video")

        if pipe is None:
            from diffsynth.pipelines.wan_video import WanVideoPipeline
            from diffsynth.core import ModelConfig
            self.pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=self.dtype,
                device=device,
                model_configs=[ModelConfig(args.model_dir)] if getattr(args, "model_dir", None) else [],
            )
        else:
            self.pipe = pipe

        self.pipe.scheduler.set_timesteps(
            getattr(args, "num_inference_steps", 50),
            denoising_strength=1.0,
            shift=getattr(args, "sigma_shift", 5.0),
        )
        if self.denoising_step_list is None:
            self.denoising_step_list = torch.arange(
                len(self.pipe.scheduler.timesteps),
                device=device,
                dtype=torch.long,
            )
        else:
            self.denoising_step_list = torch.tensor(
                self.denoising_step_list, dtype=torch.long, device=device
            )

        bl0 = self.pipe.dit.blocks[0]
        dit_config = {
            "dim": self.pipe.dit.dim,
            "in_dim": self.pipe.dit.in_dim,
            "ffn_dim": getattr(bl0, "ffn_dim", 8192),
            "out_dim": getattr(self.pipe.dit, "out_dim", 16),
            "text_dim": getattr(self.pipe.dit, "text_dim", 4096),
            "freq_dim": self.pipe.dit.freq_dim,
            "eps": getattr(bl0.norm1, "eps", 1e-6),
            "patch_size": self.pipe.dit.patch_size,
            "num_heads": self.pipe.dit.num_heads,
            "num_layers": len(self.pipe.dit.blocks),
            "has_image_input": self.pipe.dit.has_image_input,
            "action_dim": getattr(args, "action_dim", 4 * getattr(args, "joint_dim", 14)),
            "action_embed_hidden": 512,
            "seperated_timestep": True,
            "require_vae_embedding": getattr(self.pipe.dit, "require_vae_embedding", True),
            "require_clip_embedding": getattr(self.pipe.dit, "require_clip_embedding", True),
        }
        self.generator = CausalWanVideoActionDiT(
            num_frame_per_block=self.num_frame_per_block, **dit_config
        )
        if getattr(args, "generator_ckpt", None):
            from diffsynth.core import load_state_dict
            ckpt = load_state_dict(args.generator_ckpt, torch_dtype=self.dtype, device="cpu")
            self.generator.load_state_dict(ckpt, strict=False)
        else:
            own = self.generator.state_dict()
            for k, v in self.pipe.dit.state_dict().items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v.to(own[k].dtype))
        self.generator = self.generator.to(device)
        for n, p in self.generator.named_parameters():
            p.requires_grad = getattr(args, "generator_grad", True)

        self.scheduler = FlowMatchScheduler("Wan")
        self.scheduler.set_timesteps(
            getattr(args, "num_inference_steps", 50),
            denoising_strength=1.0,
            shift=getattr(args, "sigma_shift", 5.0),
        )
        self.scheduler.alphas_cumprod = getattr(self.scheduler, "alphas_cumprod", None)

    def _process_timestep(self, timestep: torch.Tensor, task_type: str) -> torch.Tensor:
        if task_type == "bidirectional_video":
            timestep = timestep[:, 0:1].expand_as(timestep)
        elif task_type == "causal_video":
            B, F = timestep.shape
            timestep = timestep.reshape(B, -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(B, -1)
        return timestep

    def _teacher_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        timestep_t = timestep.unsqueeze(0) if timestep.dim() == 1 else timestep
        if timestep_t.shape[0] == 1 and latents.shape[0] > 1:
            timestep_t = timestep_t.expand(latents.shape[0], -1)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        kwargs = dict(
            **models,
            latents=latents,
            timestep=timestep_t.to(self.device, dtype=self.dtype),
            context=conditional_dict["context"],
            clip_feature=conditional_dict.get("clip_feature"),
            y=conditional_dict.get("y"),
            action_emb=conditional_dict.get("action_emb"),
        )
        return self.pipe.model_fn(**kwargs)

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        pred_fake = self._teacher_forward(
            noisy_image_or_video, timestep, conditional_dict
        )
        pred_real_cond = self._teacher_forward(
            noisy_image_or_video, timestep, conditional_dict
        )
        pred_real_uncond = self._teacher_forward(
            noisy_image_or_video, timestep, unconditional_dict
        )
        pred_real = pred_real_cond + (
            pred_real_cond - pred_real_uncond
        ) * self.real_guidance_scale
        grad = pred_fake - pred_real
        if normalization:
            p_real = estimated_clean_image_or_video - pred_real
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-6)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        return grad, {
            "dmdtrain_clean_latent": estimated_clean_image_or_video.detach(),
            "dmdtrain_noisy_latent": noisy_image_or_video.detach(),
            "dmdtrain_pred_real_image": pred_real.detach(),
            "dmdtrain_pred_fake_image": pred_fake.detach(),
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        batch_size, num_frame = image_or_video.shape[:2]
        with torch.no_grad():
            timestep = torch.randint(
                0, self.num_train_timestep,
                (batch_size, num_frame),
                device=self.device,
                dtype=torch.long,
            )
            timestep = self._process_timestep(timestep, self.real_task_type)
            if self.timestep_shift != 1.0:
                timestep = (
                    self.timestep_shift * (timestep.float() / 1000.0)
                    / (1 + (self.timestep_shift - 1) * (timestep.float() / 1000.0))
                    * 1000
                ).long()
            timestep = timestep.clamp(self.min_step, self.max_step)
            noise = torch.randn_like(image_or_video, device=self.device, dtype=self.dtype)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=image_or_video,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )
        target = (image_or_video - grad).detach()
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                image_or_video.double()[gradient_mask],
                target.double()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                image_or_video.double(),
                target.double(),
                reduction="mean",
            )
        return dmd_loss.to(image_or_video.dtype), dmd_log_dict

    @torch.no_grad()
    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        conditional_dict: Dict[str, Any],
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        num_inference_steps = num_inference_steps or len(self.pipe.scheduler.timesteps)
        prompts = conditional_dict["prompts"]
        action_seqs = conditional_dict["action_seqs"]
        B = noise.shape[0]
        trajectories = []
        for i in range(B):
            traj = backward_simulation.inference_with_trajectory(
                self.pipe,
                noise[i : i + 1],
                prompts[i] if isinstance(prompts[i], str) else prompts[i][0],
                action_seqs[i],
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                cfg_scale=1.0,
            )
            trajectories.append(traj)
        return torch.cat(trajectories, dim=0)

    def _run_generator(
        self,
        image_or_video_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_latent: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = image_or_video_shape[0]
        if getattr(self.args, "backward_simulation", True):
            noise = torch.randn(
                image_or_video_shape,
                device=self.device,
                dtype=self.dtype,
            )
            simulated = self._consistency_backward_simulation(
                noise,
                conditional_dict,
                height=height,
                width=width,
                num_frames=num_frames,
            )
        else:
            steps = self.denoising_step_list
            simulated = []
            for t in steps:
                t_val = self.scheduler.timesteps[t.item()] if t.item() < len(self.scheduler.timesteps) else self.scheduler.timesteps[-1]
                noise = torch.randn(image_or_video_shape, device=self.device, dtype=self.dtype)
                timestep_flat = t_val * torch.ones(
                    image_or_video_shape[0] * image_or_video_shape[1],
                    device=self.device,
                    dtype=torch.long,
                )
                if t_val != 0 and clean_latent is not None:
                    noisy = self.scheduler.add_noise(
                        clean_latent.flatten(0, 1),
                        noise.flatten(0, 1),
                        timestep_flat,
                    ).unflatten(0, image_or_video_shape[:2])
                else:
                    noisy = clean_latent if clean_latent is not None else noise
                simulated.append(noisy)
            simulated = torch.stack(simulated, dim=1)

        num_steps_plus = simulated.shape[1]
        index = torch.randint(
            0, num_steps_plus,
            (B, image_or_video_shape[1]),
            device=self.device,
            dtype=torch.long,
        )
        index = self._process_timestep(index, self.generator_task_type)
        idx_exp = index.reshape(B, 1, index.shape[1], 1, 1, 1).expand(
            -1, -1, -1, *image_or_video_shape[2:]
        )
        noisy_input = torch.gather(simulated, dim=1, index=idx_exp).squeeze(1)
        step_timesteps = torch.cat([
            self.scheduler.timesteps.to(self.device),
            torch.zeros(1, device=self.device, dtype=self.scheduler.timesteps.dtype),
        ], dim=0)
        timestep = step_timesteps[index]

        action_student = conditional_dict["action_for_student"]
        pred = self.generator(
            noisy_input,
            timestep,
            conditional_dict["context"],
            action_student,
            clip_feature=conditional_dict.get("clip_feature"),
            y=conditional_dict.get("y"),
            kv_cache=None,
        )
        return pred.type_as(noisy_input), None

    def generator_loss(
        self,
        image_or_video_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_latent: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
    ) -> Tuple[torch.Tensor, dict]:
        pred_image, gradient_mask = self._run_generator(
            image_or_video_shape,
            conditional_dict,
            unconditional_dict,
            clean_latent=clean_latent,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            pred_image,
            conditional_dict,
            unconditional_dict,
            gradient_mask=gradient_mask,
        )
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape: List[int],
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        clean_latent: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
    ) -> Tuple[torch.Tensor, dict]:
        with torch.no_grad():
            generated, _ = self._run_generator(
                image_or_video_shape,
                conditional_dict,
                unconditional_dict,
                clean_latent=clean_latent,
                height=height,
                width=width,
                num_frames=num_frames,
            )
        batch_size, num_frame = image_or_video_shape[0], image_or_video_shape[1]
        critic_timestep = torch.randint(
            0, self.num_train_timestep,
            (batch_size, num_frame),
            device=self.device,
            dtype=torch.long,
        )
        critic_timestep = self._process_timestep(critic_timestep, self.fake_task_type)
        if self.timestep_shift != 1.0:
            critic_timestep = (
                self.timestep_shift * (critic_timestep.float() / 1000.0)
                / (1 + (self.timestep_shift - 1) * (critic_timestep.float() / 1000.0))
                * 1000
            ).long()
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)
        critic_noise = torch.randn_like(generated, device=self.device, dtype=self.dtype)
        noisy_generated = self.scheduler.add_noise(
            generated.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))
        pred_fake = self._teacher_forward(
            noisy_generated, critic_timestep, conditional_dict
        )
        denoising_loss = F.mse_loss(pred_fake, generated, reduction="mean")
        critic_log_dict = {
            "critictrain_latent": generated.detach(),
            "critictrain_noisy_latent": noisy_generated.detach(),
            "critictrain_pred_image": pred_fake.detach(),
            "critic_timestep": critic_timestep.detach(),
        }
        return denoising_loss, critic_log_dict
