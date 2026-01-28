# wan_video_action_dit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange

from .wan_video_dit import WanModel, sinusoidal_embedding_1d


class WanVideoActionDiT(WanModel):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        action_dim: int = 8,          # e.g., 7 joints + gripper
        action_embed_hidden: int = 512,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = True,  # MUST be True for per-frame action
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        # Enforce per-frame modulation
        assert seperated_timestep, "Per-frame action requires seperated_timestep=True"

        super().__init__(
            dim=dim,
            in_dim=in_dim,
            ffn_dim=ffn_dim,
            out_dim=out_dim,
            text_dim=text_dim,
            freq_dim=freq_dim,
            eps=eps,
            patch_size=patch_size,
            num_heads=num_heads,
            num_layers=num_layers,
            has_image_input=has_image_input,
            has_image_pos_emb=has_image_pos_emb,
            has_ref_conv=has_ref_conv,
            add_control_adapter=add_control_adapter,
            in_dim_control_adapter=in_dim_control_adapter,
            seperated_timestep=seperated_timestep,
            require_vae_embedding=require_vae_embedding,
            require_clip_embedding=require_clip_embedding,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )

        # Action embedding: project per-frame action to modulation space
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, action_embed_hidden),
            nn.GELU(),
            nn.Linear(action_embed_hidden, dim)
        )
        self.action_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)  # 6 parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,  # NEW: (B, T_video, D_action), T_video = F*H*W
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        # === Step 1: Time embedding (per-frame) ===
        # Original WanModel uses global timestep → we expand to per-frame
        t_emb_global = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )  # (B, dim)
        
        # Patchify to get spatial-temporal grid size
        x_patched, (f, h, w) = self.patchify(x)  # x_patched: (B, dim, f, h, w)
        T_video = f * h * w

        # Expand global time embedding to all tokens
        t_emb = t_emb_global.unsqueeze(1).expand(-1, T_video, -1)  # (B, T, dim)

        # Project to modulation: (B, T, 6*dim) → (B, T, 6, dim)
        t_mod = self.time_projection(t_emb).unflatten(-1, (6, self.dim))  # (B, T, 6, dim)

        # === Step 2: Action embedding (per-frame) ===
        assert action.shape[1] == T_video and action.shape[2] == self.action_embedding[0].in_features, \
            f"Expected action shape (B, {T_video}, {self.action_embedding[0].in_features}), got {action.shape}"
        
        a_emb = self.action_embedding(action)  # (B, T, dim)
        a_mod = self.action_projection(a_emb).unflatten(-1, (6, self.dim))  # (B, T, 6, dim)

        # === Step 3: Combine time + action modulation ===
        total_mod = t_mod + a_mod  # (B, T, 6, dim)

        # === Step 4: Process context (text + optional image) ===
        context = self.text_embedding(context)
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_emb = self.img_emb(clip_feature)
            context = torch.cat([clip_emb, context], dim=1)

        # Re-patchify (since we used it above only for T_video)
        x, (f, h, w) = self.patchify(x)

        # === Step 5: RoPE frequencies ===
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        # === Step 6: Forward through blocks ===
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, total_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, total_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, total_mod, freqs)

        # === Step 7: Output head ===
        # Note: Head expects global time embedding (not per-frame)
        x = self.head(x, t_emb_global)  # Use global t_emb for final modulation
        x = self.unpatchify(x, (f, h, w))
        return x