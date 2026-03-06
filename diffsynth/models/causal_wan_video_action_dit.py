# causal_wan_video_action_dit.py
# Causal student for Wan action-conditioned model: block-wise causal self-attention + KV cache for inference.

import math
from typing import Tuple, Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .wan_video_dit import (
    WanModel,
    sinusoidal_embedding_1d,
    flash_attention,
    modulate,
    GateModule,
    RMSNorm,
    CrossAttention,
    Head,
    precompute_freqs_cis_3d,
    rope_apply,
)
from .wan_video_action_dit import WanVideoActionDiT


def causal_rope_apply(
    x: torch.Tensor,
    freqs_3d: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    num_heads: int,
    f: int,
    h: int,
    w: int,
    start_frame: int = 0,
) -> torch.Tensor:
    """Apply RoPE with temporal offset start_frame for causal inference (current block only)."""
    f_freqs, h_freqs, w_freqs = freqs_3d
    # x: (B, S, n, d) or (B, S, dim) -> after reshape (B, S, n, d)
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    B, S, n, d = x.shape
    half_d = d // 2
    # freqs per dim: f has (f,), h has (h,), w has (w,) -> broadcast to (f,h,w)
    device = x.device
    f_f = f_freqs[start_frame : start_frame + f].to(device)  # (f, d_f)
    h_f = h_freqs[:h].to(device)
    w_f = w_freqs[:w].to(device)
    # Build freqs for each of the f*h*w positions: (f*h*w, d)
    # f_freqs dim, h_freqs dim, w_freqs dim
    d_f, d_h, d_w = f_f.shape[1], h_f.shape[1], w_f.shape[1]
    freqs_f = f_f.view(f, 1, 1, -1).expand(f, h, w, -1).reshape(-1, d_f)
    freqs_h = h_f.view(1, h, 1, -1).expand(f, h, w, -1).reshape(-1, d_h)
    freqs_w = w_f.view(1, 1, w, -1).expand(f, h, w, -1).reshape(-1, d_w)
    freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1)  # (S, d)
    freqs = freqs.to(torch.complex64) if device.type != "npu" else freqs
    x_c = torch.view_as_complex(
        x.to(torch.float64).reshape(B, S, n, -1, 2)
    )
    freqs = freqs.unsqueeze(0).unsqueeze(2)  # (1, S, 1, d)
    x_out = torch.view_as_real(x_c * freqs).flatten(2)
    return x_out.to(x.dtype)


def build_block_causal_mask(
    total_len: int,
    block_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build block-wise causal mask: position i can attend to j iff j is in the same or earlier block.
    Returns (1, 1, total_len, total_len) mask; 0 = attend, -inf = mask out.
    """
    num_blocks = (total_len + block_len - 1) // block_len
    mask = torch.zeros(total_len, total_len, device=device, dtype=dtype)
    for i in range(total_len):
        block_i = i // block_len
        for j in range(total_len):
            block_j = j // block_len
            if block_j > block_i:
                mask[i, j] = float("-inf")
    return mask.unsqueeze(0).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    """Self-attention with block-wise causal mask (train) or KV cache (inference)."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_3d: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        f: Optional[int] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        current_end: int = 0,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        if kv_cache is not None:
            # Inference: causal RoPE for current block only; write k,v to cache; attend to cache
            assert freqs_3d is not None and f is not None and h is not None and w is not None
            start_f = current_start // (h * w)
            q = causal_rope_apply(q, freqs_3d, self.num_heads, f, h, w, start_frame=start_f)
            k = causal_rope_apply(k, freqs_3d, self.num_heads, f, h, w, start_frame=start_f)
            kv_cache["k"][:, current_start:current_end] = k
            kv_cache["v"][:, current_start:current_end] = v
            k_full = kv_cache["k"][:, :current_end]
            v_full = kv_cache["v"][:, :current_end]
            # q (B, block_S, n, d), k_full (B, current_end, n, d) -> causal SDPA
            q = rearrange(q, "b s n d -> b n s d", n=self.num_heads)
            k_full = rearrange(k_full, "b s n d -> b n s d", n=self.num_heads)
            v_full = rearrange(v_full, "b s n d -> b n s d", n=self.num_heads)
            x_out = F.scaled_dot_product_attention(
                q, k_full, v_full, attn_mask=None, is_causal=True, dropout_p=0.0
            )
            x_out = rearrange(x_out, "b n s d -> b s (n d)", n=self.num_heads)
        else:
            # Train: full RoPE, then block causal mask
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            if block_mask is not None:
                q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
                k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
                v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
                x_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=block_mask, is_causal=False, dropout_p=0.0
                )
                x_out = rearrange(x_out, "b n s d -> b s (n d)", n=self.num_heads)
            else:
                x_out = flash_attention(q, k, v, num_heads=self.num_heads)

        return self.o(x_out)


class CausalDiTBlock(nn.Module):
    """DiT block with CausalSelfAttention; same CrossAttention and FFN as DiTBlock."""

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = CausalSelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        freqs_3d: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        f: Optional[int] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        current_start: int = 0,
        current_end: int = 0,
        block_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            shift_mlp = shift_mlp.squeeze(2)
            scale_mlp = scale_mlp.squeeze(2)
            gate_mlp = gate_mlp.squeeze(2)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(
            x,
            gate_msa,
            self.self_attn(
                input_x,
                freqs,
                freqs_3d=freqs_3d,
                f=f,
                h=h,
                w=w,
                kv_cache=kv_cache,
                current_start=current_start,
                current_end=current_end,
                block_mask=block_mask,
            ),
        )
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class CausalWanVideoActionDiT(WanVideoActionDiT):
    """Causal Wan action DiT: same as WanVideoActionDiT but with CausalDiTBlock and KV-cache inference."""

    def __init__(self, *args, num_frame_per_block: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        num_layers = len(self.blocks)
        first = self.blocks[0]
        ffn_dim = getattr(first, "ffn_dim", kwargs.get("ffn_dim", 8192))
        eps = getattr(first.norm1, "eps", 1e-6)
        self.num_frame_per_block = num_frame_per_block
        self.blocks = nn.ModuleList([
            CausalDiTBlock(
                self.has_image_input,
                self.dim,
                self.num_heads,
                ffn_dim,
                eps,
            )
            for _ in range(num_layers)
        ])
        self.block_mask = None

    def _get_patchify_flat_and_grid(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, (f, h, w)

    def _prepare_block_causal_mask(self, total_len: int, block_len: int, device: torch.device, dtype: torch.dtype):
        return build_block_causal_mask(total_len, block_len, device, dtype)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        kv_cache: Optional[List[Dict[str, torch.Tensor]]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        if kv_cache is not None:
            return self._forward_inference(
                x, timestep, context, action, clip_feature, y,
                kv_cache, current_start, current_end,
            )
        return self._forward_train(
            x, timestep, context, action, clip_feature, y,
            use_gradient_checkpointing, use_gradient_checkpointing_offload,
        )

    def _forward_train(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        t_emb_global = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
        x, (f, h, w) = self._get_patchify_flat_and_grid(x)
        T_video = f * h * w

        t_emb = t_emb_global.unsqueeze(1).expand(-1, T_video, -1)
        t_mod = self.time_projection(t_emb).unflatten(-1, (6, self.dim))

        assert action.shape[1] == T_video and action.shape[2] == self.action_embedding[0].in_features
        a_emb = self.action_embedding(action)
        a_mod = self.action_projection(a_emb).unflatten(-1, (6, self.dim))
        total_mod = t_mod + a_mod

        context = self.text_embedding(context)
        if self.has_image_input:
            clip_emb = self.img_emb(clip_feature)
            context = torch.cat([clip_emb, context], dim=1)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        block_len = self.num_frame_per_block * h * w
        if self.block_mask is None or self.block_mask.shape[2] != T_video:
            self.block_mask = self._prepare_block_causal_mask(
                T_video, block_len, x.device, x.dtype
            )

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
                            None, f, h, w, None, 0, 0, self.block_mask,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, total_mod, freqs,
                        None, f, h, w, None, 0, 0, self.block_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x, context, total_mod, freqs,
                    freqs_3d=(self.freqs[0], self.freqs[1], self.freqs[2]),
                    f=f, h=h, w=w,
                    kv_cache=None,
                    current_start=0,
                    current_end=T_video,
                    block_mask=self.block_mask,
                )

        x = self.head(x, t_emb_global)
        x = self.unpatchify(x, (f, h, w))
        return x

    def _forward_inference(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        action: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        kv_cache: List[Dict[str, torch.Tensor]],
        current_start: int,
        current_end: int,
    ) -> torch.Tensor:
        t_emb_global = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype)
        )
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
        x, (f, h, w) = self._get_patchify_flat_and_grid(x)
        block_T = f * h * w

        t_emb = t_emb_global.unsqueeze(1).expand(-1, block_T, -1)
        t_mod = self.time_projection(t_emb).unflatten(-1, (6, self.dim))

        action_block = action[:, current_start:current_end]
        assert action_block.shape[1] == block_T
        a_emb = self.action_embedding(action_block)
        a_mod = self.action_projection(a_emb).unflatten(-1, (6, self.dim))
        total_mod = t_mod + a_mod

        context = self.text_embedding(context)
        if self.has_image_input:
            clip_emb = self.img_emb(clip_feature)
            context = torch.cat([clip_emb, context], dim=1)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        freqs_3d = (self.freqs[0], self.freqs[1], self.freqs[2])

        for block_index, block in enumerate(self.blocks):
            layer_cache = kv_cache[block_index] if block_index < len(kv_cache) else None
            x = block(
                x, context, total_mod, freqs,
                freqs_3d=freqs_3d,
                f=f, h=h, w=w,
                kv_cache=layer_cache,
                current_start=current_start,
                current_end=current_end,
                block_mask=None,
            )

        x = self.head(x, t_emb_global)
        x = self.unpatchify(x, (f, h, w))
        return x

    @classmethod
    def from_teacher_state_dict(
        cls,
        teacher_state_dict: Dict[str, Any],
        num_frame_per_block: int = 1,
        strict: bool = False,
        **model_kwargs,
    ) -> "CausalWanVideoActionDiT":
        """Build CausalWanVideoActionDiT and load compatible weights from teacher (WanVideoActionDiT).
        model_kwargs must contain all arguments required for WanVideoActionDiT.__init__.
        """
        model = cls(num_frame_per_block=num_frame_per_block, **model_kwargs)
        own = model.state_dict()
        loaded = 0
        for k, v in teacher_state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded += 1
        if strict:
            assert loaded == len(own), "Some params not loaded from teacher"
        return model
