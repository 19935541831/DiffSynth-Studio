student model(Causal)和teacher model(bidirectional)的差异汇总

### 1. 整体 forward 逻辑

| 项目 | Teacher (WanModel) | Student (CausalWanModel) |
|------|--------------------|--------------------------|
| 入口 | 只有 `forward(...)`，无 `kv_cache` | `forward()` 根据是否有 `kv_cache` 分支：有 → `_forward_inference`，无 → `_forward_train` |
| 序列组织 | 全序列 pad 到 `seq_len` 后 `torch.cat` 成 `[B, L, C]`，一次前向 | 训练：同 teacher；推理：用 `current_start/current_end` 按块输入，配合 KV cache |
| 时间嵌入 | `e`: `[B, 6, C]`，整段共享 | 训练时 `e` 按 `t.shape` unflatten 成 `[B, F, 6, C]`，支持按帧/块 |

### 2. Attention 块（核心差异）

| 项目 | Teacher | Student |
|------|--------|--------|
| Block 类 | `WanAttentionBlock` | `CausalWanAttentionBlock` |
| Self-Attn 类 | `WanSelfAttention` | `CausalWanSelfAttention` |
| Self-Attn 接口 | `(x, seq_lens, grid_sizes, freqs)`，无 cache | 多了 `kv_cache, current_start, current_end` |
| 注意力类型 | **双向**：`flash_attention(..., k_lens=seq_lens)`，全序列互相可见 | **因果**：无 cache 时用 `flex_attention` + `block_mask`（block-wise causal）；有 cache 时用 `attention(roped_query, kv_cache["k"][:,:current_end], kv_cache["v"][:,:current_end])`，只看过去+当前 |
| RoPE | `rope_apply(q/k, grid_sizes, freqs)`，整段一起算 | 无 cache：同 teacher；有 cache：`causal_rope_apply(..., start_frame=...)`，只对当前块从 `current_start` 起算 |
| KV cache | 无 | 有：当前块写入 `kv_cache["k/v"][:, current_start:current_end]`，读 `[:, :current_end]` |

### 3. 主干级差异（CausalWanModel 相对 WanModel）

| 项目 | Teacher (WanModel) | Student (CausalWanModel) |
|------|--------------------|--------------------------|
| blocks | `WanAttentionBlock` × num_layers | `CausalWanAttentionBlock` × num_layers |
| head | `Head`，`e`: `[B, C]` | `CausalHead`，`e`: `[B, F, 1, C]`，按帧调制 |
| 额外状态 | 无 | `self.block_mask`（block-wise causal mask）、`self.num_frame_per_block` |
| 训练时 mask | 无（双向） | `_prepare_blockwise_causal_attn_mask()`，得到 block-wise causal 的 `BlockMask` 给 flex_attention |

### 4. 参数与配置

- **构造函数**（`__init__`）两者几乎相同（model_type, patch_size, text_len, dim, num_heads 等），**参数量一致**，可共享同一套预训练权重做初始化。
- **Wrapper**：`CausalWanDiffusionWrapper` 继承 `WanDiffusionWrapper`，只改了两点：`self.model = CausalWanModel.from_pretrained(...)`，以及 `self.uniform_timestep = False`（因果按块/帧 timestep）。

### 5. 训练 / 推理分工

- **Teacher (WanModel)**：只做一次全序列 forward，无 KV cache，用于 DMD 里算 real score 和 backward simulation（若用 wan pipeline）。
- **Student (CausalWanModel)**：  
  - 训练：走 `_forward_train`，和 teacher 一样全序列，但用 block-wise causal mask；  
  - 推理：走 `_forward_inference`，按块 + KV cache，实现逐块自回归生成。

---

## 三、一句话对照

- **Teacher**：`WanModel` + `WanSelfAttention`，全序列、双向 attention，无 cache，定义在 `wan_base/modules/model.py`。  
- **Student**：`CausalWanModel` + `CausalWanSelfAttention`，同参数规模，但 block-wise causal attention + 可选 KV cache，推理按块、支持自回归；因果相关逻辑都在 `causal_model.py`。

如果你需要，我可以再按「从 config 到 real_score/generator 的加载」或「DMD 里具体哪一步用 teacher、哪一步用 student」把调用链标出来。