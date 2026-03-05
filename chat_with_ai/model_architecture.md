

---

# 3.3_diffsynth_act_based 模型架构总览
根据代码整理出的**完整模型架构**如下（原模型 + 新增部分，按数据流组织）。

## 一、整体：Wan 视频 Pipeline

`WanVideoPipeline` 由一串 **PipelineUnit** 按顺序跑，最后用 **model_fn_wan_video** 调用 DiT 等做去噪。  
和 action 相关的只是其中一段：**ActionEmbedder → 在 model_fn 里把 action_emb 融进时间调制**，其余都是原 Wan 的 I2V/T2V 设计。

---

## 二、原模型（基座）组件

### 1. 文本与图像编码（条件侧）

| 组件 | 类 / 实现 | 作用 | 典型权重 |
|------|------------|------|----------|
| **text_encoder** | `WanTextEncoder`（包一层 T5/UMT5） | 把 prompt 编码成文本特征 | `models_t5_umt5-xxl-enc-bf16.pth` |
| **image_encoder** | `WanImageEncoder`（CLIP） | 把首帧图编码成 1280 维 CLIP 特征，供 DiT 做 I2V 条件 | `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |

Tokenizer：`google/umt5-xxl`，与 text_encoder 配套。

### 2. 视频 VAE

| 组件 | 类 | 作用 |
|------|-----|------|
| **vae** | `WanVideoVAE` | 视频 latent 的编解码：编码时把视频压成 3D latent，解码时把 DiT 输出的 latent 解回像素视频。 |

典型权重：`Wan2.1_VAE.pth`。  
空间下采样由 `height_division_factor` / `width_division_factor`（如 16）决定；时间上有 `time_division_factor=4`，所以帧数会压成 T_latent。

### 3. 去噪主干：DiT（双阶段可选）

| 组件 | 类 | 作用 |
|------|-----|------|
| **dit** | `WanModel`（`wan_video_dit.WanModel`） | 主去噪 DiT：在 latent 空间做 flow-matching，输入为噪声/中间 latent + 时间步 + 文本/图像/时间调制等，输出预测速度场或噪声。 |
| **dit2** | 同上（可选） | 部分 Wan 变体（如 I2V-A14B）用「高噪声 dit + 低噪声 dit2」两阶段；按 timestep 切换。 |

**WanModel 结构概览**（原版，未改 forward 接口）：

- `patch_embedding`：3D Conv 把 latent 打成 patch。
- `time_embedding` + `time_projection`：时间步 → 6×dim 的调制（shift/scale 等），用于每个 DiTBlock。
- `text_embedding`：文本特征投影到 dim。
- `img_emb`：CLIP 特征投影到 dim（I2V 时用）。
- `blocks`：多个 `DiTBlock`（Self-Attn + MLP，AdaLN 用 time_projection 的调制）。
- `head`：输出通道映射，得到预测的噪声/速度。
- 3D RoPE、flash attention 等都在 block 里。

权重：`diffusion_pytorch_model-0000x-of-00007.safetensors`（7 片）。

### 4. 其他原版条件/控制模块（与 action 无关）

| 组件 | 类 | 作用 |
|------|-----|------|
| **motion_controller** | `WanMotionControllerModel` | 根据 motion bucket 等生成额外调制，加在时间调制上。 |
| **vace** / **vace2** | `VaceWanModel` | 某些 Wan 变体里的辅助模型（如 VACE 分支）。 |
| **vap** | `MotWanModel` | 另一类运动/控制模块。 |
| **animate_adapter** | `WanAnimateAdapter` | 做人脸/姿态动画等时的适配。 |
| **audio_encoder** | `WanS2VAudioEncoder` | S2V 用到的音频编码。 |

这些在「只做 action 条件 I2V」时可以不加载或不用；当前 action 流程里**没有改它们**，只是和它们共用同一个 pipeline。

---

## 三、为 action 新增的组件

### 1. Action Encoder（新增小网络）

| 组件 | 类 | 作用 |
|------|-----|------|
| **action_encoder** | `WanActionEncoder`（`wan_video_action_encoder.py`） | 把「按帧打包后的动作序列」映射成与 DiT **dim 一致**的 per-token 条件。 |

**结构**：

- 输入：`(B, T_packed, 4*joint_dim)`，其中 T_packed 对应 VAE 的 T_latent（每 4 帧动作打成一 token）。
- MLP：`Linear(4*joint_dim → dit_dim) → SiLU → Linear(dit_dim → dit_dim)`，输出 `(B, T_packed, dit_dim)`。
- 输出在 pipeline 里记为 **action_emb**，在 model_fn 里与时间嵌入相加，再经 `time_projection` 变成 DiT 的调制。

权重：可单独 `action_encoder.pth`，或与 LoRA 一起放在合并 checkpoint 里（带 `pipe.action_encoder.` 前缀）。

### 2. Pipeline 里的 Action 插入点（不改 DiT 结构）

- **WanVideoUnit_ActionEmbedder**  
  - 读 `action_seq`（和 num_frames），做「首帧重复 3 次 + 每 4 帧打包」，再调用 `pipe.action_encoder(...)` 得到 **action_emb**。
- **model_fn_wan_video**（在 `wan_video.py`）  
  - 若提供了 **action_emb**：  
    - 按 T_latent、h、w 扩成 per-token，和 ref_tokens 对齐；  
    - 与时间嵌入 **t** 相加：`t_fused = t + action_emb`；  
    - 用 **t_fused** 做 `time_projection` 得到调制，后面 DiT 的 AdaLN 照常用这些调制。  
  - 即：**原 DiT 的 WanModel 不变，只是“时间条件”被改成了“时间 + action”的融合条件**。

### 3. 可选：DiT 上的 LoRA（训练时加上的 adapter）

- 训练时若 `lora_base_model="dit"`，会在 **dit**（和/或 dit2）上注入 LoRA（如 q/k/v/o、ffn.0、ffn.2）。
- **不改变 WanModel 的类定义**，只是在原有线性层旁加 `lora_A` / `lora_B`，前向时 `original_output + lora_scale * (x @ A @ B)`。
- 推理时加载同一套 base DiT + LoRA 权重即可。

**说明**：仓库里还有 `wan_video_action_dit.py` 的 **WanVideoActionDiT**（在 DiT 内部做 time+action 调制），但当前 pipeline 和 config 用的是**原 WanModel + 外部 action_emb 与 t 相加**的方式，没有用 WanVideoActionDiT 这个子类。

---

## 四、前向数据流（与 action 相关的部分）

1. **Prompt** → **text_encoder** → 文本特征；**首帧图** → **image_encoder** → CLIP 特征。  
2. **首帧** → **VAE 编码** → 首帧 latent，和噪声/中间 latent 一起作为 DiT 的 **x**；CLIP 特征进 DiT 的 **context**。  
3. **action_seq (B, T, joint_dim)** → **WanVideoUnit_ActionEmbedder**：  
   - 首帧重复 3 次 + 每 4 帧打包 → `(B, T_packed, 4*joint_dim)`；  
   - **action_encoder** → **action_emb (B, T_packed, dit_dim)**。  
4. **model_fn_wan_video**：  
   - 时间步 → time_embedding → **t**；  
   - **t + action_emb**（按空间展开并对齐 ref）→ **t_fused**；  
   - **time_projection(t_fused)** → 调制 → 送入 **dit** 的每个 block（AdaLN）。  
5. **dit**（+ 可选 LoRA）在 latent 上做去噪 → 输出预测；**VAE 解码** → 最终视频。

---

## 五、训练时：谁被冻住、谁在训

| 组件 | 是否训练 | 说明 |
|------|----------|------|
| text_encoder, image_encoder, vae, motion_controller, vace, vace2, vap, animate_adapter, audio_encoder | 否 | 全部冻结。 |
| **dit / dit2 主体** | 否 | 冻结；只通过 LoRA 适配（若启用）。 |
| **action_encoder** | 是 | 从零或 checkpoint 训，学习把动作映射成 DiT 能用的条件。 |
| **dit（及 dit2）上的 LoRA** | 可选 | `lora_base_model="dit"` 时训练；否则不挂 LoRA、不训。 |

---

## 六、小结（一句话版）

- **原模型**：Wan 2.1 I2V（T5 + CLIP + WanVideoVAE + WanModel DiT，可选 dit2 / motion / vace / vap 等），负责「文本 + 首帧 → 视频」的生成能力。  
- **新增**：**WanActionEncoder** 把 `(B, T_packed, 4*joint_dim)` 打成 **action_emb**，在 **model_fn_wan_video** 里与时间嵌入相加后送入**原有 WanModel** 的 time 调制路径；可选在 **dit** 上挂 **LoRA** 做轻量微调。  
- **整体**：仍然是「一个 Wan 视频 DiT + 一堆冻结编码器/VAE」，只是多了一个 **action → action_emb** 的编码器，以及「时间调制 = t + action_emb」的融合方式；没有替换或重写 DiT 的类本身。


---

# 参数量与 FLOP 一览（Wan2.1-I2V-14B-480P + Action）

## 一、参数量（Parameters）

以 **Wan2.1-I2V-14B-480P** 及当前 action 设定为准（`dim=5120, num_layers=40, ffn_dim=13824, num_heads=40` 等）。

| 模块 | 参数量（约） | 说明 |
|------|----------------|------|
| **DiT（dit）** | **~14B** | 主去噪 Transformer，7 个 safetensors 分片；与命名 14B 一致。 |
| **dit2**（若存在） | **~14B** | 双阶段时的低噪声 DiT，与 dit 同规模。 |
| **Text Encoder（T5/UMT5-XXL enc）** | **~4.7B** | 仅编码器，`models_t5_umt5-xxl-enc-bf16.pth`。 |
| **Image Encoder（CLIP）** | **~1–2B** | open-clip-xlm-roberta-large-vit-huge-14，以视觉编码器为主。 |
| **VAE（Wan2.1_VAE）** | **~300M 级** | 视频 VAE，量级在百 M。 |
| **Motion controller** | **~数 M** | 小模块，相对 14B 可忽略。 |
| **Action Encoder（新增）** | **~26.5M** | `joint_dim=14, dit_dim=5120`：Linear(56, 5120) + Linear(5120, 5120)。 |
| **LoRA（rank=32，可选）** | **~30–80M** | 挂在 dit 的 q/k/v/o、ffn.0、ffn.2 上，40 层 × 多矩阵。 |

**合计（单 DiT、无 dit2、无 LoRA）**：约 **20B+**（14B + 4.7B + 1~2B + 0.3B + 0.026B + 其它小头）。  
若启用 **dit2** 或 **LoRA**，在以上基础上再加对应项。

---

## 二、FLOP（浮点运算量）

仓库里**没有**现成的 FLOP 统计或 profiling 脚本，只能按结构做量级估计。

- **单次去噪步**的 FLOP 主要来自：
  - **DiT**：自注意力约 `O(2 · dim · S²)`（S = 序列长度），MLP 约 `O(2 · S · dim · ffn_dim)`；  
    - 例如 17 帧、240×320、patch (1,2,2)、time 下采样后，S 在千级到数千级，单步 DiT 在 **数十 TFLOPs** 量级很常见。
  - **VAE 编解码**：编一次、解一次，相对 DiT 少一个数量级左右。
  - **Text / Image / Action 编码**：相对 DiT 可忽略。

- **整段推理**（例如 50 步、17 帧 240×320）：
  - 总 FLOP ≈ **50 × (单步 DiT FLOP + VAE 等)**，量级在 **数百 TFLOPs ~ 1 PFLOP** 左右，与分辨率、帧数、步数强相关。

若要**精确 FLOP**，需要在本地用 `fvcore`、`ptflops`、`deepspeed` 等工具对 `model_fn_wan_video` + 单步前向做一次 profiling；当前代码库未提供现成数字。

---

## 三、表格小结（便于引用）

| 项目 | 参数量 | FLOP（单次生成，量级） |
|------|--------|--------------------------|
| DiT (dit) | ~14B | 主导，单步数十 TFLOPs 量级 |
| dit2（若用） | ~14B | 同 dit |
| Text Encoder | ~4.7B | 可忽略 |
| Image Encoder | ~1–2B | 可忽略 |
| VAE | ~0.3B | 编+解一次，相对 DiT 小 |
| Action Encoder | ~26.5M | 可忽略 |
| LoRA（若用） | ~30–80M | 与 dit 一起算在 DiT 前向里 |
| **整 pipeline（约）** | **~20B+** | **数百 TFLOPs ~ 1 PFLOP**（与分辨率/步数/帧数有关） |

如需把某一块（例如只算 DiT 或只算 Action Encoder）的参数量或 FLOP 写成文档/表格，可以指定模块和分辨率、步数，我可以按公式帮你算一版具体数字。