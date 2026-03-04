# 3.3_diffsynth_act_based 说明

## 1. 本文件夹做什么

本仓库基于 **DiffSynth-Studio**（ModelScope 的扩散模型引擎），在 **Wan 视频模型** 上增加 **动作条件（action conditioning）** 能力：

- **目标**：用「文本提示 + 首帧图像 + 动作序列」作为条件，生成/微调可控视频（例如机器人轨迹、行为克隆等）。
- **技术要点**：
  - 在 Wan 2.1 I2V（如 Wan2.1-I2V-14B-480P）等模型上接入 **Action Encoder**，将每帧/窗口的动作向量编码后注入 DiT。
  - 训练时使用 **Parquet 流式数据集** + 滑动窗口，支持长轨迹、大 batch、低显存。
  - 支持对 **action_encoder** 全量训练，并对 **DiT** 做 **LoRA** 微调，便于在自有数据上适配动作控制。

因此，本文件夹主要提供：**数据准备 → 动作条件训练 → 推理验证** 的完整 pipeline，适合做「以动作为条件的 I2V」研究与落地。

---

## 2. 文件夹结构（与 action 相关部分）

```
3.3_diffsynth_act_based/
├── diffsynth/                    # 核心库（DiffSynth-Studio）
│   ├── pipelines/
│   │   └── wan_video.py          # Wan 视频 pipeline（含 action 条件接入）
│   ├── models/
│   │   ├── wan_video_action_encoder.py   # 动作编码器
│   │   └── wan_video_action_dit.py       # 与 DiT 的 action 条件融合
│   ├── core/data/
│   │   ├── parquet_streaming_dataset.py  # Parquet 流式 + 滑动窗口
│   │   └── parquet_utils.py              # Parquet 读写工具
│   └── ...
│
├── Wan_action_fintune/           # ★ 动作微调与推理（本 pipeline 主入口）
│   ├── train/                    # 训练
│   │   ├── train.py              # 主训练脚本（Parquet 流式 + action_encoder + LoRA）
│   │   ├── train_unifieddataset.py
│   │   ├── Wan2.1-I2V-14B-480P.sh # 14B 480P 示例启动脚本
│   │   └── accelerate_config_14B.yaml
│   ├── inference/                # 推理
│   │   ├── README.md             # 推理 CSV 格式与参数说明
│   │   ├── Wan2.1-I2V-14B-480P/
│   │   │   └── inference.py     # 按 CSV(prompt, input_image, action_seq) 批量生成视频
│   │   └── Wan2.2-TI2V-5B/
│   │       └── inference.py
│   ├── data/scripts/             # 数据准备
│   │   ├── convert_to_parquet.py         # 通用：CSV 或 RoboTwin 原始 → Parquet
│   │   ├── convert_robotwin_dataset.py   # RoboTwin 原始 → 视频+npy+CSV
│   │   ├── convert_to_parquet_residual.py
│   │   └── ...
│   └── docs/                     # 说明文档
│       ├── README_PARQUET_STREAMING.md   # Parquet 流式训练用法
│       ├── CONVERSION_GUIDE.md           # RoboTwin 数据转换
│       ├── SLIDING_WINDOW_GUIDE.md
│       └── TENSORBOARD_MONITORING.md
│
├── examples/                     # 官方示例（wanvideo / z_image / flux 等）
├── docs/                         # 项目级文档（中/英）
├── pyproject.toml               # 依赖与安装
└── README.md                    # DiffSynth-Studio 总览
```

**简要对应**：

- 要 **跑通整条 pipeline**：主要用 `Wan_action_fintune/` 下的 `data/scripts`、`train/`、`inference/`。
- **动作条件** 的实现与配置在 `diffsynth/models/`、`diffsynth/pipelines/wan_video.py` 及训练脚本参数（如 `--extra_inputs input_image,action_seq`、`--trainable_models action_encoder`）。

---

## 3. 新人如何跑通整条 Pipeline

### 3.1 环境

在仓库根目录执行：

```bash
cd /path/to/3.3_diffsynth_act_based
pip install -e .
```

需要 GPU、CUDA，以及 `accelerate` 做分布式/单机训练。详细依赖见 `pyproject.toml` 和项目 [安装文档](docs/en/Pipeline_Usage/Setup.md)。

### 3.2 准备基座模型

- 下载 **Wan2.1-I2V-14B-480P**（或你使用的 Wan 版本）到本地目录，例如：
  - `MODEL_DIR=/path/to/Wan2.1-I2V-14B-480P`
- 若基座里没有预训练好的 `action_encoder.pth`，训练脚本可通过 `--action_joint_dim` 自动初始化 action encoder；若有，则放在同一 `MODEL_DIR` 即可。

### 3.3 数据准备（必须得到 Parquet）

训练 **只支持 Parquet 流式**，需先把原始数据转成 Parquet。

**方式 A：从 RoboTwin 原始格式**

```bash
# 1）可选：先转成「视频 + npy + CSV」中间格式
python Wan_action_fintune/data/scripts/convert_robotwin_dataset.py \
  --raw_data_dir /path/to/robotwin_raw \
  --output_dir /path/to/robotwin_processed

# 2）再转成 Parquet（若上一步已生成 CSV，可用 --mode csv 指向该 CSV）
python Wan_action_fintune/data/scripts/convert_to_parquet.py \
  --mode robotwin \
  --raw_data_dir /path/to/robotwin_raw \
  --output_dir /path/to/parquet_output \
  --shard_size 10000
```

**方式 B：从已有 CSV（每行：视频/首帧 + 动作文件路径）**

```bash
python Wan_action_fintune/data/scripts/convert_to_parquet.py \
  --mode csv \
  --csv_path /path/to/metadata.csv \
  --base_path /path/to/data \
  --output_dir /path/to/parquet_output \
  --shard_size 10000
```

得到 `parquet_output` 目录后，记作 `PARQUET_DIR`，后面训练用。

### 3.4 训练

以 Wan2.1-I2V-14B-480P 为例，参考 `Wan_action_fintune/train/Wan2.1-I2V-14B-480P.sh`，按你的路径改 `MODEL_DIR`、`PARQUET_DIR`、`output_path` 等：

```bash
export MODEL_DIR=/path/to/Wan2.1-I2V-14B-480P
export PARQUET_DIR=/path/to/parquet_output

accelerate launch --mixed_precision bf16 Wan_action_fintune/train/train.py \
  --dataset_base_path "${PARQUET_DIR}" \
  --parquet_dir "${PARQUET_DIR}" \
  --height 240 \
  --width 320 \
  --model_paths "[...]" \   # 与 .sh 中一致：DiT 分片 + T5 + VAE + CLIP + action_encoder
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --num_frames 17 \
  --window_stride 5 \
  --shuffle_buffer_size 5000 \
  --extra_inputs "input_image,action_seq" \
  --trainable_models "action_encoder" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --output_path "/path/to/wan_action_lora_output" \
  --gradient_accumulation_steps 4 \
  # ... 其他参数见 train.py 或 .sh
```

- `--extra_inputs input_image,action_seq` 表示使用首帧图 + 动作序列作为条件。
- `--trainable_models action_encoder` 表示训练 action encoder；DiT 部分由 `--lora_base_model dit` 等做 LoRA 微调。
- 更多参数（如 `--action_joint_dim`、`--remove_prefix_in_ckpt`）见 `Wan_action_fintune/docs/README_PARQUET_STREAMING.md`。

### 3.5 推理验证

训练结束后，用 CSV 指定「prompt + 首帧图 + 动作序列」批量生成视频：

```bash
python Wan_action_fintune/inference/Wan2.1-I2V-14B-480P/inference.py \
  --csv_path /path/to/validation_data.csv \
  --model_dir /path/to/Wan2.1-I2V-14B-480P \
  --lora_dir /path/to/wan_action_lora_output \
  --output_dir ./validation_outputs \
  --num_frames 17 \
  --height 240 \
  --width 320
```

CSV 格式需包含列：`prompt`, `input_image`, `action_seq`（详见 `Wan_action_fintune/inference/README.md`）。  
动作序列为 `.npy`，形状 `(num_frames, joint_dim)`，与训练时的 `action_joint_dim` 一致。

---

## 4. 文档速查

| 需求           | 文档 |
|----------------|------|
| Parquet 流式训练、滑动窗口、核心参数 | `Wan_action_fintune/docs/README_PARQUET_STREAMING.md` |
| RoboTwin 数据转换与目录结构         | `Wan_action_fintune/docs/CONVERSION_GUIDE.md` |
| 推理 CSV 格式与全部推理参数         | `Wan_action_fintune/inference/README.md` |
| 训练脚本示例（14B 480P）           | `Wan_action_fintune/train/Wan2.1-I2V-14B-480P.sh` |

按「数据 → 训练 → 推理」顺序走完上述三步，即可跑通整条 **DiffSynth action-based** pipeline。
