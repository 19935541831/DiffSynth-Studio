#!/usr/bin/env bash
# Wan video direct distill — inputs aligned with inference (prompt, input_image, action_seq).
# Fill in the variables below and run from repo root: bash Wan_action_fintune/distill/direct_distill/run.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# ----- 输入：与 inference 一致，请按需修改 -----
# 训练数据根目录（CSV 中相对路径均相对此处；若 CSV 用绝对路径可留空或与 inference 一致）
DATASET_BASE_PATH="${DATASET_BASE_PATH:-/path/to/your/distill_data}"
# 元数据 CSV 路径（列：prompt,input_image,action_seq,video,seed,rand_device,num_inference_steps,cfg_scale）
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-/path/to/your/metadata_distill.csv}"
# Base 模型目录（与 inference 的 --model_dir 一致：dit、VAE、text encoder、tokenizer、可选 action_encoder.pth）
MODEL_DIR="${MODEL_DIR:-/path/to/Wan2.1-I2V-14B-480P}"
# 蒸馏结果输出目录
OUTPUT_PATH="${OUTPUT_PATH:-./models/train/wan_direct_distill}"

# 分辨率/帧数/动作维度：与 inference 默认一致
HEIGHT=240
WIDTH=320
NUM_FRAMES=17
ACTION_JOINT_DIM=14

# 本地模型权重（JSON 列表，与 inference 同目录结构）
MODEL_PATHS="[
  \"${MODEL_DIR}/diffusion_pytorch_model-00001-of-00007.safetensors\",
  \"${MODEL_DIR}/diffusion_pytorch_model-00002-of-00007.safetensors\",
  \"${MODEL_DIR}/diffusion_pytorch_model-00003-of-00007.safetensors\",
  \"${MODEL_DIR}/diffusion_pytorch_model-00004-of-00007.safetensors\",
  \"${MODEL_DIR}/diffusion_pytorch_model-00005-of-00007.safetensors\",
  \"${MODEL_DIR}/diffusion_pytorch_model-00006-of-00007.safetensors\",
  \"${MODEL_DIR}/diffusion_pytorch_model-00007-of-00007.safetensors\",
  \"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",
  \"${MODEL_DIR}/Wan2.1_VAE.pth\",
  \"${MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\",
  \"${MODEL_DIR}/action_encoder.pth\"
]"

accelerate launch --mixed_precision bf16 \
  Wan_action_fintune/distill/direct_distill/train_distill.py \
  --dataset_base_path "$DATASET_BASE_PATH" \
  --dataset_metadata_path "$DATASET_METADATA_PATH" \
  --height $HEIGHT \
  --width $WIDTH \
  --num_frames $NUM_FRAMES \
  --action_joint_dim $ACTION_JOINT_DIM \
  --model_paths "$MODEL_PATHS" \
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --extra_inputs "input_image,action_seq,seed,rand_device,num_inference_steps,cfg_scale" \
  --trainable_models "dit" \
  --task "direct_distill" \
  --output_path "$OUTPUT_PATH" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --dataset_repeat 160 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --use_gradient_checkpointing
