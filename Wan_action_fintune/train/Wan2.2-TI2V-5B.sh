MODEL_DIR="/mnt/hdfs/zhufangqi/code/wan/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.2-TI2V-5B"
DATA_DIR="/home/tiger/data_cache/robotwin_finetune/robotwin_expert5_train"
TRAIN_METADATA="${DATA_DIR}/metadata.csv"
VAL_METADATA="/home/tiger/data_cache/robotwin_finetune/robotwin_expert5_val/metadata.csv"
OUTPUT_DIR="/mnt/hdfs/zhufangqi/code/wan/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.2-TI2V-5B_sft"

export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200000

accelerate launch --mixed_precision "bf16" --main_process_port=51631 /mnt/hdfs/zhufangqi/code/wan/DiffSynth-Studio/Wan_action_fintune/train/train_unifieddataset.py \
  --dataset_base_path "${DATA_DIR}" \
  --dataset_metadata_path "${TRAIN_METADATA}" \
  --val_dataset_metadata_path "${VAL_METADATA}" \
  --validation_steps 200 \
  --data_file_keys "video,action_seq" \
  --height 256 \
  --width 320 \
  --num_frames 9 \
  --model_paths "[\
[\
\"${MODEL_DIR}/diffusion_pytorch_model-00001-of-00003.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00002-of-00003.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00003-of-00003.safetensors\"\
],\
\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\
\"${MODEL_DIR}/Wan2.2_VAE.pth\"\
]" \
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --learning_rate 1e-5 \
  --num_epochs 100 \
  --dataset_num_workers 4 \
  --output_path "${OUTPUT_DIR}" \
  --trainable_models "dit,action_encoder" \
  --window_stride 1 \
  --extra_inputs "input_image,action_seq" \
  --action_joint_dim 14 \
  --gradient_accumulation_steps 2 \
  --wandb_project "Wan2.2-TI2V-5B-action-finetune" \
  --disable_prompt