module load nvhpc-hpcx-cuda12/23.11

MODEL_DIR="/project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P"

accelerate launch /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/train/train.py \
  --dataset_base_path /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/data/robotwin_dataset_train \
  --parquet_dir /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/data/robotwin_dataset_train \
  --height 240 \
  --width 320 \
  --model_paths "[\
[\
\"${MODEL_DIR}/diffusion_pytorch_model-00001-of-00007.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00002-of-00007.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00003-of-00007.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00004-of-00007.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00005-of-00007.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00006-of-00007.safetensors\",\
\"${MODEL_DIR}/diffusion_pytorch_model-00007-of-00007.safetensors\"\
],\
\"${MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth\",\
\"${MODEL_DIR}/Wan2.1_VAE.pth\",\
\"${MODEL_DIR}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth\",\
\"${MODEL_DIR}/action_encoder.pth\"\
]" \
  --tokenizer_path "${MODEL_DIR}/google/umt5-xxl" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --num_frames 17 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora" \
  --trainable_models "action_encoder" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --window_stride 5 \
  --extra_inputs "input_image,action_seq"

