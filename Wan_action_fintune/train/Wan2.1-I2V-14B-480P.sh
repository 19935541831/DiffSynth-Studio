MODEL_DIR="/project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P"

accelerate launch /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/train/train.py \
  --dataset_base_path /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/data/robotwin_dataset_train \
  --dataset_metadata_path /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/data/robotwin_dataset_train/metadata.csv \
  --data_file_keys video,action_seq \
  --height 240 \
  --width 320 \
  --dataset_repeat 10 \
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
  --num_epochs 5 \
  --num_frames 17 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora" \
  --trainable_models "action_encoder" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 16 \
  --enable_sliding_window \
  --window_stride 1 \
  --extra_inputs "input_image,action_seq"

