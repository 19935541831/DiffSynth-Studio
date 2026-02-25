MODEL_DIR="/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200000

accelerate launch  --mixed_precision "bf16" --main_process_port=51631 /opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/train/train.py \
  --dataset_base_path /opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/data/robotwin_expert50_train_norm \
  --parquet_dir /opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/data/robotwin_expert50_train_norm \
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
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --num_frames 17 \
  --dataset_num_worker 7 \
  --prefetch_factor 4 \
  --dataloader_seed 42 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora_2" \
  --trainable_models "action_encoder" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --window_stride 5 \
  --shuffle_buffer_size 5000 \
  --extra_inputs "input_image,action_seq" \
  --gradient_accumulation_steps 4 \
  --max_grad_norm 0.5 \
  --warmup_steps 0 \
 

