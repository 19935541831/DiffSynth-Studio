import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
import glob
import os
base_path = "/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P"
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=glob.glob(os.path.join(base_path, "diffusion_pytorch_model*.safetensors")), skip_download=True),
        ModelConfig(path=os.path.join(base_path, "models_t5_umt5-xxl-enc-bf16.pth"),skip_download=True),
        ModelConfig(path=os.path.join(base_path, "Wan2.1_VAE.pth"),skip_download=True),
        ModelConfig(path=os.path.join(base_path, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),skip_download=True),
    ],
    tokenizer_config=ModelConfig(path=os.path.join(base_path, "google/umt5-xxl"),skip_download=True),
)

image = Image.open("/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/inference/examples/outputs/first_frame.jpg")

# Image-to-video
video = pipe(
    prompt="click the bell with the right arm",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    seed=0, 
    tiled=True,
    num_frames=17,
    height=240,
    width=320,
)
save_video(video, "/opt/tiger/wan_action_finetune_workspace/DiffSynth-Studio/Wan_action_fintune/inference/examples/outputs/test.mp4", fps=30, quality=10)
