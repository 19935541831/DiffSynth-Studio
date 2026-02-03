# Wan Action Inference Script

This script validates trained Wan video models with action_condition by generating videos from CSV input.

## Usage

```bash
python inference.py --csv_path <path_to_csv> [options]
```

## CSV Format

The CSV file should contain the following columns:
- `prompt`: Text prompt describing the video (str)
- `input_image`: Path to the input image file (supports .jpg, .png, etc.) or video file (.mp4, .avi, etc. - first frame will be extracted)
- `action_seq`: Path to the action sequence .npy file

Example CSV:
```csv
prompt,input_image,action_seq
"Take the green bottle from the table",/path/to/image.jpg,/path/to/action.npy
"Lift the bottle head-up",/path/to/video.mp4,/path/to/action2.npy
```

Note: If `input_image` points to a video file, the first frame will be automatically extracted and used as the input image.

## Arguments

### Required
- `--csv_path`: Path to CSV file with columns: prompt, input_image, action_seq

### Optional
- `--model_dir`: Directory containing base model checkpoints (default: `/project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P`)
- `--lora_dir`: Directory containing LoRA checkpoints (default: `/project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P_lora`)
- `--lora_epoch`: Specific LoRA epoch to load (default: latest)
- `--output_dir`: Directory to save generated videos (default: `./outputs`)
- `--joint_dim`: Dimension of action sequence vectors (default: 14)
- `--dit_dim`: Dimension of DiT model (default: inferred from config.json)
- `--num_frames`: Number of frames to generate (default: 17)
- `--height`: Video height (default: 240)
- `--width`: Video width (default: 320)
- `--num_inference_steps`: Number of inference steps (default: 50)
- `--cfg_scale`: Classifier-free guidance scale (default: 5.0)
- `--seed`: Random seed (default: random)
- `--device`: Device to use (default: cuda)
- `--fps`: Frame rate for output video (default: 15)
- `--quality`: Video quality (1-10, higher is better quality, default: 5)

## Example

```bash
python inference.py \
    --csv_path validation_data.csv \
    --model_dir /project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P \
    --lora_dir /project/peilab/Puxin/Wan_action/checkpoints/Wan2.1-I2V-14B-480P_lora \
    --output_dir ./validation_outputs \
    --num_frames 17 \
    --height 240 \
    --width 320 \
    --fps 15 \
    --quality 5
```

## Notes

- The action sequence files (.npy) should have shape `(num_frames, joint_dim)`
- Input images will be automatically converted to RGB format
- Generated videos are saved as MP4 files with configurable fps and quality
- If LoRA directory exists, the latest epoch checkpoint will be loaded automatically
