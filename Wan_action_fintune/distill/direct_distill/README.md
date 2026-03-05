# Video Direct Distill

端到端蒸馏：少步生成对齐多步 teacher。数据格式与 `Wan_action_fintune/inference` 一致。

## 代码结构

```
Wan_action_fintune/distill/direct_distill/
├── README.md           # 本说明与输入位置
├── train_distill.py    # 入口：UnifiedDataset(CSV+视频) + WanTrainingModule + launch_training_task
└── run.sh              # 启动脚本，填写/覆盖各输入路径与超参
```

## 输入放在哪里

与 **inference**（`inference/Wan2.1-I2V-14B-480P/inference.py`）保持一致：


| 输入含义            | 在 run.sh / 命令行里填的位置                                            | 说明                                                                                                                    |
| --------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **训练数据根目录**     | `--dataset_base_path` 或 env `DATASET_BASE_PATH`                | CSV 中**相对路径**的根目录；若 CSV 用绝对路径，可与 inference 一致。                                                                        |
| **元数据 CSV**     | `--dataset_metadata_path` 或 env `DATASET_METADATA_PATH`        | 列：**prompt, input_image, action_seq**（与 inference 相同） + **video, seed, rand_device, num_inference_steps, cfg_scale**。 |
| **Base 模型目录**   | `--model_paths`（JSON）或 env `MODEL_DIR`（run.sh 里拼成 model_paths） | 与 inference 的 `--model_dir` 同一目录结构（dit、VAE、text encoder、tokenizer、可选 action_encoder.pth）。                             |
| **Tokenizer**   | `--tokenizer_path`                                             | 与 inference 一致，如 `{model_dir}/google/umt5-xxl`。                                                                       |
| **蒸馏结果输出**      | `--output_path` 或 env `OUTPUT_PATH`                            | 仅 distill 使用。                                                                                                         |
| **分辨率/帧数/动作维度** | `--height`、`--width`、`--num_frames`、`--action_joint_dim`       | 与 inference 默认一致（240, 320, 17, 14）。                                                                                   |


**run.sh 里主要改这几处即可：**

- `DATASET_BASE_PATH`、`DATASET_METADATA_PATH`：数据与 CSV 路径。
- `MODEL_DIR`：与 inference 的 `--model_dir` 一致。
- `OUTPUT_PATH`：蒸馏 ckpt 输出目录。

## CSV 格式（与 inference 一致 + 蒸馏列）

与 inference 相同的三列 + 蒸馏用列：

```csv
prompt,input_image,action_seq,video,seed,rand_device,num_inference_steps,cfg_scale
"click the bell with the right arm",/path/to/first_frame.jpg,/path/to/action.npy,/path/to/teacher_video.mp4,0,cpu,4,1
```

- **prompt, input_image, action_seq**：与 inference 的 CSV 一致（路径可为绝对或相对 `dataset_base_path`）。
- **video**：该条样本对应的 teacher 多步生成视频路径。
- **seed, rand_device, num_inference_steps, cfg_scale**：训练时 student 少步用的参数（num_inference_steps 一般为 4，cfg_scale 一般为 1）。

## 运行

从**仓库根目录**（`3.3_diffsynth_act_based`）执行：

```bash
bash Wan_action_fintune/distill/direct_distill/run.sh
```

或先设置环境变量再运行：

```bash
export DATASET_BASE_PATH=/path/to/data
export DATASET_METADATA_PATH=/path/to/metadata_distill.csv
export MODEL_DIR=/path/to/Wan2.1-I2V-14B-480P
export OUTPUT_PATH=./models/train/wan_direct_distill
bash Wan_action_fintune/distill/direct_distill/run.sh
```

若使用 ModelScope 下载模型而非本地 `model_paths`，可在 `train_distill.py` 中改用 `--model_id_with_origin_paths`，与官方 `examples/wanvideo/model_training/special/direct_distill` 用法一致。