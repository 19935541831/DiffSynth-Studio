# inference所需：需对方提供清单

---

## 一、推理用 Checkpoint（需对方提供）


| 项                      | 说明                                                                                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **训练权重（二选一）**          | **A.** 一个 combined 的 `.safetensors`（含 LoRA + action_encoder）→ 用 `--checkpoint_path` **B.** 一个目录，内含 `epoch-N.safetensors` → 用 `--lora_dir`；此时基座目录里还需有 `action_encoder.pth`（或对方说明从哪份 ckpt 加载） |
| **action_encoder.pth** | 仅当用 `--lora_dir` 且基座里没有、combined 里也没有时需要；若对方仅做过 DiT LoRA 而 action_encoder 未训，可自行用 `scripts/create_action_encoder_init.py` 生成初始权重                                                            |


---

## 二、Data（需对方提供）


| 项                             | 说明                                                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **推理 CSV**                    | 列：`prompt`, `input_image`, `action_seq`；每行一条样本                                                                                |
| **CSV 指向的文件**                 | 每行的首帧图/视频（`input_image` 路径）+ 对应动作序列 `.npy`（`action_seq` 路径）；`.npy` shape 为 `(T, joint_dim)`，与 `--num_frames`、`--joint_dim` 一致 |
| **action_normalization.json** | 可选；训练时若做了 action 归一化，推理用 `--action_norm_file` 指向该 JSON 以与训练一致                                                                 |


