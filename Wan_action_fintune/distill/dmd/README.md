# Causal DMD for Wan Action-Conditioned Model

两阶段蒸馏：**Phase 1 ODE 轨迹回归** → **Phase 2 DMD 分布匹配**，用于训练因果版 Wan action-conditioned 学生模型（CausalWanVideoActionDiT）。

## 流程概览

1. **Phase 1 (ODE)**：用教师（WanVideoActionDiT + pipeline）对若干 prompt/action 生成完整 ODE 轨迹并保存；再用 ODE Regression 训练学生，使其在随机采样的 (noisy_latent, timestep) 上预测 x0，与轨迹末端 clean latent 做 MSE。
2. **Phase 2 (DMD)**：用 backward simulation（教师从噪声逐步 denoise）得到轨迹，学生单步预测 + DMD 梯度匹配损失；可选加 critic（教师）的 denoising loss。

## 目录与文件

| 文件 | 说明 |
|------|------|
| `ode_trajectory.py` | 用教师 pipeline 生成 ODE 轨迹 `[noise, x_1, ..., x_T]`，可写 .pt 或返回 in-memory |
| `ode_regression.py` | ODE 阶段：ODERegression 类，读轨迹、随机采 (noisy, t)、学生预测 x0，MSE 到 clean |
| `dataset_ode.py` | 读 .pt 轨迹数据（ode_latent, prompt, action_seq） |
| `backward_simulation.py` | 给定 noise、prompt、action，用教师多步 denoise 得到轨迹，供 DMD 使用 |
| `dmd.py` | DMD 类：generator=学生、real/fake_score=教师；KL grad、distribution matching loss；backward 调用 action-conditioned backward_simulation |
| `train_ode.py` | ODE 阶段训练脚本 |
| `train_dmd.py` | DMD 阶段训练脚本 |
| `config_ode.yaml` / `config_dmd.yaml` | 两阶段超参与路径 |
| `README.md` | 本说明 |

## 数据格式

- **ODE 轨迹**：每条样本保存为 `.pt`，包含：
  - `ode_latent`: `(num_steps+1, F, C, H, W)`，即 `[noise, x_1, ..., x_T]`
  - `prompt`: `str`
  - `action_seq`: `(T_act, joint_dim)`，如 (81, 14)
- **生成 ODE 数据**：使用 `ode_trajectory.generate_trajectory_for_sample` 或 `generate_ode_trajectories_batch`，传入 pipeline、prompt、action_seq 等，将返回/写入的轨迹放入同一目录作为 `data_dir`。

## 运行方式

在仓库根目录（保证可 `import diffsynth`）下：

**Phase 1 – ODE 回归**

```bash
# 生成 ODE 轨迹（示例：需自己实现调用或使用已有脚本）
# 将 .pt 写入某目录，例如 ./ode_data

python -m Wan_action_fintune.distill.dmd.train_ode \
  --config Wan_action_fintune/distill/dmd/config_ode.yaml \
  --data_dir ./ode_data \
  --output_dir ./ode_ckpts \
  --batch_size 2 \
  --max_steps 10000
```

**Phase 2 – DMD**

```bash
# 使用与 ODE 相同或类似格式的 .pt（至少含 prompt、action_seq）作为 data_dir

python -m Wan_action_fintune.distill.dmd.train_dmd \
  --config Wan_action_fintune/distill/dmd/config_dmd.yaml \
  --data_dir ./ode_data \
  --output_dir ./dmd_ckpts \
  --generator_ckpt ./ode_ckpts/generator_010000.pt \
  --batch_size 2 \
  --max_steps 5000
```

## 配置要点

- `config_ode.yaml`：`data_dir`、`output_dir`、`model_dir`（可选）、`generator_ckpt`（可选）、`num_frame_per_block`、`joint_dim`、`num_ode_steps`。
- `config_dmd.yaml`：`data_dir`、`output_dir`、`generator_ckpt`（建议用 ODE 阶段 checkpoint）、`image_or_video_shape`（B,F,C,H,W  latent）、`backward_simulation`、`real_guidance_scale`。

## 与 direct_distill 的关系

- `direct_distill`：用已渲染好的教师视频 + 同一 prompt/action 训学生。
- 本目录 DMD：无预存视频，用教师 ODE 轨迹 + backward 模拟训学生；两套流程独立，DMD 仅依赖本目录与因果学生/推理代码。
