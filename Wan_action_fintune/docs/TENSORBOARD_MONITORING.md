# TensorBoard 训练监控使用指南

## 概述

训练框架现已集成 TensorBoard 支持，可以实时监控训练过程中的各项指标。

## 功能特性

### 监控指标

1. **训练指标**
   - `train/loss` - 每步的损失值
   - `train/grad_norm` - 梯度L2范数（用于检测梯度爆炸）

2. **系统指标**
   - `system/gpu_memory_allocated_gb` - GPU已分配内存（GB）
   - `system/gpu_memory_reserved_gb` - GPU保留内存（GB）

3. **性能指标**
   - `performance/samples_per_second` - 训练吞吐量

## 使用方法

### 1. 安装依赖

```bash
pip install tensorboard
# 或者重新安装项目
pip install -e .
```

### 2. 启动训练

TensorBoard 监控默认已启用，无需修改训练脚本。日志将自动保存到：

```
{output_path}/logs/
```

例如，对于你的训练配置：
```
/project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora/logs/
```

### 3. 查看监控

在训练运行时，打开新终端窗口并运行：

```bash
tensorboard --logdir=/project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune/checkpoints/Wan2.1-I2V-14B-480P_lora/logs
```

然后在浏览器中访问：
```
http://localhost:6006
```

### 4. 远程服务器访问

如果在远程服务器上训练，可以使用SSH端口转发：

```bash
# 在本地机器上运行
ssh -L 6006:localhost:6006 user@remote-server
```

然后在远程服务器上启动 TensorBoard，在本地浏览器访问 `http://localhost:6006`

## 高级配置

### 禁用 TensorBoard

如果需要禁用 TensorBoard 监控：

```python
model_logger = ModelLogger(
    args.output_path,
    remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    enable_tensorboard=False  # 禁用 TensorBoard
)
```

### 调整日志频率

默认每 10 步记录一次指标。调整记录频率：

```python
model_logger = ModelLogger(
    args.output_path,
    remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    log_interval=50  # 每50步记录一次
)
```

### 自定义日志路径

```python
# 日志默认保存在 {output_path}/logs/
# 如需自定义，可以在初始化后修改
model_logger = ModelLogger(args.output_path)
# 日志将保存在 args.output_path/logs/
```

## 分布式训练

在多GPU或多节点训练时：

- TensorBoard 日志仅由主进程（rank 0）写入
- Loss 值会自动在所有进程间同步（accelerate 特性）
- 所有进程的梯度范数都会被记录

## 监控最佳实践

### 1. Loss 监控

- 观察 `train/loss` 曲线是否平滑下降
- 如果出现震荡，考虑降低学习率或增加 batch size

### 2. 梯度监控

- 监控 `train/grad_norm` 检测梯度爆炸
- 正常情况下梯度范数应该相对稳定
- 如果梯度范数突然增大（>100），可能需要梯度裁剪

### 3. 内存监控

- 观察 `system/gpu_memory_allocated_gb` 确保不会 OOM
- 如果内存使用接近上限，考虑：
  - 减小 batch size
  - 启用梯度检查点（已默认启用）
  - 使用混合精度训练

### 4. 性能监控

- `performance/samples_per_second` 反映训练速度
- 如果速度突然下降，可能是：
  - 数据加载瓶颈（考虑增加 num_workers）
  - 内存交换（减少 batch size）
  - 网络 I/O 问题

## 故障排除

### TensorBoard 无法启动

```bash
# 检查端口是否被占用
lsof -i:6006

# 使用其他端口
tensorboard --logdir=path/to/logs --port=6007
```

### 无数据显示

1. 确认训练已开始且运行了至少 `log_interval` 步
2. 检查日志目录是否正确
3. 确认有写入权限

### 日志文件过大

TensorBoard 日志会随训练增长。可以定期清理旧日志：

```bash
# 删除旧的日志
rm -rf {output_path}/logs/events.out.tfevents.*
```

## 示例输出

训练时终端输出示例：

```
Epoch 1/5:   2%|▏         | 10/500 [00:15<12:35,  1.54s/it]
```

TensorBoard 界面将显示：
- Loss 曲线随步数变化
- 梯度范数实时更新
- GPU 内存使用趋势
- 训练速度波动

## 与其他工具集成

如需更高级的监控功能，可以考虑：

- **Weights & Biases (wandb)** - 云端托管，团队协作
- **MLflow** - 实验追踪和模型管理
- **Prometheus + Grafana** - 生产环境监控

当前的 TensorBoard 实现为扩展这些工具提供了基础架构。
