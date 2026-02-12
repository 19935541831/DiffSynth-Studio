# Parquet 流式数据加载器训练指南

本训练脚本已简化为**仅支持 Parquet 流式模式**，删除了传统的 UnifiedDataset 相关代码。

## 主要改动

### 1. 删除的功能
- ❌ UnifiedDataset 相关导入和代码
- ❌ CSV + 独立视频/动作文件的加载方式
- ❌ `--enable_sliding_window` 参数（流式模式默认启用）
- ❌ `--use_parquet_streaming` 参数（现在是唯一模式）
- ❌ `--video_key` 和 `--action_key` 参数
- ❌ 传统的 `launch_training_task` 和 `launch_data_process_task`

### 2. 保留的核心功能
- ✅ ParquetStreamingDataset（高效流式加载）
- ✅ 滑动窗口自动处理（通过 `--num_frames` 和 `--window_stride`）
- ✅ Shuffle buffer 随机化
- ✅ 多 Worker 并行加载
- ✅ GPU 加速帧解码（可选）

## 使用方法

### 步骤 1: 转换数据到 Parquet 格式

#### 从 CSV 格式转换
```bash
python Wan_action_fintune/data/scripts/convert_to_parquet.py \
    --mode csv \
    --csv_path /path/to/metadata.csv \
    --base_path /path/to/data \
    --output_dir /path/to/parquet_output \
    --shard_size 10000 \
    --jpeg_quality 95
```

#### 从 RoboTwin 原始格式转换
```bash
python Wan_action_fintune/data/scripts/convert_to_parquet.py \
    --mode robotwin \
    --raw_data_dir /path/to/robotwin_raw \
    --output_dir /path/to/parquet_output \
    --shard_size 10000
```

### 步骤 2: 训练

```bash
accelerate launch Wan_action_fintune/train/train.py \
    --task sft:train \
    --model_paths /path/to/model \
    --output_path ./outputs/wan_action \
    --parquet_dir /path/to/parquet_output \
    --num_frames 17 \
    --window_stride 1 \
    --shuffle_buffer_size 10000 \
    --height 240 \
    --width 320 \
    --action_joint_dim 14 \
    --extra_inputs input_image,action_seq \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --dataset_num_workers 4
```

## 核心参数说明

### 必需参数
- `--parquet_dir`: Parquet 分片文件所在目录（**必需**）
- `--num_frames`: 窗口大小（每个样本的帧数）

### 滑动窗口参数
- `--window_stride`: 窗口滑动步长
  - `1`: 最大重叠（每次移动1帧）
  - `4`: 中等重叠（每次移动4帧）
  - `--num_frames`: 无重叠（每个窗口独立）

### 随机化参数
- `--shuffle_buffer_size`: Shuffle 缓冲区大小
  - 较大值（如 10000）提供更好的随机性
  - 较小值减少内存占用
- `--dataloader_seed`: 随机种子（可选）

### Action 参数
- `--action_joint_dim`: 动作向量维度（如机器人为 14）
- `--extra_inputs`: 额外输入（通常为 `input_image,action_seq`）

### 性能参数
- `--dataset_num_workers`: 数据加载 Worker 数量（推荐 4-8）
- 多 Worker 时自动启用：
  - `prefetch_factor=2`（每个 worker 预取 2 批）
  - `persistent_workers=True`（保持 worker 存活）

## 支持的训练任务

- `sft`: Standard Fine-Tuning
- `sft:train`: SFT Training mode
- `direct_distill`: Direct Distillation
- `direct_distill:train`: Distillation Training mode

## 性能优化

### 推荐配置
```bash
--shuffle_buffer_size 10000      # 较大的 buffer 提供更好的随机性
--dataset_num_workers 4          # 4 个并行 worker
--window_stride 1                # 最大重叠（数据增强）
--jpeg_quality 95                # 高质量压缩（转换时设置）
```

### 内存受限配置
```bash
--shuffle_buffer_size 1000       # 减少 buffer 大小
--dataset_num_workers 2          # 减少 worker 数量
--window_stride 4                # 增加步长，减少窗口数量
```

## 常见问题

### Q: 如何设置滑动窗口大小？
**A**: 使用 `--num_frames` 参数。例如 `--num_frames 17` 表示每个窗口包含 17 帧。

### Q: 我的数据还是 CSV 格式，怎么办？
**A**: 必须先使用 `convert_to_parquet.py` 转换为 Parquet 格式。

### Q: 能否使用旧的 UnifiedDataset？
**A**: 不能。本版本已完全移除 UnifiedDataset 支持。如需使用，请切换到旧版本分支。

### Q: 如何验证 Parquet 文件是否正确？
**A**: 运行测试脚本：
```bash
python Wan_action_fintune/data/scripts/test_parquet_dataloader.py --test all
```

## 数据格式

Parquet 文件 Schema:
```python
{
    "episode_id": string,        # 轨迹唯一标识
    "task_name": string,         # 任务类别
    "instruction": string,       # 指令文本
    "total_frames": int32,       # 轨迹总帧数
    "frame_idx": int32,          # 帧索引
    "frame_data": binary,        # JPEG 压缩的帧数据
    "action": list<float32>,     # 动作向量
    "timestamp": float64,        # 时间戳
}
```

## 优势总结

1. **I/O 效率**: 5-10x 提升（列式存储 + 压缩）
2. **内存效率**: O(buffer_size) vs O(dataset_size)
3. **可扩展性**: 支持任意大小数据集的流式加载
4. **灵活性**: 易于添加新字段（深度、位姿等）
5. **并行性**: 自动分片分配给多个 Worker

## 相关文件

- `train.py`: 简化的训练脚本
- `convert_to_parquet.py`: 数据转换工具
- `test_parquet_dataloader.py`: 测试套件
- `Wan2.1-I2V-14B-480P-parquet.sh`: 示例训练脚本
