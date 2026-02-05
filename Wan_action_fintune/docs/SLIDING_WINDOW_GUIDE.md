# Sliding Window Dataset Usage Guide

## Overview

The sliding window feature allows you to train on long videos (hundreds of frames) by automatically splitting them into overlapping or non-overlapping windows during training. This ensures:

1. **Efficient use of data**: Every frame in your videos can be part of multiple training samples
2. **Cross-video randomization**: Samples from different videos are interleaved randomly during training
3. **Memory efficiency**: Only loads the required window frames on demand (lazy loading)

## Quick Start

### 1. Enable Sliding Window in Training

Add these flags to your training script:

```bash
accelerate launch train.py \
  --dataset_metadata_path ./data/train.csv \
  --num_frames 81 \
  --enable_sliding_window \
  --window_stride 1 \
  # ... other parameters ...
```

### 2. Prepare Your Metadata CSV

**Option A: With num_frames column (Recommended - Fast)**

```csv
prompt,video,action_seq,num_frames
"Take the bottle",video1.mp4,action1.npy,300
"Lift object",video2.mp4,action2.npy,250
"Push button",video3.mp4,action3.npy,180
```

**Option B: Without num_frames column (Slower - reads from video files)**

```csv
prompt,video,action_seq
"Take the bottle",video1.mp4,action1.npy
"Lift object",video2.mp4,action2.npy
"Push button",video3.mp4,action3.npy
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--enable_sliding_window` | flag | False | Enable sliding window sampling |
| `--window_stride` | int | 1 | Stride for sliding window (1 = maximum overlap) |
| `--num_frames` | int | required | Window size (number of frames per sample) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--video_key` | str | "video" | Column name for video paths in CSV |
| `--action_key` | str | "action_seq" | Column name for action paths in CSV |

## How It Works

### Window Generation

For a video with 300 frames and `num_frames=81`, `stride=1`:

```
Video: [frame 0, frame 1, ..., frame 299]

Windows generated:
- Window 0: frames [0:81)
- Window 1: frames [1:82)
- Window 2: frames [2:83)
- ...
- Window 219: frames [219:300)

Total: 220 windows
```

### Cross-Video Randomization

With 3 videos generating 490 total windows:
- Video 0: 220 windows
- Video 1: 170 windows  
- Video 2: 100 windows

DataLoader with `shuffle=True` produces random order like:
```
[Video2-Window45, Video0-Window123, Video1-Window67, Video2-Window2, Video0-Window89, ...]
```

This ensures different videos' samples are interleaved, providing better randomization than processing videos sequentially.

## Examples

### Example 1: Maximum Data Utilization (stride=1)

```bash
python train.py \
  --enable_sliding_window \
  --num_frames 81 \
  --window_stride 1
```

- **Use case**: Small dataset, want maximum training samples
- **Result**: Every possible 81-frame window becomes a sample
- **Note**: High overlap between consecutive samples

### Example 2: Reduced Overlap (stride=10)

```bash
python train.py \
  --enable_sliding_window \
  --num_frames 81 \
  --window_stride 10
```

- **Use case**: Large dataset, reduce redundancy
- **Result**: Windows start at frames 0, 10, 20, 30, ...
- **Benefit**: Less overlap, faster epoch completion

### Example 3: Non-Overlapping Windows (stride=num_frames)

```bash
python train.py \
  --enable_sliding_window \
  --num_frames 81 \
  --window_stride 81
```

- **Use case**: No overlap desired
- **Result**: Windows cover distinct segments [0:81), [81:162), [162:243), ...
- **Note**: Some frames at the end may not be used

## Data Statistics Example

Assuming 3 videos with lengths 300, 250, and 180 frames:

| Configuration | Windows per Video | Total Windows | Epoch Size |
|---------------|-------------------|---------------|------------|
| stride=1, num_frames=81 | 220, 170, 100 | 490 | 490 samples |
| stride=10, num_frames=81 | 22, 17, 10 | 49 | 49 samples |
| stride=81, num_frames=81 | 3, 2, 1 | 6 | 6 samples |

## Important Notes

### ✅ Requirements

1. All videos must have **at least `num_frames` frames**
2. Action sequences must have **same length as videos**
3. Cannot be used with cached data loading mode

### ⚠️ Error Handling

**Error: Video too short**
```
ValueError: Video 2 (path: video3.mp4) has 50 frames, 
which is less than required window size 81
```
**Solution**: Either increase video length or decrease `num_frames`

**Error: Action-video mismatch**
```
ValueError: Video has 300 frames but action sequence has 250 frames
```
**Solution**: Ensure action sequences match video lengths exactly

### 💡 Tips

1. **Add `num_frames` to CSV**: This avoids reading every video file during initialization, significantly speeding up dataset loading

2. **Choose stride wisely**: 
   - Small stride (1-5): More samples, more overlap
   - Medium stride (10-20): Balance between data and efficiency
   - Large stride (≥num_frames): Non-overlapping, minimal redundancy

3. **Monitor training**: With stride=1, you may see very similar consecutive batches. This is expected and actually provides good gradient stability.

## Testing Your Setup

Run the validation script to verify correctness:

```bash
cd Wan_action_fintune
python test_sliding_window.py
```

Expected output:
```
============================================================
✓ ALL TESTS PASSED!
============================================================
```

## Complete Training Example

```bash
# Setup
cd /project/peilab/Puxin/DiffSynth-Studio
export MODEL_PATH="/path/to/model"
export DATA_PATH="/path/to/data"

# Train with sliding window
accelerate launch Wan_action_fintune/train/train.py \
  --model_paths "model.dit=${MODEL_PATH}/dit.safetensors" \
  --model_paths "model.vae=${MODEL_PATH}/vae.safetensors" \
  --model_paths "model.text_encoder=${MODEL_PATH}/text_encoder.safetensors" \
  --model_paths "model.action_encoder=${MODEL_PATH}/action_encoder.safetensors" \
  --trainable_models "model.action_encoder" \
  --dataset_base_path "${DATA_PATH}" \
  --dataset_metadata_path "${DATA_PATH}/train.csv" \
  --data_file_keys "video,action_seq" \
  --extra_inputs "action_seq" \
  --num_frames 81 \
  --height 480 \
  --width 640 \
  --enable_sliding_window \
  --window_stride 1 \
  --action_joint_dim 10 \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --save_steps 1000 \
  --output_path "./checkpoints"
```

## Implementation Details

### Modified Files

1. **`diffsynth/core/data/operators.py`**
   - Added `frame_range` parameter to `LoadVideo`, `LoadGIF`, `LoadActionSequence`
   - Operators can now load specific frame ranges instead of full sequences

2. **`diffsynth/core/data/unified_dataset.py`**
   - Added sliding window mode with parameters
   - New method: `_build_sliding_windows()` - constructs window indices
   - New method: `_get_video_length()` - gets video length from metadata or file
   - Modified `__getitem__()` - handles windowed data loading
   - Modified `__len__()` - returns total windows instead of videos

3. **`Wan_action_fintune/train/train.py`**
   - Added command-line arguments for sliding window configuration
   - Passes parameters to UnifiedDataset initialization

### Architecture

```
Metadata CSV
    ↓
UnifiedDataset (with enable_sliding_window=True)
    ↓
Scan all videos → Build window index: [(video_idx, start_frame), ...]
    ↓
DataLoader (shuffle=True) → Randomly order all windows
    ↓
__getitem__(window_idx) → Extract (video_idx, start_frame)
    ↓
Load video[start:start+num_frames] and action[start:start+num_frames]
    ↓
Return training sample
```

## Troubleshooting

### Issue: Initialization is very slow

**Cause**: Dataset is reading video lengths from files

**Solution**: Add `num_frames` column to your CSV metadata

### Issue: Out of memory errors

**Cause**: Loading too many workers or frames

**Solutions**:
- Reduce `--dataset_num_workers`
- Reduce `--num_frames`
- Check video resolution and reduce if needed

### Issue: Training seems stuck on similar samples

**Cause**: Stride is very small (e.g., 1), consecutive windows are nearly identical

**Solution**: This is expected behavior. The model will still learn. If concerned, increase stride to 5-10.

## Performance Considerations

### Speed vs. Data Trade-off

| Stride | Init Time | Epoch Time | Data Utilization |
|--------|-----------|------------|------------------|
| 1 | Medium | Longest | Maximum (100%) |
| 10 | Fast | Medium | ~10% of frames |
| 81 | Fastest | Shortest | Minimal (~27%) |

### Memory Usage

- **Lazy loading**: Only requested frames are loaded
- **No caching**: Each window is loaded on-demand
- **Multi-worker safe**: Each worker loads independently

## Future Enhancements

Potential improvements for future versions:

1. **Smart stride selection**: Automatically adjust stride based on dataset size
2. **Frame caching**: Cache frequently accessed frames
3. **Temporal augmentation**: Random temporal offsets during training
4. **Multi-scale windows**: Support different window sizes in same batch

## Support

If you encounter issues:

1. Run the test script: `python test_sliding_window.py`
2. Check your CSV has correct paths and num_frames
3. Verify videos are accessible and readable
4. Check that action sequences match video lengths

For questions or bug reports, include:
- Your command line arguments
- Sample of your CSV metadata
- Error messages with full traceback
