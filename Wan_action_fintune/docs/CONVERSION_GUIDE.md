# RoboTwin Dataset Conversion Guide

## Overview

This guide explains how to convert the raw RoboTwin dataset into a format compatible with the sliding window training feature.

## Key Advantage

**Old approach** (using `convert_episode.py`):
- Pre-generates thousands of small clip files
- Uses stride=1 to create every possible window
- Results in huge disk space usage and slow initialization

**New approach** (using `convert_robotwin_dataset.py`):
- Keeps videos full-length
- Creates metadata CSV with `num_frames` column
- Lets sliding window dataloader handle windowing during training
- Much more efficient!

## Raw Data Structure

```
robotwin_dataset_raw/
  adjust_bottle/
    video/
      episode0.mp4
      episode1.mp4
      ...
    data/
      episode0.hdf5
      episode1.hdf5
      ...
    instructions/
      episode0.json
      episode1.json
      ...
  pick_cup/
    video/...
    data/...
    instructions/...
  ...
```

## Usage

### Basic Usage

```bash
python data/scripts/convert_robotwin_dataset.py \
  --raw_data_dir /path/to/robotwin_dataset_raw \
  --output_dir /path/to/robotwin_dataset_processed
```

### With Options

```bash
python data/scripts/convert_robotwin_dataset.py \
  --raw_data_dir ./data/robotwin_dataset_raw \
  --output_dir ./data/robotwin_dataset_processed \
  --instruction_mode random_per_episode \
  --use_relative_paths \
  --seed 42
```

## Parameters

### Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--raw_data_dir` | Path to raw dataset directory (with task subdirectories) |
| `--output_dir` | Output directory for processed dataset |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--instruction_mode` | `random_per_episode` | How to assign instructions (see below) |
| `--use_relative_paths` | `False` | Use relative paths in CSV |
| `--seed` | `42` | Random seed for instruction selection |

### Instruction Modes

**`random_per_episode`** (Recommended):
- Randomly selects one instruction from the "seen" list per episode
- One metadata row per episode
- Good for training with diverse instructions

**`first`**:
- Uses only the first instruction from each episode
- One metadata row per episode
- Simpler, less variety

**`all_random`**:
- Creates one metadata row for each instruction (shuffled order)
- Multiple rows per episode (one per instruction)
- Maximum variety but longer epoch time

## Output Structure

```
robotwin_dataset_processed/
  videos/
    adjust_bottle_episode0.mp4
    adjust_bottle_episode1.mp4
    pick_cup_episode0.mp4
    ...
  actions/
    adjust_bottle_episode0.npy
    adjust_bottle_episode1.npy
    pick_cup_episode0.npy
    ...
  metadata.csv
```

### Metadata CSV Format

With `--use_relative_paths` (recommended):

```csv
prompt,video,action_seq,num_frames,task,episode
"Pick up the bottle...",videos/adjust_bottle_episode0.mp4,actions/adjust_bottle_episode0.npy,523,adjust_bottle,0
"Lift the cup...",videos/pick_cup_episode0.mp4,actions/pick_cup_episode0.npy,612,pick_cup,0
...
```

Without relative paths:

```csv
prompt,video,action_seq,num_frames,task,episode
"Pick up the bottle...",/full/path/to/videos/adjust_bottle_episode0.mp4,/full/path/to/actions/adjust_bottle_episode0.npy,523,adjust_bottle,0
...
```

## Complete Example

### Step 1: Convert Dataset

```bash
cd /project/peilab/Puxin/DiffSynth-Studio/Wan_action_fintune

python data/scripts/convert_robotwin_dataset.py \
  --raw_data_dir /project/peilab/Puxin/Wan_action/data/robotwin_dataset_raw \
  --output_dir /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train \
  --instruction_mode random_per_episode \
  --use_relative_paths \
  --seed 42
```

Expected output:
```
======================================================================
RoboTwin Dataset Converter (Sliding Window Compatible)
======================================================================
Raw data directory: /project/peilab/Puxin/Wan_action/data/robotwin_dataset_raw
Output directory: /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train
Instruction mode: random_per_episode
Use relative paths: True
Random seed: 42

Scanning for episodes...
Found 15 episodes across tasks

Episodes by task:
  adjust_bottle: 3 episodes
  pick_cup: 5 episodes
  push_button: 7 episodes

Processing episodes...
[1/15] Processing adjust_bottle/episode0...
[2/15] Processing adjust_bottle/episode1...
...

======================================================================
Conversion Complete!
======================================================================
Total metadata rows: 15
Unique episodes: 15
Unique tasks: 3
Unique prompts: 15

Total frames: 7,850
Average frames per episode: 523.3
Min frames: 412
Max frames: 678

Output files:
  Videos: /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train/videos
  Actions: /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train/actions
  Metadata CSV: /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train/metadata.csv
```

### Step 2: Train with Sliding Window

```bash
accelerate launch train/train.py \
  --dataset_base_path /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train \
  --dataset_metadata_path /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train/metadata.csv \
  --data_file_keys video,action_seq \
  --extra_inputs action_seq \
  --num_frames 81 \
  --enable_sliding_window \
  --window_stride 1 \
  --action_joint_dim 10 \
  --model_paths "model.dit=/path/to/dit.safetensors" \
  --model_paths "model.vae=/path/to/vae.safetensors" \
  --model_paths "model.text_encoder=/path/to/text_encoder.safetensors" \
  --model_paths "model.action_encoder=/path/to/action_encoder.safetensors" \
  --trainable_models "model.action_encoder" \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --save_steps 500
```

## Data Statistics

### Example Dataset (after conversion)

Assuming 15 episodes with average 500 frames each:

| Configuration | Total Samples per Epoch |
|---------------|-------------------------|
| **No sliding window** | 15 (one sample per episode) |
| **Sliding window, stride=1** | ~6,300 (all possible 81-frame windows) |
| **Sliding window, stride=10** | ~630 (every 10th window) |
| **Sliding window, stride=81** | ~78 (non-overlapping windows) |

With stride=1 and num_frames=81:
- Episode with 500 frames → 420 windows
- Episode with 600 frames → 520 windows
- Total across 15 episodes → thousands of training samples!

## Troubleshooting

### Issue: Missing episodes

**Error message:**
```
Warning: Skipping task_name - missing required subdirectories
Warning: Missing data for task_name/episode0
```

**Solution:** Ensure your raw data has the correct structure:
```
task_name/
  video/episode0.mp4
  data/episode0.hdf5
  instructions/episode0.json
```

### Issue: Action-video length mismatch

**Warning message:**
```
Warning: Action (523) and video (520) length mismatch. Truncating to minimum.
```

**Solution:** This is usually fine - the script automatically aligns them. If mismatches are large, check your raw data for corruption.

### Issue: Cannot open video

**Error:**
```
ValueError: Cannot open video: /path/to/video.mp4
```

**Solutions:**
- Check video file exists and is readable
- Try playing the video manually to verify it's not corrupted
- Ensure OpenCV is properly installed

## Comparison: Old vs New Approach

### Old Approach (convert_episode.py)

```bash
# For 1 episode with 500 frames, num_frames=81, stride=1:
# Generates: 420 clip files
# Disk usage: ~420 * video_size (huge!)
# CSV rows: 420 entries

python convert_episode.py \
  --episode_video episode0.mp4 \
  --hdf5_path episode0.hdf5 \
  --instruction_file episode0.json \
  --num_frames 81 \
  --output_video_dir ./clips/videos \
  --output_action_dir ./clips/actions \
  --output_csv_path ./clips/metadata.csv
```

### New Approach (convert_robotwin_dataset.py)

```bash
# For same episode:
# Generates: 1 video file, 1 action file
# Disk usage: 1 * video_size (much less!)
# CSV rows: 1 entry (with num_frames=500)
# Sliding window handles the rest during training!

python convert_robotwin_dataset.py \
  --raw_data_dir ./robotwin_dataset_raw \
  --output_dir ./robotwin_dataset_train \
  --use_relative_paths
```

**Benefits:**
- ✅ 420x less disk space
- ✅ Much faster conversion
- ✅ Cleaner organization
- ✅ Easier to manage
- ✅ More flexible (change stride without re-converting)

## Advanced Usage

### Filter by Task

To process only specific tasks, modify the script or filter afterward:

```python
# After conversion, filter metadata.csv
import pandas as pd

df = pd.read_csv('metadata.csv')
df_filtered = df[df['task'].isin(['adjust_bottle', 'pick_cup'])]
df_filtered.to_csv('metadata_filtered.csv', index=False)
```

### Split Train/Val

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('metadata.csv')

# Split by episode (not by row) to avoid data leakage
episodes = df[['task', 'episode']].drop_duplicates()
train_eps, val_eps = train_test_split(episodes, test_size=0.2, random_state=42)

# Create train/val splits
train_df = df.merge(train_eps, on=['task', 'episode'])
val_df = df.merge(val_eps, on=['task', 'episode'])

train_df.to_csv('metadata_train.csv', index=False)
val_df.to_csv('metadata_val.csv', index=False)

print(f"Train episodes: {len(train_eps)}")
print(f"Val episodes: {len(val_eps)}")
```

### Check Action Dimensions

```python
import numpy as np
import pandas as pd

df = pd.read_csv('metadata.csv')

# Check first action file
action_path = df.iloc[0]['action_seq']
action = np.load(action_path)

print(f"Action shape: {action.shape}")
print(f"Joint dimension: {action.shape[1]}")
```

Use this joint dimension in training:
```bash
--action_joint_dim <joint_dimension>
```

## Next Steps

After conversion:

1. **Verify the output:**
   ```bash
   ls -lh robotwin_dataset_processed/videos/ | head
   ls -lh robotwin_dataset_processed/actions/ | head
   head -n 5 robotwin_dataset_processed/metadata.csv
   ```

2. **Check action dimensions:**
   ```python
   import numpy as np
   action = np.load('robotwin_dataset_processed/actions/adjust_bottle_episode0.npy')
   print(f"Shape: {action.shape}")  # Should be (T, joint_dim)
   ```

3. **Start training with sliding window** (see SLIDING_WINDOW_GUIDE.md)

## Support

For issues or questions about the conversion process, check:
- This guide for common issues
- `SLIDING_WINDOW_GUIDE.md` for training details
- The script's help: `python convert_robotwin_dataset.py --help`
