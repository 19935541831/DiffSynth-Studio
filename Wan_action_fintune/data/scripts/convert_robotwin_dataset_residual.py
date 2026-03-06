#!/usr/bin/env python3
"""
RoboTwin Raw Dataset to Training Format Converter (Sliding Window Compatible)

This script converts the raw RoboTwin dataset into a format compatible with
the sliding window training feature. Unlike the old approach that pre-generates
all clips, this script:

1. Keeps videos in full-length format
2. Saves full-length action sequences
3. Creates a metadata CSV with num_frames column
4. Lets the sliding window dataloader handle windowing during training

Raw data structure:
  robotwin_dataset_raw/
    task_name/
      video/episode0.mp4
      data/episode0.hdf5
      instructions/episode0.json

Output structure:
  robotwin_dataset_processed/
    videos/task_name_episode0.mp4
    actions/task_name_episode0.npy
    metadata.csv (with prompt, video, action_seq, num_frames columns)
"""

import os
import sys
import argparse
import h5py
import numpy as np
import pandas as pd
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
from scipy.spatial.transform import Rotation as R

def convert_pose_to_euler(pose_array: np.ndarray) -> np.ndarray:
    """
    Convert (T, 7) pose array [x, y, z, qw, qx, qy, qz]
    to (T, 6) pose array [x, y, z, roll, pitch, yaw].
    """
    pos = pose_array[:, :3]
    # Reorder [qw, qx, qy, qz] to [qx, qy, qz, qw] for scipy R
    qw = pose_array[:, 3:4]
    qx = pose_array[:, 4:5]
    qy = pose_array[:, 5:6]
    qz = pose_array[:, 6:7]
    quat = np.concatenate([qx, qy, qz, qw], axis=-1)
    
    # R.from_quat expects format [x, y, z, w]
    euler = R.from_quat(quat).as_euler('xyz', degrees=False)
    return np.concatenate([pos, euler], axis=-1)

def load_endpose_action(hdf5_path: str) -> np.ndarray:
    """
    Load and concatenate endpose action data from HDF5 file.
    
    Expected HDF5 structure:
      /endpose/left_endpose: (T, 7)
      /endpose/left_gripper: (T,) or (T, 1)
      /endpose/right_endpose: (T, 7)
      /endpose/right_gripper: (T,) or (T, 1)
    
    Returns:
        np.ndarray: Concatenated action array of shape (T, D)
    """
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/endpose/left_endpose"][()]
        left_gripper = f["/endpose/left_gripper"][()]
        right_arm = f["/endpose/right_endpose"][()]
        right_gripper = f["/endpose/right_gripper"][()]
    
    # Ensure all are 2D
    if left_arm.ndim == 1:
        left_arm = left_arm[:, None]
    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_arm.ndim == 1:
        right_arm = right_arm[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]
        
    # Convert quat to euler
    left_arm = convert_pose_to_euler(left_arm)
    right_arm = convert_pose_to_euler(right_arm)
    
    # Concatenate: left_arm + left_gripper + right_arm + right_gripper
    endpose_action = np.concatenate([
        left_arm, left_gripper, right_arm, right_gripper
    ], axis=1).astype(np.float32)
    
    return endpose_action


def compute_residual_actions(actions: np.ndarray) -> np.ndarray:
    """
    Convert absolute actions into residual actions.

    residual[t] = actions[t] - actions[t-1], residual[0] = 0
    """
    if actions.ndim != 2:
        raise ValueError(f"Actions must be 2D, got shape={actions.shape}")
    if len(actions) == 0:
        return actions.astype(np.float32)

    residual = np.zeros_like(actions, dtype=np.float32)
    residual[1:] = actions[1:] - actions[:-1]
    return residual


def compute_zscore_stats_from_residual_arrays(
    residual_arrays: List[np.ndarray],
    eps: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """
    Compute global per-dimension mean/std for residual actions.
    """
    total_count = 0
    total_sum = None
    total_sumsq = None
    action_dim = None

    for residual in residual_arrays:
        if residual.ndim != 2:
            raise ValueError(f"Residual actions must be 2D, got shape={residual.shape}")
        if len(residual) == 0:
            continue

        if action_dim is None:
            action_dim = residual.shape[1]
            total_sum = np.zeros(action_dim, dtype=np.float64)
            total_sumsq = np.zeros(action_dim, dtype=np.float64)
        elif residual.shape[1] != action_dim:
            raise ValueError(
                f"Action dim mismatch across dataset: {residual.shape[1]} vs {action_dim}"
            )

        total_count += residual.shape[0]
        total_sum += residual.sum(axis=0, dtype=np.float64)
        total_sumsq += np.square(residual, dtype=np.float64).sum(axis=0, dtype=np.float64)

    if total_count == 0 or action_dim is None:
        raise ValueError("No valid residual action data found for computing z-score stats")

    mean = total_sum / total_count
    var = total_sumsq / total_count - np.square(mean)
    var = np.maximum(var, eps ** 2)
    std = np.sqrt(var)

    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
    }


def normalize_residual_action_zscore(
    residual_action: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    sigma_clip: float = 3.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Z-score normalize residual action and clip to +/- sigma_clip.
    """
    residual_action = residual_action.astype(np.float32, copy=True)
    denom = np.maximum(std, eps)
    z = (residual_action - mean) / denom
    z = np.clip(z, -sigma_clip, sigma_clip)
    return z.astype(np.float32)


def save_normalization_params(
    output_dir: str,
    stats_filename: str,
    action_dim: int,
    mean: np.ndarray,
    std: np.ndarray,
    sigma_clip: float,
):
    """
    Save residual + z-score normalization parameters for training/inference reuse.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, stats_filename)
    payload = {
        "representation": "residual_action",
        "residual_definition": "delta_t = action_t - action_{t-1}, delta_0 = 0",
        "normalization": "zscore",
        "action_dim": action_dim,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "sigma_clip": float(sigma_clip),
        "clip_range": [-float(sigma_clip), float(sigma_clip)],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved normalization params: {stats_path}")


def load_instructions(instruction_file: str) -> List[str]:
    """
    Load task instructions from JSON file.
    
    Supports JSON with 'seen' field or plain text format.
    
    Returns:
        List[str]: List of instruction strings (from 'seen' field if available)
    """
    with open(instruction_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        data = json.loads(content)
        if isinstance(data, dict) and 'seen' in data:
            return data['seen']
        elif isinstance(data, list):
            return data
        else:
            return [content]
    except json.JSONDecodeError:
        return [content]


def get_video_frame_count(video_path: str) -> int:
    """Get the number of frames in a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return frame_count


def process_episode(
    task_name: str,
    episode_idx: int,
    video_path: str,
    hdf5_path: str,
    instruction_file: str,
    output_video_dir: str,
    output_action_dir: str,
    mean: np.ndarray,
    std: np.ndarray,
    sigma_clip: float = 3.0,
    instruction_mode: str = "random_per_episode"
) -> Dict:
    """
    Process a single episode: copy video, save action, return metadata.
    
    Args:
        task_name: Name of the task (e.g., "adjust_bottle")
        episode_idx: Episode number
        video_path: Path to source video
        hdf5_path: Path to HDF5 action data
        instruction_file: Path to instruction JSON
        output_video_dir: Directory to save processed videos
        output_action_dir: Directory to save action arrays
        instruction_mode: How to assign instructions
            - "random_per_episode": Pick one random instruction for the whole episode
            - "first": Use first instruction only
            - "all_random": Include all instructions (one metadata row per instruction)
    
    Returns:
        Dict or List[Dict]: Metadata row(s) for CSV
    """
    # Load instructions
    instructions = load_instructions(instruction_file)
    
    # Load action data
    endpose_action = load_endpose_action(hdf5_path)
    residual_action = compute_residual_actions(endpose_action)
    action_norm = normalize_residual_action_zscore(
        residual_action=residual_action,
        mean=mean,
        std=std,
        sigma_clip=sigma_clip,
    )
    
    action_frames = len(action_norm)
    
    # Get video frame count
    video_frames = get_video_frame_count(video_path)
    
    # Align lengths
    if action_frames != video_frames:
        print(f"  Warning: Action ({action_frames}) and video ({video_frames}) "
              f"length mismatch. Truncating to minimum.")
        total_frames = min(action_frames, video_frames)
        action_norm = action_norm[:total_frames]
    else:
        total_frames = action_frames
    
    # Create output filenames
    base_name = f"{task_name}_episode{episode_idx}"
    output_video_path = os.path.join(output_video_dir, f"{base_name}.mp4")
    output_action_path = os.path.join(output_action_dir, f"{base_name}.npy")
    
    # Copy video file
    shutil.copy2(video_path, output_video_path)
    
    # Save action array
    np.save(output_action_path, action_norm)
    
    # Create metadata based on instruction mode
    if instruction_mode == "random_per_episode":
        # Pick one random instruction for this episode
        instruction = np.random.choice(instructions)
        return {
            'prompt': instruction,
            'video': output_video_path,
            'action_seq': output_action_path,
            'num_frames': total_frames,
            'task': task_name,
            'episode': episode_idx
        }
    
    elif instruction_mode == "first":
        # Use only the first instruction
        return {
            'prompt': instructions[0],
            'video': output_video_path,
            'action_seq': output_action_path,
            'num_frames': total_frames,
            'task': task_name,
            'episode': episode_idx
        }
    
    elif instruction_mode == "all_random":
        # Create one row per instruction (shuffled)
        rows = []
        shuffled_instructions = instructions.copy()
        np.random.shuffle(shuffled_instructions)
        for instruction in shuffled_instructions:
            rows.append({
                'prompt': instruction,
                'video': output_video_path,
                'action_seq': output_action_path,
                'num_frames': total_frames,
                'task': task_name,
                'episode': episode_idx
            })
        return rows
    
    else:
        raise ValueError(f"Unknown instruction_mode: {instruction_mode}")


def find_episodes(raw_data_dir: str) -> List[Tuple[str, int, str, str, str]]:
    """
    Scan raw data directory and find all episodes.
    
    Returns:
        List of tuples: (task_name, episode_idx, video_path, hdf5_path, instruction_path)
    """
    episodes = []
    raw_path = Path(raw_data_dir)
    
    # Iterate through task directories
    for task_dir in sorted(raw_path.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        video_dir = task_dir / "aloha-agilex_clean_50" / "video"
        data_dir = task_dir / "aloha-agilex_clean_50" / "data"
        instr_dir = task_dir / "aloha-agilex_clean_50" / "instructions"
        
        if not (video_dir.exists() and data_dir.exists() and instr_dir.exists()):
            print(f"Warning: Skipping {task_name} - missing required subdirectories")
            continue
        
        # Find all episodes in this task
        for video_file in sorted(video_dir.glob("episode*.mp4")):
            episode_name = video_file.stem  # e.g., "episode0"
            episode_idx = int(episode_name.replace("episode", ""))
            
            hdf5_file = data_dir / f"{episode_name}.hdf5"
            instr_file = instr_dir / f"{episode_name}.json"
            
            if hdf5_file.exists() and instr_file.exists():
                episodes.append((
                    task_name,
                    episode_idx,
                    str(video_file),
                    str(hdf5_file),
                    str(instr_file)
                ))
            else:
                print(f"Warning: Missing data for {task_name}/{episode_name}")
    
    return episodes


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoboTwin raw dataset to training format (sliding window compatible)"
    )
    parser.add_argument(
        "--raw_data_dir",
        required=True,
        help="Path to robotwin_dataset_raw directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--val_output_dir",
        type=str,
        default=None,
        help="Output directory for validation dataset (last episode of each task)"
    )
    parser.add_argument(
        "--instruction_mode",
        choices=["random_per_episode", "first", "all_random"],
        default="random_per_episode",
        help="How to assign instructions to episodes (default: random_per_episode)"
    )
    parser.add_argument(
        "--use_relative_paths",
        action="store_true",
        help="Use relative paths in CSV (relative to output_dir)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for instruction selection (default: 42)"
    )
    parser.add_argument(
        "--sigma_clip",
        type=float,
        default=3.0,
        help="Clip z-score normalized action into [-sigma_clip, sigma_clip] (default: 3.0)",
    )
    parser.add_argument(
        "--norm_stats_filename",
        type=str,
        default="action_residual_zscore.json",
        help="Filename for saving residual z-score params in output_dir",
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print("="*70)
    print("RoboTwin Dataset Converter (Sliding Window Compatible)")
    print("="*70)
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Output directory (Train): {args.output_dir}")
    if args.val_output_dir:
        print(f"Output directory (Val): {args.val_output_dir}")
    print(f"Instruction mode: {args.instruction_mode}")
    print(f"Use relative paths: {args.use_relative_paths}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Find all episodes
    print("Scanning for episodes...")
    episodes = find_episodes(args.raw_data_dir)
    print(f"Found {len(episodes)} total episodes across tasks")
    print()
    
    # Split into train and val episodes
    train_episodes = []
    val_episodes = []
    
    if args.val_output_dir:
        # Group by task
        task_episodes = {}
        for ep in episodes:
            task_name = ep[0]
            if task_name not in task_episodes:
                task_episodes[task_name] = []
            task_episodes[task_name].append(ep)
        
        for task_name, eps in task_episodes.items():
            # Sort by episode_idx ascending
            eps.sort(key=lambda x: x[1])
            # Last episode goes to val
            if len(eps) > 1:
                train_episodes.extend(eps[:-1])
                val_episodes.append(eps[-1])
            else:
                print(f"Warning: Task {task_name} only has 1 episode. Putting it in train.")
                train_episodes.extend(eps)
    else:
        train_episodes = episodes
        
    print(f"Train episodes: {len(train_episodes)}")
    if args.val_output_dir:
        print(f"Val episodes: {len(val_episodes)}")
    print()
    
    # Count by task
    task_counts = {}
    for task_name, _, _, _, _ in train_episodes:
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
    
    print("Train Episodes by task:")
    for task_name, count in sorted(task_counts.items()):
        print(f"  {task_name}: {count} episodes")
    print()
    
    # Compute residual action z-score stats from training data
    print("Computing residual action z-score stats from training data...")
    from tqdm import tqdm
    residual_arrays = []
    for _, _, _, hdf5_path, _ in tqdm(train_episodes, desc="Scanning actions"):
        try:
            actions = load_endpose_action(hdf5_path)
            residual_arrays.append(compute_residual_actions(actions))
        except Exception as e:
            print(f"Warning: failed to load actions from {hdf5_path}: {e}")

    stats = compute_zscore_stats_from_residual_arrays(residual_arrays)
    save_normalization_params(
        output_dir=args.output_dir,
        stats_filename=args.norm_stats_filename,
        action_dim=len(stats["mean"]),
        mean=stats["mean"],
        std=stats["std"],
        sigma_clip=args.sigma_clip,
    )
    if args.val_output_dir:
        save_normalization_params(
            output_dir=args.val_output_dir,
            stats_filename=args.norm_stats_filename,
            action_dim=len(stats["mean"]),
            mean=stats["mean"],
            std=stats["std"],
            sigma_clip=args.sigma_clip,
        )
    print()
    
    def process_split(split_episodes, out_dir, split_name):
        _output_video_dir = os.path.join(out_dir, "videos")
        _output_action_dir = os.path.join(out_dir, "actions")
        os.makedirs(_output_video_dir, exist_ok=True)
        os.makedirs(_output_action_dir, exist_ok=True)
        
        print(f"Processing {split_name} episodes...")
        metadata_rows = []
        
        for i, (task_name, episode_idx, video_path, hdf5_path, instr_path) in enumerate(split_episodes):
            print(f"[{i+1}/{len(split_episodes)}] Processing {task_name}/episode{episode_idx}...")
            
            try:
                result = process_episode(
                    task_name=task_name,
                    episode_idx=episode_idx,
                    video_path=video_path,
                    hdf5_path=hdf5_path,
                    instruction_file=instr_path,
                    output_video_dir=_output_video_dir,
                    output_action_dir=_output_action_dir,
                    mean=stats["mean"],
                    std=stats["std"],
                    sigma_clip=args.sigma_clip,
                    instruction_mode=args.instruction_mode
                )
                
                # Handle single row or multiple rows
                if isinstance(result, list):
                    metadata_rows.extend(result)
                else:
                    metadata_rows.append(result)
                    
            except Exception as e:
                print(f"  ERROR: Failed to process {task_name}/episode{episode_idx}: {e}")
                continue
                
        if args.use_relative_paths:
            print("Converting to relative paths...")
            for row in metadata_rows:
                row['video'] = os.path.relpath(row['video'], out_dir)
                row['action_seq'] = os.path.relpath(row['action_seq'], out_dir)
        
        df = pd.DataFrame(metadata_rows)
        column_order = ['prompt', 'video', 'action_seq', 'num_frames', 'task', 'episode']
        if not df.empty:
            df = df[column_order]
            csv_path = os.path.join(out_dir, "metadata.csv")
            df.to_csv(csv_path, index=False)
            return df, csv_path, _output_video_dir, _output_action_dir
        return None, None, None, None

    train_df, train_csv_path, train_video_dir, train_action_dir = process_split(train_episodes, args.output_dir, "train")
    
    val_df = None
    if args.val_output_dir and val_episodes:
        print()
        val_df, val_csv_path, val_video_dir, val_action_dir = process_split(val_episodes, args.val_output_dir, "val")
    
    # Print statistics
    print()
    print("="*70)
    print("Conversion Complete!")
    print("="*70)
    if train_df is not None:
        print(f"Train metadata rows: {len(train_df)}")
        print(f"Train tasks: {train_df['task'].nunique()}, prompts: {train_df['prompt'].nunique()}")
        print(f"Train total frames: {train_df['num_frames'].sum():,}")
    if val_df is not None:
        print()
        print(f"Val metadata rows: {len(val_df)}")
        print(f"Val tasks: {val_df['task'].nunique()}, prompts: {val_df['prompt'].nunique()}")
        print(f"Val total frames: {val_df['num_frames'].sum():,}")
    
    print()
    print("Train Output files:")
    print(f"  Videos: {train_video_dir}")
    print(f"  Actions: {train_action_dir}")
    print(f"  Metadata CSV: {train_csv_path}")
    print()
    print("To train with sliding window sampling:")
    print(f"  python train.py \\")
    print(f"    --dataset_base_path {args.output_dir} \\")
    print(f"    --dataset_metadata_path {train_csv_path} \\")
    print(f"    --num_frames 81 \\")
    print(f"    --enable_sliding_window \\")
    print(f"    --window_stride 1 \\")
    print(f"    --data_file_keys video,action_seq \\")
    print(f"    --extra_inputs action_seq")
    print()


if __name__ == "__main__":
    main()
