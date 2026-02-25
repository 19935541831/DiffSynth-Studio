#!/usr/bin/env python3
"""
Convert Robot Dataset to Parquet Format (Streaming-Optimized)

This script converts robot trajectory data (video + action + instruction)
into sharded Parquet files optimized for streaming dataloader.

Features:
    - Row-per-frame storage for efficient streaming
    - JPEG compression for frames (10-20x size reduction)
    - Automatic sharding for parallel reading
    - Support for RoboTwin dataset format and generic CSV format

Input Formats Supported:
    1. RoboTwin raw format (video + HDF5 + JSON instructions)
    2. Processed CSV format (video paths + npy action paths)

Output:
    Sharded Parquet files with schema:
    - episode_id, task_name, instruction, total_frames
    - frame_idx, frame_data (JPEG bytes), action (float list)

Usage:
    # From RoboTwin raw format
    python convert_to_parquet.py --mode robotwin \\
        --raw_data_dir /path/to/robotwin_raw \\
        --output_dir /path/to/parquet_output

    # From processed CSV format  
    python convert_to_parquet.py --mode csv \\
        --csv_path /path/to/metadata.csv \\
        --base_path /path/to/data \\
        --output_dir /path/to/parquet_output
"""

import os
import sys
import argparse
import json
import h5py
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from diffsynth.core.data.parquet_utils import (
    ParquetTrajectoryWriter,
    encode_frame_to_jpeg,
)


def get_action_layout_indices(
    action_dim: int,
    left_arm_dim: int,
    right_arm_dim: int,
) -> Tuple[List[int], List[int]]:
    """
    Infer arm/gripper indices from action layout:
    [left_arm, left_gripper(1), right_arm, right_gripper(1)]
    """
    expected_dim = left_arm_dim + 1 + right_arm_dim + 1
    if action_dim != expected_dim:
        raise ValueError(
            f"Action dim mismatch: got {action_dim}, expected {expected_dim} "
            f"(left_arm_dim={left_arm_dim}, right_arm_dim={right_arm_dim})"
        )

    left_arm_indices = list(range(0, left_arm_dim))
    right_arm_start = left_arm_dim + 1
    right_arm_indices = list(range(right_arm_start, right_arm_start + right_arm_dim))
    arm_indices = left_arm_indices + right_arm_indices

    gripper_indices = [left_arm_dim, action_dim - 1]
    return arm_indices, gripper_indices


def compute_arm_stats_from_arrays(
    action_arrays: List[np.ndarray],
    left_arm_dim: int,
    right_arm_dim: int,
) -> Dict[str, np.ndarray]:
    """
    Compute dataset-level min/max stats for arm dimensions only.
    """
    arm_min = None
    arm_max = None
    arm_indices = None
    gripper_indices = None

    for actions in action_arrays:
        if actions.ndim != 2:
            raise ValueError(f"Actions must be 2D, got shape={actions.shape}")
        if len(actions) == 0:
            continue

        curr_arm_indices, curr_gripper_indices = get_action_layout_indices(
            action_dim=actions.shape[1],
            left_arm_dim=left_arm_dim,
            right_arm_dim=right_arm_dim,
        )
        if arm_indices is None:
            arm_indices = curr_arm_indices
            gripper_indices = curr_gripper_indices

        arm_values = actions[:, arm_indices]
        curr_min = np.min(arm_values, axis=0)
        curr_max = np.max(arm_values, axis=0)

        if arm_min is None:
            arm_min = curr_min
            arm_max = curr_max
        else:
            arm_min = np.minimum(arm_min, curr_min)
            arm_max = np.maximum(arm_max, curr_max)

    if arm_min is None or arm_max is None or arm_indices is None or gripper_indices is None:
        raise ValueError("No valid action data found for computing normalization stats")

    return {
        "arm_min": arm_min.astype(np.float32),
        "arm_max": arm_max.astype(np.float32),
        "arm_indices": np.array(arm_indices, dtype=np.int32),
        "gripper_indices": np.array(gripper_indices, dtype=np.int32),
    }


def normalize_arm_action(
    action: np.ndarray,
    arm_indices: np.ndarray,
    arm_min: np.ndarray,
    arm_max: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize arm dimensions to [-1, 1], keep gripper dimensions unchanged.
    """
    action = action.astype(np.float32, copy=True)
    denom = np.maximum(arm_max - arm_min, eps)
    action_arm = action[arm_indices]
    action_arm_norm = 2.0 * (action_arm - arm_min) / denom - 1.0
    action[arm_indices] = np.clip(action_arm_norm, -1.0, 1.0)
    return action


def save_normalization_params(
    output_dir: str,
    stats_filename: str,
    left_arm_dim: int,
    right_arm_dim: int,
    arm_indices: np.ndarray,
    gripper_indices: np.ndarray,
    arm_min: np.ndarray,
    arm_max: np.ndarray,
):
    """
    Save normalization parameters for reuse in training/inference.
    """
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, stats_filename)
    payload = {
        "layout": "left_arm,left_gripper,right_arm,right_gripper",
        "left_arm_dim": left_arm_dim,
        "right_arm_dim": right_arm_dim,
        "arm_indices": arm_indices.tolist(),
        "gripper_indices": gripper_indices.tolist(),
        "arm_min": arm_min.tolist(),
        "arm_max": arm_max.tolist(),
        "normalize_range": [-1.0, 1.0],
        "gripper_range_assumed": [0.0, 1.0],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved normalization params: {stats_path}")


def load_joint_action_hdf5(hdf5_path: str) -> np.ndarray:
    """
    Load and concatenate joint action data from HDF5 file (RoboTwin format).
    
    Expected HDF5 structure:
        /joint_action/left_arm: (T, D1)
        /joint_action/left_gripper: (T,) or (T, 1)
        /joint_action/right_arm: (T, D3)
        /joint_action/right_gripper: (T,) or (T, 1)
    
    Returns:
        np.ndarray: Concatenated action array of shape (T, D)
    """
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/joint_action/left_arm"][()]
        left_gripper = f["/joint_action/left_gripper"][()]
        right_arm = f["/joint_action/right_arm"][()]
        right_gripper = f["/joint_action/right_gripper"][()]
    
    # Ensure all are 2D
    if left_arm.ndim == 1:
        left_arm = left_arm[:, None]
    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_arm.ndim == 1:
        right_arm = right_arm[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]
    
    joint_action = np.concatenate([
        left_arm, left_gripper, right_arm, right_gripper
    ], axis=1)
    
    return joint_action.astype(np.float32)


def load_instructions_json(instruction_file: str) -> List[str]:
    """
    Load task instructions from JSON file.
    
    Supports JSON with 'seen' field or plain text format.
    
    Returns:
        List[str]: List of instruction strings
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


def extract_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract all frames from a video file.
    
    Args:
        video_path: Path to video file
    
    Returns:
        List of frames as numpy arrays (H, W, C) in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames


def find_robotwin_episodes(raw_data_dir: str) -> List[Tuple[str, int, str, str, str]]:
    """
    Scan RoboTwin raw data directory and find all episodes.
    
    Returns:
        List of tuples: (task_name, episode_idx, video_path, hdf5_path, instruction_path)
    """
    episodes = []
    raw_path = Path(raw_data_dir)
    
    for task_dir in sorted(raw_path.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        video_dir = task_dir / "aloha-agilex_clean_50" / "video"
        data_dir = task_dir / "aloha-agilex_clean_50" / "data"
        instr_dir = task_dir / "aloha-agilex_clean_50" / "instructions"
        
        if not (video_dir.exists() and data_dir.exists() and instr_dir.exists()):
            # Try alternate structure without subdirectory
            video_dir = task_dir / "video"
            data_dir = task_dir / "data"
            instr_dir = task_dir / "instructions"
            
            if not (video_dir.exists() and data_dir.exists() and instr_dir.exists()):
                print(f"Warning: Skipping {task_name} - missing required subdirectories")
                continue
        
        for video_file in sorted(video_dir.glob("episode*.mp4")):
            episode_name = video_file.stem
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


def convert_robotwin_to_parquet(
    raw_data_dir: str,
    output_dir: str,
    shard_size: int = 10000,
    jpeg_quality: int = 95,
    instruction_mode: str = "random",
    seed: int = 42,
    left_arm_dim: int = 6,
    right_arm_dim: int = 6,
    stats_filename: str = "action_normalization.json",
):
    """
    Convert RoboTwin raw dataset to Parquet format.
    
    Args:
        raw_data_dir: Path to robotwin_dataset_raw directory
        output_dir: Output directory for Parquet shards
        shard_size: Number of rows (frames) per shard
        jpeg_quality: JPEG compression quality
        instruction_mode: How to select instruction ("random", "first", "all")
        seed: Random seed for instruction selection
    """
    np.random.seed(seed)
    
    print("="*70)
    print("RoboTwin to Parquet Converter")
    print("="*70)
    print(f"Input: {raw_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Shard size: {shard_size} rows")
    print(f"JPEG quality: {jpeg_quality}")
    print()
    
    # Find episodes
    print("Scanning for episodes...")
    episodes = find_robotwin_episodes(raw_data_dir)
    print(f"Found {len(episodes)} episodes")
    print()

    # Pass 1: Compute global arm min/max stats
    print("Computing action normalization stats (arm dims only)...")
    robotwin_actions = []
    for _, _, _, hdf5_path, _ in tqdm(episodes, desc="Scanning actions"):
        try:
            robotwin_actions.append(load_joint_action_hdf5(hdf5_path))
        except Exception as e:
            print(f"Warning: failed to load actions from {hdf5_path}: {e}")
    stats = compute_arm_stats_from_arrays(
        action_arrays=robotwin_actions,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
    )
    save_normalization_params(
        output_dir=output_dir,
        stats_filename=stats_filename,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
        arm_indices=stats["arm_indices"],
        gripper_indices=stats["gripper_indices"],
        arm_min=stats["arm_min"],
        arm_max=stats["arm_max"],
    )
    print()
    
    # Create writer
    with ParquetTrajectoryWriter(
        output_dir=output_dir,
        shard_size=shard_size,
        jpeg_quality=jpeg_quality,
    ) as writer:
        
        for task_name, episode_idx, video_path, hdf5_path, instr_path in tqdm(episodes, desc="Converting"):
            episode_id = f"{task_name}_episode{episode_idx}"
            
            try:
                # Load data
                frames = extract_video_frames(video_path)
                actions = load_joint_action_hdf5(hdf5_path)
                instructions = load_instructions_json(instr_path)
                
                # Align lengths
                num_frames = min(len(frames), len(actions))
                frames = frames[:num_frames]
                actions = actions[:num_frames]
                
                # Select instruction
                if instruction_mode == "random":
                    instruction = np.random.choice(instructions)
                elif instruction_mode == "first":
                    instruction = instructions[0]
                else:  # all - use first for now, could expand to multiple rows
                    instruction = instructions[0]
                
                # Write frames
                for frame_idx, (frame, action) in enumerate(zip(frames, actions)):
                    action = normalize_arm_action(
                        action=action,
                        arm_indices=stats["arm_indices"],
                        arm_min=stats["arm_min"],
                        arm_max=stats["arm_max"],
                    )
                    writer.write_frame(
                        episode_id=episode_id,
                        task_name=task_name,
                        instruction=instruction,
                        total_frames=num_frames,
                        frame_idx=frame_idx,
                        frame=frame,
                        action=action,
                        timestamp=frame_idx / 30.0,  # Assume 30 FPS
                    )
                    
            except Exception as e:
                print(f"Error processing {episode_id}: {e}")
                continue
    
    print()
    print("Conversion complete!")


def convert_csv_to_parquet(
    csv_path: str,
    base_path: str,
    output_dir: str,
    shard_size: int = 10000,
    jpeg_quality: int = 95,
    video_col: str = "video",
    action_col: str = "action_seq",
    prompt_col: str = "prompt",
    task_col: Optional[str] = "task",
    episode_col: Optional[str] = "episode",
    left_arm_dim: int = 7,
    right_arm_dim: int = 7,
    stats_filename: str = "action_normalization.json",
):
    """
    Convert processed CSV format dataset to Parquet.
    
    Args:
        csv_path: Path to metadata CSV file
        base_path: Base path for resolving relative paths
        output_dir: Output directory for Parquet shards
        shard_size: Number of rows per shard
        jpeg_quality: JPEG compression quality
        video_col: Column name for video paths
        action_col: Column name for action sequence paths
        prompt_col: Column name for instruction/prompt
        task_col: Column name for task name (optional)
        episode_col: Column name for episode index (optional)
    """
    print("="*70)
    print("CSV to Parquet Converter")
    print("="*70)
    print(f"CSV: {csv_path}")
    print(f"Base path: {base_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV")
    print()

    # Pass 1: Compute global arm min/max stats
    print("Computing action normalization stats (arm dims only)...")
    csv_actions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning actions"):
        action_path = row[action_col]
        if not os.path.isabs(action_path):
            action_path = os.path.join(base_path, action_path)
        try:
            csv_actions.append(np.load(action_path).astype(np.float32))
        except Exception as e:
            print(f"Warning: failed to load actions from {action_path}: {e}")
    stats = compute_arm_stats_from_arrays(
        action_arrays=csv_actions,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
    )
    save_normalization_params(
        output_dir=output_dir,
        stats_filename=stats_filename,
        left_arm_dim=left_arm_dim,
        right_arm_dim=right_arm_dim,
        arm_indices=stats["arm_indices"],
        gripper_indices=stats["gripper_indices"],
        arm_min=stats["arm_min"],
        arm_max=stats["arm_max"],
    )
    print()
    
    # Create writer
    with ParquetTrajectoryWriter(
        output_dir=output_dir,
        shard_size=shard_size,
        jpeg_quality=jpeg_quality,
    ) as writer:
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            # Resolve paths
            video_path = row[video_col]
            action_path = row[action_col]
            
            if not os.path.isabs(video_path):
                video_path = os.path.join(base_path, video_path)
            if not os.path.isabs(action_path):
                action_path = os.path.join(base_path, action_path)
            
            # Get metadata
            instruction = row[prompt_col]
            task_name = row.get(task_col, "unknown") if task_col and task_col in row else "unknown"
            episode_idx = row.get(episode_col, idx) if episode_col and episode_col in row else idx
            episode_id = f"{task_name}_episode{episode_idx}"
            
            try:
                # Load data
                frames = extract_video_frames(video_path)
                actions = np.load(action_path).astype(np.float32)
                
                # Align lengths
                num_frames = min(len(frames), len(actions))
                frames = frames[:num_frames]
                actions = actions[:num_frames]
                
                # Write frames
                for frame_idx, (frame, action) in enumerate(zip(frames, actions)):
                    action = normalize_arm_action(
                        action=action,
                        arm_indices=stats["arm_indices"],
                        arm_min=stats["arm_min"],
                        arm_max=stats["arm_max"],
                    )
                    writer.write_frame(
                        episode_id=episode_id,
                        task_name=task_name,
                        instruction=instruction,
                        total_frames=num_frames,
                        frame_idx=frame_idx,
                        frame=frame,
                        action=action,
                        timestamp=frame_idx / 30.0,
                    )
                    
            except Exception as e:
                print(f"Error processing {episode_id}: {e}")
                continue
    
    print()
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert robot trajectory data to Parquet format"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["robotwin", "csv"],
        required=True,
        help="Input data format mode"
    )
    
    # RoboTwin mode arguments
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        help="Path to RoboTwin raw data directory (for robotwin mode)"
    )
    
    # CSV mode arguments
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to metadata CSV file (for csv mode)"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="",
        help="Base path for resolving relative paths in CSV"
    )
    
    # Common arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Parquet shards"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Number of rows (frames) per Parquet shard (default: 10000)"
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG compression quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--instruction_mode",
        choices=["random", "first", "all"],
        default="random",
        help="How to select instruction for each episode (default: random)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # CSV column names
    parser.add_argument("--video_col", type=str, default="video", help="CSV column for video paths")
    parser.add_argument("--action_col", type=str, default="action_seq", help="CSV column for action paths")
    parser.add_argument("--prompt_col", type=str, default="prompt", help="CSV column for instructions")
    parser.add_argument("--task_col", type=str, default="task", help="CSV column for task name")
    parser.add_argument("--episode_col", type=str, default="episode", help="CSV column for episode index")
    parser.add_argument(
        "--left_arm_dim",
        type=int,
        default=6,
        help="Left arm action dimensions (default: 7)"
    )
    parser.add_argument(
        "--right_arm_dim",
        type=int,
        default=6,
        help="Right arm action dimensions (default: 7)"
    )
    parser.add_argument(
        "--norm_stats_filename",
        type=str,
        default="action_normalization.json",
        help="Filename for saving normalization params in output_dir"
    )
    
    args = parser.parse_args()
    
    if args.mode == "robotwin":
        if not args.raw_data_dir:
            parser.error("--raw_data_dir is required for robotwin mode")
        
        convert_robotwin_to_parquet(
            raw_data_dir=args.raw_data_dir,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            jpeg_quality=args.jpeg_quality,
            instruction_mode=args.instruction_mode,
            seed=args.seed,
            left_arm_dim=args.left_arm_dim,
            right_arm_dim=args.right_arm_dim,
            stats_filename=args.norm_stats_filename,
        )
        
    elif args.mode == "csv":
        if not args.csv_path:
            parser.error("--csv_path is required for csv mode")
        
        convert_csv_to_parquet(
            csv_path=args.csv_path,
            base_path=args.base_path,
            output_dir=args.output_dir,
            shard_size=args.shard_size,
            jpeg_quality=args.jpeg_quality,
            video_col=args.video_col,
            action_col=args.action_col,
            prompt_col=args.prompt_col,
            task_col=args.task_col,
            episode_col=args.episode_col,
            left_arm_dim=args.left_arm_dim,
            right_arm_dim=args.right_arm_dim,
            stats_filename=args.norm_stats_filename,
        )


if __name__ == "__main__":
    main()
