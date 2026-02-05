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


def load_joint_action(hdf5_path: str) -> np.ndarray:
    """
    Load and concatenate joint action data from HDF5 file.
    
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
    
    # Concatenate: left_arm + left_gripper + right_arm + right_gripper
    joint_action = np.concatenate([
        left_arm, left_gripper, right_arm, right_gripper
    ], axis=1)
    
    return joint_action


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
    joint_action = load_joint_action(hdf5_path)
    action_frames = len(joint_action)
    
    # Get video frame count
    video_frames = get_video_frame_count(video_path)
    
    # Align lengths
    if action_frames != video_frames:
        print(f"  Warning: Action ({action_frames}) and video ({video_frames}) "
              f"length mismatch. Truncating to minimum.")
        total_frames = min(action_frames, video_frames)
        joint_action = joint_action[:total_frames]
    else:
        total_frames = action_frames
    
    # Create output filenames
    base_name = f"{task_name}_episode{episode_idx}"
    output_video_path = os.path.join(output_video_dir, f"{base_name}.mp4")
    output_action_path = os.path.join(output_action_dir, f"{base_name}.npy")
    
    # Copy video file
    shutil.copy2(video_path, output_video_path)
    
    # Save action array
    np.save(output_action_path, joint_action)
    
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
        video_dir = task_dir / "video"
        data_dir = task_dir / "data"
        instr_dir = task_dir / "instructions"
        
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
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directories
    output_video_dir = os.path.join(args.output_dir, "videos")
    output_action_dir = os.path.join(args.output_dir, "actions")
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_action_dir, exist_ok=True)
    
    print("="*70)
    print("RoboTwin Dataset Converter (Sliding Window Compatible)")
    print("="*70)
    print(f"Raw data directory: {args.raw_data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Instruction mode: {args.instruction_mode}")
    print(f"Use relative paths: {args.use_relative_paths}")
    print(f"Random seed: {args.seed}")
    print()
    
    # Find all episodes
    print("Scanning for episodes...")
    episodes = find_episodes(args.raw_data_dir)
    print(f"Found {len(episodes)} episodes across tasks")
    print()
    
    # Count by task
    task_counts = {}
    for task_name, _, _, _, _ in episodes:
        task_counts[task_name] = task_counts.get(task_name, 0) + 1
    
    print("Episodes by task:")
    for task_name, count in sorted(task_counts.items()):
        print(f"  {task_name}: {count} episodes")
    print()
    
    # Process all episodes
    print("Processing episodes...")
    metadata_rows = []
    
    for i, (task_name, episode_idx, video_path, hdf5_path, instr_path) in enumerate(episodes):
        print(f"[{i+1}/{len(episodes)}] Processing {task_name}/episode{episode_idx}...")
        
        try:
            result = process_episode(
                task_name=task_name,
                episode_idx=episode_idx,
                video_path=video_path,
                hdf5_path=hdf5_path,
                instruction_file=instr_path,
                output_video_dir=output_video_dir,
                output_action_dir=output_action_dir,
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
    
    print()
    
    # Convert to relative paths if requested
    if args.use_relative_paths:
        print("Converting to relative paths...")
        for row in metadata_rows:
            row['video'] = os.path.relpath(row['video'], args.output_dir)
            row['action_seq'] = os.path.relpath(row['action_seq'], args.output_dir)
    
    # Create DataFrame and save CSV
    print("Saving metadata CSV...")
    df = pd.DataFrame(metadata_rows)
    
    # Reorder columns for clarity
    column_order = ['prompt', 'video', 'action_seq', 'num_frames', 'task', 'episode']
    df = df[column_order]
    
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    
    # Print statistics
    print()
    print("="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"Total metadata rows: {len(df)}")
    print(f"Unique episodes: {len(episodes)}")
    print(f"Unique tasks: {df['task'].nunique()}")
    print(f"Unique prompts: {df['prompt'].nunique()}")
    print()
    print(f"Total frames: {df['num_frames'].sum():,}")
    print(f"Average frames per episode: {df['num_frames'].mean():.1f}")
    print(f"Min frames: {df['num_frames'].min()}")
    print(f"Max frames: {df['num_frames'].max()}")
    print()
    print("Output files:")
    print(f"  Videos: {output_video_dir}")
    print(f"  Actions: {output_action_dir}")
    print(f"  Metadata CSV: {csv_path}")
    print()
    print("To train with sliding window sampling:")
    print(f"  python train.py \\")
    print(f"    --dataset_base_path {args.output_dir} \\")
    print(f"    --dataset_metadata_path {csv_path} \\")
    print(f"    --num_frames 81 \\")
    print(f"    --enable_sliding_window \\")
    print(f"    --window_stride 1 \\")
    print(f"    --data_file_keys video,action_seq \\")
    print(f"    --extra_inputs action_seq")
    print()


if __name__ == "__main__":
    main()
