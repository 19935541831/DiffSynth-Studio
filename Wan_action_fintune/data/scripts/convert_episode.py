#!/usr/bin/env python3
"""
RoboTwin Episode to Structured Format Converter
 
Converts a single RoboTwin episode into a set of fixed-length video-action clips
using a sliding window approach. Each clip consists of:
  - A video segment of `num_frames` consecutive frames
  - A corresponding joint action sequence of shape (num_frames, joint_dim)
  - Metadata linking the clip to a high-level task instruction
 
The output includes:
  - Video clips saved as MP4 files
  - Action sequences saved as .npy files
  - A CSV manifest mapping prompts, videos, and action sequences
"""
 
import os
import argparse
import h5py
import numpy as np
import pandas as pd
import cv2
import json
from pathlib import Path
 
 
def load_episode_data(hdf5_path):
    """
    Load joint action data from an HDF5 file.

    Expected structure in HDF5:
      - /joint_action/left_arm: (T, D1)
      - /joint_action/left_gripper: (T,) or (T, 1)
      - /joint_action/right_arm: (T, D3)
      - /joint_action/right_gripper: (T,) or (T, 1)

    Returns:
        joint_action (np.ndarray): Concatenated array of shape (T, D),
                                   where D = D1 + 1 + D3 + 1 (assuming scalar grippers).
    """
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/joint_action/left_arm"][()]
        left_gripper = f["/joint_action/left_gripper"][()]
        right_arm = f["/joint_action/right_arm"][()]
        right_gripper = f["/joint_action/right_gripper"][()]

    # Ensure all components are 2D: (T, ?)
    if left_arm.ndim == 1:
        left_arm = left_arm[:, None]
    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_arm.ndim == 1:
        right_arm = right_arm[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]

    # Concatenate along feature dimension
    joint_action = np.concatenate([
        left_arm, left_gripper, right_arm, right_gripper
    ], axis=1)

    return joint_action
 
 
def load_instructions(instruction_file):
    """
    Load task instructions from a file.
    
    Supports two formats:
      1. JSON format with 'seen' and/or 'unseen' fields - returns only 'seen' list
      2. Plain text format - returns single instruction as a list
    
    Args:
        instruction_file (str): Path to instruction file (.json or .txt)
    
    Returns:
        List[str]: List of instruction strings from 'seen' field
    
    Raises:
        ValueError: If JSON file doesn't contain 'seen' field or is empty
    """
    with open(instruction_file, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    # Try to parse as JSON first
    try:
        data = json.loads(content)
        if isinstance(data, dict) and 'seen' in data:
            instructions = data['seen']
            if not instructions:
                raise ValueError("'seen' field is empty in instruction file")
            return instructions
        elif isinstance(data, list):
            # Direct list format
            return data
        else:
            # JSON but unexpected format, treat as plain text
            return [content]
    except json.JSONDecodeError:
        # Plain text format - single instruction
        return [content]
 
 
def load_video_frames(video_path):
    """
    Read all frames from a video file into a list of NumPy arrays.
 
    Args:
        video_path (str): Path to the input video file (e.g., .mp4).
 
    Returns:
        List[np.ndarray]: List of BGR frames in temporal order.
 
    Raises:
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
 
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
 
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
 
    cap.release()
    return frames
 
 
def save_video_clip(frames, output_path, fps=30):
    """
    Save a list of frames as an MP4 video file.
 
    Args:
        frames (List[np.ndarray]): List of BGR frames.
        output_path (str): Output video file path.
        fps (int): Frames per second for the output video.
 
    Returns:
        bool: True if video was successfully written, False if no frames provided.
    """
    if len(frames) == 0:
        return False
 
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
 
    for frame in frames:
        out.write(frame)
 
    out.release()
    return True
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Convert a RoboTwin episode into structured video-action clips."
    )
    parser.add_argument("--episode_video", required=True,
                       help="Path to the original episode video (e.g., .mp4).")
    parser.add_argument("--hdf5_path", required=True,
                       help="Path to the HDF5 file containing joint_action data.")
    parser.add_argument("--instruction_file", required=True,
                       help="Path to instruction file (JSON with 'seen' field or plain text).")
    parser.add_argument("--num_frames", type=int, required=True,
                       help="Number of frames per output clip (sliding window size).")
    parser.add_argument("--output_video_dir", required=True,
                       help="Directory to save output video clips.")
    parser.add_argument("--output_action_dir", required=True,
                       help="Directory to save output action sequences (.npy files).")
    parser.add_argument("--output_csv_path", required=True,
                       help="Full path to the output CSV manifest file.")
 
    args = parser.parse_args()
 
    # Ensure output directories exist
    os.makedirs(args.output_video_dir, exist_ok=True)
    os.makedirs(args.output_action_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv_path), exist_ok=True)
 
    # Load task instructions (from 'seen' field if JSON)
    print("Loading task instructions...")
    instructions = load_instructions(args.instruction_file)
    print(f"Loaded {len(instructions)} instruction(s) from 'seen' field")
    print(f"Sample instruction: {instructions[0][:80]}...")
 
    # Load joint action data
    print("\nLoading joint_action data from HDF5...")
    joint_action = load_episode_data(args.hdf5_path)
    total_action_frames = len(joint_action)
    print(f"Loaded joint_action with {total_action_frames} timesteps, "
          f"feature dimension: {joint_action.shape[1]}")
 
    # Load video frames
    print("Loading video frames...")
    video_frames = load_video_frames(args.episode_video)
    video_frame_count = len(video_frames)
    print(f"Loaded {video_frame_count} video frames")
 
    # Align action and video lengths by taking the minimum
    if total_action_frames != video_frame_count:
        print(f"Warning: Mismatch between action ({total_action_frames}) "
              f"and video ({video_frame_count}) lengths. Truncating to minimum.")
        total_frames = min(total_action_frames, video_frame_count)
        joint_action = joint_action[:total_frames]
        video_frames = video_frames[:total_frames]
    else:
        total_frames = total_action_frames
 
    # Generate clips using a sliding window
    print(f"\nGenerating clips with window size: {args.num_frames} frames...")
    print(f"Instruction assignment: Random selection from 'seen' list")
 
    metadata_rows = []
    valid_clip_count = 0
    instruction_usage = {inst: 0 for inst in instructions}
 
    for start_idx in range(total_frames):
        end_idx = start_idx + args.num_frames
        if end_idx > total_frames:
            break  # Skip incomplete clips at the end
 
        # Randomly select an instruction from 'seen' list for this clip
        instruction = np.random.choice(instructions)
        instruction_usage[instruction] += 1
 
        # Define output filenames
        clip_name = f"clip_{start_idx:06d}"
        video_path = os.path.join(args.output_video_dir, f"{clip_name}.mp4")
        action_path = os.path.join(args.output_action_dir, f"{clip_name}.npy")
 
        # Save video clip
        clip_frames = video_frames[start_idx:end_idx]
        save_video_clip(clip_frames, video_path)
 
        # Save corresponding action sequence
        action_seq = joint_action[start_idx:end_idx]  # Shape: (num_frames, joint_dim)
        np.save(action_path, action_seq)
 
        # Record metadata
        metadata_rows.append({
            'prompt': instruction,
            'video': video_path,
            'action_seq': action_path
        })
 
        valid_clip_count += 1
        if valid_clip_count % 50 == 0:
            print(f"  Processed {valid_clip_count} clips...")
 
    # Write metadata CSV
    print("\nSaving metadata manifest...")
    df = pd.DataFrame(metadata_rows)
    df.to_csv(args.output_csv_path, index=False)
 
    # Compute instruction statistics
    unique_instructions_used = sum(1 for count in instruction_usage.values() if count > 0)
    unique_in_csv = df['prompt'].nunique()
 
    print("\n" + "="*60)
    print("Conversion completed successfully!")
    print("="*60)
    print(f"Total clips generated: {valid_clip_count}")
    print(f"Frames discarded at end: {total_frames - valid_clip_count}")
    print(f"Available instructions: {len(instructions)} (from 'seen')")
    print(f"Instructions actually used: {unique_instructions_used}")
    print(f"Unique instructions in CSV: {unique_in_csv}")
    print(f"\nOutput locations:")
    print(f"  Video clips: {args.output_video_dir}")
    print(f"  Action sequences: {args.output_action_dir}")
    print(f"  Manifest file: {args.output_csv_path}")
 
    # Show top 5 most frequently used instructions
    print(f"\nTop 5 most used instructions:")
    sorted_usage = sorted(instruction_usage.items(), key=lambda x: x[1], reverse=True)
    for i, (inst, count) in enumerate(sorted_usage[:5], 1):
        if count > 0:
            print(f"  {i}. [{count:3d}x] {inst[:70]}...")
 
 
if __name__ == "__main__":
    main()

"""
python data/scripts/convert_episode.py \
    --episode_video /project/peilab/Puxin/Wan_action/data/robotwin_dataset_raw/adjust_bottle/video/episode0.mp4 \
    --hdf5_path /project/peilab/Puxin/Wan_action/data/robotwin_dataset_raw/adjust_bottle/data/episode0.hdf5 \
    --instruction_file /project/peilab/Puxin/Wan_action/data/robotwin_dataset_raw/adjust_bottle/instructions/episode0.json \
    --num_frames 17 \
    --output_video_dir /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train_episode0/videos \
    --output_action_dir /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train_episode0/actions \
    --output_csv_path /project/peilab/Puxin/Wan_action/data/robotwin_dataset_train_episode0/metadata.csv
"""