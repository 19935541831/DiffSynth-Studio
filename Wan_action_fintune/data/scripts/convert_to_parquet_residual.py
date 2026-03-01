#!/usr/bin/env python3
"""
Convert Robot Dataset to Parquet Format (Residual Action + Z-score)

This script converts robot trajectory data (video + action + instruction)
into sharded Parquet files with the same row schema as convert_to_parquet.py,
but action representation is:

1) Residual action: delta_t = action_t - action_{t-1}, delta_0 = 0
2) Z-score normalization: (delta - mean) / std
3) 3-sigma clipping: clip(z, -sigma_clip, sigma_clip)

Input formats:
    1. RoboTwin raw format (video + HDF5 + JSON instructions)
    2. Processed CSV format (video paths + npy action paths)

Output Parquet schema is unchanged:
    - episode_id, task_name, instruction, total_frames
    - frame_idx, frame_data (JPEG bytes), action (float list), timestamp
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

from diffsynth.core.data.parquet_utils import ParquetTrajectoryWriter


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


def load_joint_action_hdf5(hdf5_path: str) -> np.ndarray:
    """
    Load and concatenate joint action data from HDF5 file (RoboTwin format).
    """
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/joint_action/left_arm"][()]
        left_gripper = f["/joint_action/left_gripper"][()]
        right_arm = f["/joint_action/right_arm"][()]
        right_gripper = f["/joint_action/right_gripper"][()]

    if left_arm.ndim == 1:
        left_arm = left_arm[:, None]
    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_arm.ndim == 1:
        right_arm = right_arm[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]

    joint_action = np.concatenate(
        [left_arm, left_gripper, right_arm, right_gripper],
        axis=1,
    )
    return joint_action.astype(np.float32)


def load_instructions_json(instruction_file: str) -> List[str]:
    """
    Load task instructions from JSON file.
    """
    with open(instruction_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    try:
        data = json.loads(content)
        if isinstance(data, dict) and "seen" in data:
            return data["seen"]
        if isinstance(data, list):
            return data
        return [content]
    except json.JSONDecodeError:
        return [content]


def extract_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Extract all frames from a video file in RGB format.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
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
                episodes.append(
                    (
                        task_name,
                        episode_idx,
                        str(video_file),
                        str(hdf5_file),
                        str(instr_file),
                    )
                )
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
    sigma_clip: float = 3.0,
    stats_filename: str = "action_residual_zscore.json",
):
    """
    Convert RoboTwin raw dataset to Parquet with residual z-score actions.
    """
    np.random.seed(seed)

    print("=" * 70)
    print("RoboTwin to Parquet Converter (Residual + Z-score)")
    print("=" * 70)
    print(f"Input: {raw_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Shard size: {shard_size} rows")
    print(f"JPEG quality: {jpeg_quality}")
    print(f"Sigma clip: +/-{sigma_clip}")
    print()

    print("Scanning for episodes...")
    episodes = find_robotwin_episodes(raw_data_dir)
    print(f"Found {len(episodes)} episodes")
    print()

    print("Computing residual action z-score stats...")
    residual_arrays = []
    for _, _, _, hdf5_path, _ in tqdm(episodes, desc="Scanning actions"):
        try:
            actions = load_joint_action_hdf5(hdf5_path)
            residual_arrays.append(compute_residual_actions(actions))
        except Exception as e:
            print(f"Warning: failed to load actions from {hdf5_path}: {e}")

    stats = compute_zscore_stats_from_residual_arrays(residual_arrays)
    save_normalization_params(
        output_dir=output_dir,
        stats_filename=stats_filename,
        action_dim=len(stats["mean"]),
        mean=stats["mean"],
        std=stats["std"],
        sigma_clip=sigma_clip,
    )
    print()

    with ParquetTrajectoryWriter(
        output_dir=output_dir,
        shard_size=shard_size,
        jpeg_quality=jpeg_quality,
    ) as writer:
        for task_name, episode_idx, video_path, hdf5_path, instr_path in tqdm(
            episodes,
            desc="Converting",
        ):
            episode_id = f"{task_name}_episode{episode_idx}"

            try:
                frames = extract_video_frames(video_path)
                actions = load_joint_action_hdf5(hdf5_path)
                residual_actions = compute_residual_actions(actions)
                instructions = load_instructions_json(instr_path)

                num_frames = min(len(frames), len(residual_actions))
                frames = frames[:num_frames]
                residual_actions = residual_actions[:num_frames]

                if instruction_mode == "random":
                    instruction = np.random.choice(instructions)
                elif instruction_mode == "first":
                    instruction = instructions[0]
                else:
                    instruction = instructions[0]

                for frame_idx, (frame, residual_action) in enumerate(
                    zip(frames, residual_actions)
                ):
                    action_norm = normalize_residual_action_zscore(
                        residual_action=residual_action,
                        mean=stats["mean"],
                        std=stats["std"],
                        sigma_clip=sigma_clip,
                    )
                    writer.write_frame(
                        episode_id=episode_id,
                        task_name=task_name,
                        instruction=instruction,
                        total_frames=num_frames,
                        frame_idx=frame_idx,
                        frame=frame,
                        action=action_norm,
                        timestamp=frame_idx / 30.0,
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
    sigma_clip: float = 3.0,
    stats_filename: str = "action_residual_zscore.json",
):
    """
    Convert processed CSV format dataset to Parquet with residual z-score actions.
    """
    print("=" * 70)
    print("CSV to Parquet Converter (Residual + Z-score)")
    print("=" * 70)
    print(f"CSV: {csv_path}")
    print(f"Base path: {base_path}")
    print(f"Output: {output_dir}")
    print(f"Sigma clip: +/-{sigma_clip}")
    print()

    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} entries in CSV")
    print()

    print("Computing residual action z-score stats...")
    residual_arrays = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning actions"):
        action_path = row[action_col]
        if not os.path.isabs(action_path):
            action_path = os.path.join(base_path, action_path)
        try:
            actions = np.load(action_path).astype(np.float32)
            residual_arrays.append(compute_residual_actions(actions))
        except Exception as e:
            print(f"Warning: failed to load actions from {action_path}: {e}")

    stats = compute_zscore_stats_from_residual_arrays(residual_arrays)
    save_normalization_params(
        output_dir=output_dir,
        stats_filename=stats_filename,
        action_dim=len(stats["mean"]),
        mean=stats["mean"],
        std=stats["std"],
        sigma_clip=sigma_clip,
    )
    print()

    with ParquetTrajectoryWriter(
        output_dir=output_dir,
        shard_size=shard_size,
        jpeg_quality=jpeg_quality,
    ) as writer:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            video_path = row[video_col]
            action_path = row[action_col]

            if not os.path.isabs(video_path):
                video_path = os.path.join(base_path, video_path)
            if not os.path.isabs(action_path):
                action_path = os.path.join(base_path, action_path)

            instruction = row[prompt_col]
            task_name = row.get(task_col, "unknown") if task_col and task_col in row else "unknown"
            episode_idx = row.get(episode_col, idx) if episode_col and episode_col in row else idx
            episode_id = f"{task_name}_episode{episode_idx}"

            try:
                frames = extract_video_frames(video_path)
                actions = np.load(action_path).astype(np.float32)
                residual_actions = compute_residual_actions(actions)

                num_frames = min(len(frames), len(residual_actions))
                frames = frames[:num_frames]
                residual_actions = residual_actions[:num_frames]

                for frame_idx, (frame, residual_action) in enumerate(
                    zip(frames, residual_actions)
                ):
                    action_norm = normalize_residual_action_zscore(
                        residual_action=residual_action,
                        mean=stats["mean"],
                        std=stats["std"],
                        sigma_clip=sigma_clip,
                    )
                    writer.write_frame(
                        episode_id=episode_id,
                        task_name=task_name,
                        instruction=instruction,
                        total_frames=num_frames,
                        frame_idx=frame_idx,
                        frame=frame,
                        action=action_norm,
                        timestamp=frame_idx / 30.0,
                    )

            except Exception as e:
                print(f"Error processing {episode_id}: {e}")
                continue

    print()
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert robot trajectory data to Parquet (residual + z-score actions)"
    )

    parser.add_argument(
        "--mode",
        choices=["robotwin", "csv"],
        required=True,
        help="Input data format mode",
    )

    parser.add_argument(
        "--raw_data_dir",
        type=str,
        help="Path to RoboTwin raw data directory (for robotwin mode)",
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to metadata CSV file (for csv mode)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="",
        help="Base path for resolving relative paths in CSV",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for Parquet shards",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10000,
        help="Number of rows (frames) per Parquet shard (default: 10000)",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=100,
        help="JPEG compression quality 1-100 (default: 100)",
    )
    parser.add_argument(
        "--instruction_mode",
        choices=["random", "first", "all"],
        default="random",
        help="How to select instruction for each episode (default: random)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--sigma_clip",
        type=float,
        default=3.0,
        help="Clip z-score normalized action into [-sigma_clip, sigma_clip] (default: 3.0)",
    )

    parser.add_argument("--video_col", type=str, default="video", help="CSV column for video paths")
    parser.add_argument("--action_col", type=str, default="action_seq", help="CSV column for action paths")
    parser.add_argument("--prompt_col", type=str, default="prompt", help="CSV column for instructions")
    parser.add_argument("--task_col", type=str, default="task", help="CSV column for task name")
    parser.add_argument("--episode_col", type=str, default="episode", help="CSV column for episode index")
    parser.add_argument(
        "--norm_stats_filename",
        type=str,
        default="action_residual_zscore.json",
        help="Filename for saving residual z-score params in output_dir",
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
            sigma_clip=args.sigma_clip,
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
            sigma_clip=args.sigma_clip,
            stats_filename=args.norm_stats_filename,
        )


if __name__ == "__main__":
    main()
