#!/usr/bin/env python3
"""
Convert RoboTwin raw data to inference.py-compatible short clips.

功能：
1) 从 raw 数据中按任务采样短片段
2) 每个片段长度固定为 --num_frames
3) 每个任务生成数量由 --samples_per_task 指定
4) 导出 inference.py 可直接使用的 CSV（prompt,input_image,action_seq）

输出目录结构：
  output_dir/
    videos/*.mp4
    actions/*.npy
    inference_input.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import numpy as np
import pandas as pd


@dataclass
class EpisodeRecord:
    task_name: str
    episode_idx: int
    video_path: Path
    hdf5_path: Path
    instruction_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create short fixed-length clips from raw data for inference.py"
    )
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to raw dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--csv_name",
        type=str,
        default="inference_input.csv",
        help="CSV filename under output_dir",
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="aloha-agilex_clean_50",
        help="Subdirectory under each task folder",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=17,
        help="Target clip length (must match inference.py --num_frames)",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=1,
        help="How many clips to generate per task",
    )
    parser.add_argument(
        "--episode_selection",
        type=str,
        choices=["random", "first"],
        default="random",
        help="How to choose episode for each sample",
    )
    parser.add_argument(
        "--instruction_mode",
        type=str,
        choices=["random", "first"],
        default="random",
        help="How to pick prompt from instruction file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS for saved short clip videos",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use_relative_paths",
        action="store_true",
        help="Use paths relative to output_dir in CSV",
    )
    return parser.parse_args()


def load_joint_action(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as file:
        left_arm = file["/joint_action/left_arm"][()]
        left_gripper = file["/joint_action/left_gripper"][()]
        right_arm = file["/joint_action/right_arm"][()]
        right_gripper = file["/joint_action/right_gripper"][()]

    if left_arm.ndim == 1:
        left_arm = left_arm[:, None]
    if left_gripper.ndim == 1:
        left_gripper = left_gripper[:, None]
    if right_arm.ndim == 1:
        right_arm = right_arm[:, None]
    if right_gripper.ndim == 1:
        right_gripper = right_gripper[:, None]

    return np.concatenate([left_arm, left_gripper, right_arm, right_gripper], axis=1).astype(np.float32)


def load_instructions(instruction_path: Path) -> List[str]:
    text = instruction_path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [text] if text else [""]

    if isinstance(data, dict):
        seen = data.get("seen")
        if isinstance(seen, list) and seen:
            return [str(item) for item in seen]
        unseen = data.get("unseen")
        if isinstance(unseen, list) and unseen:
            return [str(item) for item in unseen]
        return [text] if text else [""]

    if isinstance(data, list) and data:
        return [str(item) for item in data]

    return [text] if text else [""]


def pick_prompt(instructions: List[str], mode: str, rng: np.random.Generator) -> str:
    candidates = [item.strip() for item in instructions if isinstance(item, str) and item.strip()]
    if not candidates:
        return ""
    if mode == "first":
        return candidates[0]
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def get_video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def extract_video_window(video_path: Path, start: int, num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    frames: List[np.ndarray] = []
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)

    cap.release()
    return frames


def save_video(frames: List[np.ndarray], output_path: Path, fps: int) -> None:
    if not frames:
        raise ValueError(f"No frames to save for {output_path}")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def maybe_rel(path: Path, output_dir: Path, use_relative_paths: bool) -> str:
    if use_relative_paths:
        return os.path.relpath(path, output_dir)
    return str(path.resolve())


def find_episodes(raw_data_dir: Path, dataset_subdir: str) -> Dict[str, List[EpisodeRecord]]:
    task_to_episodes: Dict[str, List[EpisodeRecord]] = {}

    for task_dir in sorted(raw_data_dir.iterdir()):
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        base_dir = task_dir / dataset_subdir
        video_dir = base_dir / "video"
        data_dir = base_dir / "data"
        instruction_dir = base_dir / "instructions"

        if not (video_dir.exists() and data_dir.exists() and instruction_dir.exists()):
            continue

        records: List[EpisodeRecord] = []
        for video_path in sorted(video_dir.glob("episode*.mp4")):
            episode_name = video_path.stem
            try:
                episode_idx = int(episode_name.replace("episode", ""))
            except ValueError:
                continue

            hdf5_path = data_dir / f"{episode_name}.hdf5"
            instruction_path = instruction_dir / f"{episode_name}.json"
            if hdf5_path.exists() and instruction_path.exists():
                records.append(
                    EpisodeRecord(
                        task_name=task_name,
                        episode_idx=episode_idx,
                        video_path=video_path,
                        hdf5_path=hdf5_path,
                        instruction_path=instruction_path,
                    )
                )

        if records:
            task_to_episodes[task_name] = records

    return task_to_episodes


def main() -> None:
    args = parse_args()

    if args.num_frames < 1:
        raise ValueError("--num_frames must be >= 1")
    if args.samples_per_task < 1:
        raise ValueError("--samples_per_task must be >= 1")

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    video_dir = output_dir / "videos"
    action_dir = output_dir / "actions"
    csv_path = output_dir / args.csv_name

    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"raw_data_dir not found: {raw_data_dir}")

    rng = np.random.default_rng(args.seed)
    task_to_episodes = find_episodes(raw_data_dir, args.dataset_subdir)
    if not task_to_episodes:
        raise RuntimeError("No valid episodes found. Check --raw_data_dir and --dataset_subdir")

    rows = []
    print(f"Found {len(task_to_episodes)} tasks. Generating {args.samples_per_task} sample(s) per task...")

    # Cache episode-level data to avoid repeated disk load
    action_cache: Dict[str, np.ndarray] = {}
    frame_count_cache: Dict[str, int] = {}
    instruction_cache: Dict[str, List[str]] = {}

    for task_name, episodes in sorted(task_to_episodes.items()):
        valid_episodes = []
        for record in episodes:
            episode_key = f"{task_name}:{record.episode_idx}"
            if episode_key not in action_cache:
                action_cache[episode_key] = load_joint_action(record.hdf5_path)
            if episode_key not in frame_count_cache:
                frame_count_cache[episode_key] = get_video_frame_count(record.video_path)

            total_frames = min(action_cache[episode_key].shape[0], frame_count_cache[episode_key])
            if total_frames >= args.num_frames:
                valid_episodes.append(record)

        if not valid_episodes:
            print(f"[Skip task] {task_name}: no episode with >= {args.num_frames} frames")
            continue

        generated = 0
        for sample_idx in range(args.samples_per_task):
            if args.episode_selection == "first":
                selected = valid_episodes[sample_idx % len(valid_episodes)]
            else:
                selected = valid_episodes[int(rng.integers(0, len(valid_episodes)))]

            episode_key = f"{task_name}:{selected.episode_idx}"
            action_seq_all = action_cache[episode_key]
            total_frames = min(action_seq_all.shape[0], frame_count_cache[episode_key])
            max_start = total_frames - args.num_frames
            start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
            end = start + args.num_frames

            clip_frames = extract_video_window(selected.video_path, start=start, num_frames=args.num_frames)
            if len(clip_frames) != args.num_frames:
                print(
                    f"[Skip sample] {task_name} ep{selected.episode_idx} start={start}: "
                    f"video frames {len(clip_frames)} != {args.num_frames}"
                )
                continue

            clip_action = action_seq_all[start:end]
            if clip_action.shape[0] != args.num_frames:
                print(
                    f"[Skip sample] {task_name} ep{selected.episode_idx} start={start}: "
                    f"action frames {clip_action.shape[0]} != {args.num_frames}"
                )
                continue

            sample_name = f"{task_name}_s{sample_idx:03d}_ep{selected.episode_idx}_st{start:04d}"
            clip_video_path = video_dir / f"{sample_name}.mp4"
            clip_action_path = action_dir / f"{sample_name}.npy"

            save_video(clip_frames, clip_video_path, fps=args.fps)
            np.save(clip_action_path, clip_action.astype(np.float32))

            if episode_key not in instruction_cache:
                instruction_cache[episode_key] = load_instructions(selected.instruction_path)
            prompt = pick_prompt(instruction_cache[episode_key], args.instruction_mode, rng)

            rows.append(
                {
                    "prompt": prompt,
                    "input_image": maybe_rel(clip_video_path, output_dir, args.use_relative_paths),
                    "action_seq": maybe_rel(clip_action_path, output_dir, args.use_relative_paths),
                    "task": task_name,
                    "episode": selected.episode_idx,
                    "start": start,
                }
            )
            generated += 1

        print(f"[{task_name}] generated {generated}/{args.samples_per_task}")

    if not rows:
        raise RuntimeError("No samples were generated. Please check input data and arguments.")

    df = pd.DataFrame(rows, columns=["prompt", "input_image", "action_seq", "task", "episode", "start"])
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Total rows: {len(df)}")
    print(f"CSV path: {csv_path}")
    print(f"Videos dir: {video_dir}")
    print(f"Actions dir: {action_dir}")


if __name__ == "__main__":
    main()

"""
python /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/scripts/convert_raw_to_inference_csv.py \
    --raw_data_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/robotwin_expert50_raw \
    --output_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/short_video_infer \
    --num_frames 9 \
    --samples_per_task 5 \
    --episode_selection random \
    --instruction_mode random \
    --seed 42
"""