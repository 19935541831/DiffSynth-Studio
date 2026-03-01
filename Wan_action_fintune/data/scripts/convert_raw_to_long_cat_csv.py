#!/usr/bin/env python3
"""
Convert RoboTwin raw data to long_cat_infer.py CSV format.

目标：从 raw 数据中“每个任务选择一个 episode”，生成 long_cat_infer.py 需要的 CSV。

输入 raw 目录结构（示例）:
  raw_data_dir/
    task_name/
      aloha-agilex_clean_50/
        video/episode0.mp4
        data/episode0.hdf5
        instructions/episode0.json

输出:
  output_dir/
    actions/
      taskA_episode0.npy
      taskB_episode7.npy
    long_cat_input.csv

CSV 列:
  - prompt
  - input_image   (可直接填写 episode 的 mp4，long_cat_infer.py 会自动取首帧)
  - action_seq    (.npy 路径)
  - output_name   (可选，这里默认生成 task_episode.mp4)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


EpisodeRecord = Tuple[str, int, Path, Path, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select one episode per task from raw data and export long_cat inference CSV."
    )
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Path to raw dataset root (contains multiple task folders).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for converted actions and CSV.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="long_cat_input.csv",
        help="Output CSV filename under output_dir.",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["first", "random"],
        default="first",
        help="Episode selection strategy per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used when selection_mode=random).",
    )
    parser.add_argument(
        "--instruction_mode",
        type=str,
        choices=["first", "random"],
        default="random",
        help="How to pick prompt from instruction list for selected episode.",
    )
    parser.add_argument(
        "--dataset_subdir",
        type=str,
        default="aloha-agilex_clean_50",
        help="Subdirectory name under each task folder.",
    )
    parser.add_argument(
        "--use_relative_paths",
        action="store_true",
        help="Write input_image/action_seq as paths relative to output_dir.",
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
        if isinstance(seen, list) and len(seen) > 0:
            return [str(item) for item in seen]
        unseen = data.get("unseen")
        if isinstance(unseen, list) and len(unseen) > 0:
            return [str(item) for item in unseen]
        return [text]

    if isinstance(data, list) and len(data) > 0:
        return [str(item) for item in data]

    return [text]


def find_task_episodes(raw_data_dir: Path, dataset_subdir: str) -> Dict[str, List[EpisodeRecord]]:
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

        episodes: List[EpisodeRecord] = []
        for video_path in sorted(video_dir.glob("episode*.mp4")):
            episode_name = video_path.stem
            try:
                episode_idx = int(episode_name.replace("episode", ""))
            except ValueError:
                continue

            hdf5_path = data_dir / f"{episode_name}.hdf5"
            instruction_path = instruction_dir / f"{episode_name}.json"
            if hdf5_path.exists() and instruction_path.exists():
                episodes.append((task_name, episode_idx, video_path, hdf5_path, instruction_path))

        if episodes:
            task_to_episodes[task_name] = episodes

    return task_to_episodes


def select_episode(
    episodes: List[EpisodeRecord], selection_mode: str, rng: np.random.Generator
) -> EpisodeRecord:
    if selection_mode == "first":
        return episodes[0]
    index = int(rng.integers(0, len(episodes)))
    return episodes[index]


def pick_prompt(instructions: List[str], instruction_mode: str, rng: np.random.Generator) -> str:
    valid_instructions = [item.strip() for item in instructions if isinstance(item, str) and item.strip()]
    if not valid_instructions:
        return ""
    if instruction_mode == "first":
        return valid_instructions[0]
    index = int(rng.integers(0, len(valid_instructions)))
    return valid_instructions[index]


def maybe_rel(path: Path, output_dir: Path, use_relative_paths: bool) -> str:
    if use_relative_paths:
        return os.path.relpath(path, output_dir)
    return str(path.resolve())


def main() -> None:
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    action_dir = output_dir / "actions"
    csv_path = output_dir / args.csv_name

    output_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"raw_data_dir not found: {raw_data_dir}")

    rng = np.random.default_rng(args.seed)
    task_to_episodes = find_task_episodes(raw_data_dir, args.dataset_subdir)

    if not task_to_episodes:
        raise RuntimeError("No valid episodes found. Check raw_data_dir and dataset_subdir.")

    rows: List[Dict[str, str]] = []
    print(f"Found {len(task_to_episodes)} tasks. Selecting one episode per task...")

    for task_name, episodes in sorted(task_to_episodes.items()):
        selected = select_episode(episodes, args.selection_mode, rng)
        _, episode_idx, video_path, hdf5_path, instruction_path = selected

        action_seq = load_joint_action(hdf5_path)
        action_output_path = action_dir / f"{task_name}_episode{episode_idx}.npy"
        np.save(action_output_path, action_seq)

        instructions = load_instructions(instruction_path)
        prompt = pick_prompt(instructions, args.instruction_mode, rng)

        rows.append(
            {
                "prompt": prompt,
                "input_image": maybe_rel(video_path, output_dir, args.use_relative_paths),
                "action_seq": maybe_rel(action_output_path, output_dir, args.use_relative_paths),
                "output_name": f"{task_name}_episode{episode_idx}.mp4",
            }
        )

        print(
            f"[{task_name}] episode{episode_idx} -> action: {action_output_path.name}, "
            f"frames={action_seq.shape[0]}, dim={action_seq.shape[1]}"
        )

    df = pd.DataFrame(rows, columns=["prompt", "input_image", "action_seq", "output_name"])
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print("\nDone.")
    print(f"Tasks selected: {len(df)}")
    print(f"Actions dir: {action_dir}")
    print(f"CSV path: {csv_path}")


if __name__ == "__main__":
    main()

"""
python /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/scripts/convert_raw_to_long_cat_csv.py \
    --raw_data_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/data/robotwin_expert50_raw \
    --output_dir /opt/tiger/workspace/DiffSynth-Studio/Wan_action_fintune/inference/long_video_infer \
    --selection_mode random \
    --instruction_mode random \
    --seed 42
"""