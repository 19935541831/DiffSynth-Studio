#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HDF5 数据集正负样本比例检查脚本

统计内容：
  1. Episode 级别：按 attrs['result'] 统计成功(正)/失败(负) episode 数量与比例
  2. 训练/验证 样本级别：按 videomae_hdf5 的 _build_index 逻辑统计实际参与训练的 window 样本中正负比例

数据格式与 videomae_hdf5.py 一致：
  DATA_ROOT/<task_name>/episode_*.hdf5
  每个 HDF5：attrs['result'] (bool), attrs['episode_length'] (int)
  result=True → 正样本(success), False → 负样本(failure)

用法：
  python check_hdf5_class_balance.py
  python check_hdf5_class_balance.py --data_root /path/to/your/hdf5/root
"""

import os
import glob
import argparse
from collections import defaultdict

import h5py


def scan_hdf5_episodes(data_root: str):
    """与 videomae_hdf5 一致：返回 data_root 下所有 episode HDF5 路径（已排序）。"""
    paths = sorted(glob.glob(os.path.join(data_root, "*", "episode_*.hdf5")))
    return paths


def train_val_split(paths, val_ratio: float, seed: int):
    """与 videomae_hdf5 一致：按 task 划分 train/val。"""
    import random
    by_task = defaultdict(list)
    for p in paths:
        task = os.path.basename(os.path.dirname(p))
        by_task[task].append(p)
    train_paths, val_paths = [], []
    rng = random.Random(seed)
    for task_eps in by_task.values():
        shuffled = task_eps[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_paths.extend(shuffled[:n_val])
        train_paths.extend(shuffled[n_val:])
    return train_paths, val_paths


def count_episode_labels(paths, window: int = 8, stride: int = 8):
    """
    遍历 HDF5 列表，统计 episode 级别正负数量。
    同时返回每个 episode 的 (path, T, result) 用于后续 sample 级统计。
    """
    episodes = []  # (path, T, result)
    for path in paths:
        with h5py.File(path, "r") as f:
            T = int(f.attrs["episode_length"])
            result = bool(f.attrs["result"])
        if T < window:
            continue
        episodes.append((path, T, result))
    return episodes


def count_train_samples(episodes, window: int = 8, stride: int = 8):
    """
    与 HDF5EpisodeWindowDataset(mode='train')._build_index 逻辑一致：
    - 每个 episode：1 个 terminal window，label = result
    - 每个 episode（若 T - stride >= window）：1 个 negative 槽位（label=0，动态采样）
    因此每个 episode 贡献：1 个正或负（terminal）+ 1 个负（若长度够）。
    """
    n_pos, n_neg = 0, 0
    for path, T, result in episodes:
        # terminal window
        if result:
            n_pos += 1
        else:
            n_neg += 1
        # negative slot (non-terminal)
        if T - stride >= window:
            n_neg += 1
    return n_pos, n_neg


def count_val_samples(episodes, window: int = 8, stride: int = 1):
    """
    与 HDF5EpisodeWindowDataset(mode='val')._build_index 逻辑一致：
    - 每个 episode：1 个 terminal window，label = result
    - 每个 episode：非 terminal 的 windows，stride 滑动，全部 label=0
    """
    n_pos, n_neg = 0, 0
    for path, T, result in episodes:
        # terminal window
        if result:
            n_pos += 1
        else:
            n_neg += 1
        # non-terminal windows
        for end in range(T - stride, window - 1, -stride):
            n_neg += 1
    return n_pos, n_neg


def main():
    parser = argparse.ArgumentParser(description="HDF5 数据集正负样本比例检查")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/project/peilab/licheng/datasets/5_task_rollout",
        help="HDF5 根目录，与 videomae_hdf5 的 DATA_ROOT 一致",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例，与训练脚本一致")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    parser.add_argument("--window", type=int, default=8, help="窗口长度")
    parser.add_argument("--stride_train", type=int, default=8, help="训练时 stride")
    parser.add_argument("--stride_val", type=int, default=1, help="验证时 stride")
    args = parser.parse_args()

    data_root = args.data_root
    if not os.path.isdir(data_root):
        print(f"[Error] data_root 不存在: {data_root}")
        return

    all_paths = scan_hdf5_episodes(data_root)
    print(f"数据根目录: {data_root}")
    print(f"总 episode 文件数: {len(all_paths)}")
    if len(all_paths) == 0:
        print("未找到任何 episode_*.hdf5，请检查路径。")
        return

    # ---------- Episode 级别统计（全量） ----------
    all_episodes = count_episode_labels(all_paths, window=args.window)
    n_ep_success = sum(1 for _, _, r in all_episodes if r)
    n_ep_failure = sum(1 for _, _, r in all_episodes if not r)
    n_ep_skip = len(all_paths) - len(all_episodes)
    if n_ep_skip:
        print(f"跳过长度 < window 的 episode 数: {n_ep_skip}")

    print("\n" + "=" * 60)
    print("【Episode 级别】按 attrs['result'] 统计")
    print("=" * 60)
    print(f"  正样本 (success, result=True):  {n_ep_success}  ({100 * n_ep_success / len(all_episodes):.2f}%)")
    print(f"  负样本 (failure, result=False): {n_ep_failure}  ({100 * n_ep_failure / len(all_episodes):.2f}%)")
    print(f"  正:负 比例 ≈ 1:{n_ep_failure / max(1, n_ep_success):.2f}")
    if n_ep_failure > n_ep_success * 3:
        print("  >>> 提示：负样本 episode 明显多于正样本，可能导致验证集上 precision 低、recall 一般、F1 偏低。")

    # 按 task 统计
    by_task = defaultdict(lambda: {"pos": 0, "neg": 0})
    for path, T, result in all_episodes:
        task = os.path.basename(os.path.dirname(path))
        if result:
            by_task[task]["pos"] += 1
        else:
            by_task[task]["neg"] += 1
    print("\n  按 task 统计 (Episode 级别):")
    for task in sorted(by_task.keys()):
        p, n = by_task[task]["pos"], by_task[task]["neg"]
        total = p + n
        print(f"    {task}: 正={p}, 负={n}, 正比例={100*p/total:.1f}%")

    # ---------- Train/Val 划分后样本级别统计 ----------
    train_paths, val_paths = train_val_split(all_paths, args.val_ratio, args.seed)
    train_episodes = count_episode_labels(train_paths, window=args.window)
    val_episodes = count_episode_labels(val_paths, window=args.window)

    tr_pos, tr_neg = count_train_samples(train_episodes, window=args.window, stride=args.stride_train)
    va_pos, va_neg = count_val_samples(val_episodes, window=args.window, stride=args.stride_val)

    print("\n" + "=" * 60)
    print("【训练集】Window 样本级别（与 DataLoader 一致）")
    print("=" * 60)
    tr_total = tr_pos + tr_neg
    print(f"  正样本窗口数: {tr_pos}  ({100 * tr_pos / max(1, tr_total):.2f}%)")
    print(f"  负样本窗口数: {tr_neg}  ({100 * tr_neg / max(1, tr_total):.2f}%)")
    print(f"  正:负 ≈ 1:{tr_neg / max(1, tr_pos):.2f}")

    print("\n" + "=" * 60)
    print("【验证集】Window 样本级别（与 DataLoader 一致）")
    print("=" * 60)
    va_total = va_pos + va_neg
    print(f"  正样本窗口数: {va_pos}  ({100 * va_pos / max(1, va_total):.2f}%)")
    print(f"  负样本窗口数: {va_neg}  ({100 * va_neg / max(1, va_total):.2f}%)")
    print(f"  正:负 ≈ 1:{va_neg / max(1, va_pos):.2f}")
    if va_neg > va_pos * 5:
        print("  >>> 提示：验证集负样本远多于正样本（非 terminal 窗口均为负），易出现高 accuracy、低 precision。")

    # ---------- 不同 stride_val 下的验证集比例，用于选较均衡的 stride ----------
    print("\n" + "=" * 60)
    print("【验证集】不同 stride_val 下的正负比例（供调参）")
    print("=" * 60)
    stride_candidates = [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256]
    # 只保留不超过「单 episode 最大长度」的 stride（粗略用 500）
    stride_candidates = [s for s in stride_candidates if s >= 1]
    print(f"  {'stride_val':>10}  {'正样本':>8}  {'负样本':>8}  正:负")
    best_for_5 = None
    best_for_10 = None
    for s in stride_candidates:
        vp, vn = count_val_samples(val_episodes, window=args.window, stride=s)
        ratio = vn / max(1, vp)
        print(f"  {s:>10}  {vp:>8}  {vn:>8}  1:{ratio:.1f}")
        if best_for_5 is None or abs(ratio - 5) < abs(best_for_5[1] - 5):
            best_for_5 = (s, ratio)
        if best_for_10 is None or abs(ratio - 10) < abs(best_for_10[1] - 10):
            best_for_10 = (s, ratio)
    print()
    print("  建议：若希望验证集正:负约 1:5，可将 videomae_hdf5 中 STRIDE_VAL 设为", best_for_5[0], f"（当前约 1:{best_for_5[1]:.1f}）")
    print("        若希望约 1:10，可设 STRIDE_VAL =", best_for_10[0], f"（当前约 1:{best_for_10[1]:.1f}）")

    print("\n" + "=" * 60)
    print("结论与建议")
    print("=" * 60)
    if n_ep_failure > n_ep_success * 2 or va_neg > va_pos * 5:
        print("  当前数据存在明显类别不平衡（负样本偏多）。")
        print("  可考虑：对正样本过采样、调高正类权重、或收集更多成功 trajectory 以平衡比例。")
    else:
        print("  正负比例相对均衡；若仍出现低 precision，可检查阈值或模型校准。")
    print()


if __name__ == "__main__":
    main()
