#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VideoMAE success/failure classifier — HDF5 version (DDP, step-based training/eval)

Data layout expected:
    DATA_ROOT/
        <task_name>/
            episode_0.hdf5
            episode_1.hdf5
            ...
Each HDF5 episode contains:
    observation/images/cam_high        (T, 3, H, W) uint8
    observation/images/cam_left_wrist  (T, 3, H, W) uint8
    observation/images/cam_right_wrist (T, 3, H, W) uint8
    attrs: result (bool), episode_length (int)

The three camera images are concatenated vertically (height axis) per frame.

Launch (single node 8 GPUs):
    torchrun --standalone --nproc_per_node=8 videomae_hdf5.py
"""

import os, glob, random
import logging
from datetime import datetime
from typing import List, Dict, Any
from collections import OrderedDict

import numpy as np
from PIL import Image
import h5py

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
)

try:
    from transformers import VideoMAEImageProcessor as VideoMAEFeatureExtractor
except Exception:
    from transformers import VideoMAEFeatureExtractor

# =========================
# CONFIG
# =========================
CFG = dict(
    DATA_ROOT="/project/peilab/licheng/datasets/5_task_rollout",
    VAL_RATIO=0.15,              # fraction of episodes held out per task for validation
    IMG_SIZE=224,
    WINDOW=8,
    STRIDE_TRAIN=8,
    STRIDE_VAL=200,             # 1 时验证集负样本极多(≈1:464)；50 约 1:9，100 约 1:4
    BATCH_SIZE=4,                # per-GPU
    VAL_BATCH_SIZE=64,           # per-GPU
    NUM_WORKERS=4,
    PERSISTENT_WORKERS=True,
    PREFETCH_FACTOR=2,
    LR=1e-4,
    WEIGHT_DECAY=1e-4,
    # —— 训练完全用 step 控制，不再依赖 epoch —— #
    MAX_STEPS=200_000,           # 你可按需改
    EVAL_STEPS=1000,             # 训练多少步验证一次 & 保存一次
    CKPT_DIR="ckpts_videomae_hdf5",
    LOG_DIR="logs_videomae_hdf5",       # 本地 log 文件目录
    TENSORBOARD_DIR=None,              # None 则用 LOG_DIR/tensorboard
    SEED=42,
    MODEL_NAME="/project/peilab/licheng/models/videomae-base",
    NUM_LABELS=2,
    THRESH_MIN=0.3,
    THRESH_MAX=1.0,
    THRESH_STEPS=20,
    USE_RESAMPLE_TRAIN=True,     # 训练使用无限数据流（epoch 循环重建迭代器）
    SHUFFLE_BUF=500,             # 训练 DataLoader 内部 shuffle buffer 大小
    DROP_LAST=True,              # 训练时丢弃最后一小批，batch 尺寸稳定
)

# =========================
# Helpers
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dist_env():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return world_size, rank, local_rank, device

def collate_fn(batch):
    vids = torch.stack([b[0] for b in batch])             # (B, C, T, H, W)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    meta_keys = batch[0][2].keys()
    meta: Dict[str, Any] = {k: [b[2][k] for b in batch] for k in meta_keys}
    return vids, ys, meta

def scan_hdf5_episodes(data_root: str) -> List[str]:
    """Return sorted list of all episode HDF5 paths under data_root."""
    paths = sorted(glob.glob(os.path.join(data_root, "*", "episode_*.hdf5")))
    return paths

def setup_logging(log_dir: str, rank: int) -> str:
    """
    仅在 rank 0 设置：同时输出到控制台和 log 文件。
    返回当前 log 文件路径；rank != 0 返回空字符串。
    """
    if rank != 0:
        return ""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger = logging.getLogger("videomae_hdf5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return log_file

def log_info(msg: str, rank: int):
    if rank == 0:
        logging.getLogger("videomae_hdf5").info(msg)

def train_val_split(paths: List[str], val_ratio: float, seed: int) -> (List[str], List[str]):
    """
    Split by task so each task contributes proportionally to val.
    Episodes within each task are shuffled before splitting.
    """
    from collections import defaultdict
    by_task: Dict[str, List[str]] = defaultdict(list)
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

# =========================
# Dataset
# =========================
class HDF5EpisodeWindowDataset(Dataset):
    """
    Map-style dataset over sliding windows extracted from HDF5 episodes.

    Each HDF5 episode stores three camera streams:
        observation/images/cam_high        (T, 3, H, W) uint8
        observation/images/cam_left_wrist  (T, 3, H, W) uint8
        observation/images/cam_right_wrist (T, 3, H, W) uint8

    Per frame, the three cameras are vertically concatenated (height axis):
        stacked frame shape: (3, H*3, W)  →  fed to VideoMAEFeatureExtractor

    Label comes from attrs['result'] (True = success = 1, False = 0).

    Window sampling strategy
    ------------------------
    train mode:
        - positive window: last W frames of the episode
        - negative window: one window uniformly sampled from [0, T-W) if episode
          is a failure, or from [0, T-W) of a success episode (non-terminal region)
    val mode:
        - positive window: last W frames of the episode (label = episode result)
        - all windows with stride S from [W, T) (label = 0, non-terminal)
    """
    CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    def __init__(
        self,
        episode_paths: List[str],
        window: int = 8,
        stride: int = 8,
        img_size: int = 224,
        mode: str = "train",
        shuffle_buf: int = 0,
    ):
        super().__init__()
        assert mode in {"train", "val"}
        self.window = window
        self.stride = stride
        self.mode = mode
        self.shuffle_buf = shuffle_buf
        self.fe = VideoMAEFeatureExtractor(size=img_size)
        self.samples = self._build_index(episode_paths)
        # Apply in-memory shuffle of the index to approximate wds.shuffle(shuffle_buf)
        if mode == "train" and shuffle_buf > 0:
            random.shuffle(self.samples)

    def _build_index(self, paths: List[str]):
        """
        Pre-build a flat list of (path, start, end, label) tuples so that
        __getitem__ is O(1) and DataLoader workers can index directly.
        """
        samples = []
        W, S = self.window, self.stride

        for path in paths:
            with h5py.File(path, "r") as f:
                T = int(f.attrs["episode_length"])
                result = bool(f.attrs["result"])

            if T < W:
                continue

            if self.mode == "train":
                # positive: terminal window
                samples.append((path, T - W, T, int(result)))
                # negative: start=-1 is a sentinel — actual window is sampled
                # dynamically in __getitem__ so each epoch sees different negatives
                if T - S >= W:
                    samples.append((path, -1, T, 0))
            else:
                # val: terminal window with true label
                samples.append((path, T - W, T, int(result)))
                # val: non-terminal windows with stride
                for end in range(T - S, W - 1, -S):
                    samples.append((path, end - W, end, 0))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, start, end, label = self.samples[idx]
        if start == -1:
            # Dynamic negative sampling: end stores T (episode length)
            T, W, S = end, self.window, self.stride
            neg_candidates = list(range(T - S, W - 1, -S)) or list(range(T - 1, W - 1, -1))
            end = random.choice(neg_candidates)
            start = end - W
        clip = self._load_clip(path, start, end)       # (W, 3, H*3, W_img) uint8
        tensor = self._to_tensor(clip)                 # (C, T, H, W) float
        meta = {"path": path, "start": start, "end": end, "label": label}
        return tensor, label, meta

    def _load_clip(self, path: str, start: int, end: int) -> np.ndarray:
        """Load and vertically stack three camera frames for [start, end)."""
        with h5py.File(path, "r") as f:
            cams = [
                f[f"observation/images/{cam}"][start:end]  # (W, 3, H, img_W)
                for cam in self.CAMERAS
            ]
        # Each cam: (W, 3, H, img_W) — transpose to (W, H, img_W, 3) for stacking
        cams_hwc = [c.transpose(0, 2, 3, 1) for c in cams]  # (W, H, img_W, 3)
        # Vertical concatenation along height axis → (W, H*3, img_W, 3)
        stacked = np.concatenate(cams_hwc, axis=1)
        return stacked  # (W, H*3, img_W, 3) uint8

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        """clip: (W, H, img_W, 3) uint8 → (C, T, H, W) float tensor."""
        frames = [Image.fromarray(frame.astype(np.uint8)) for frame in clip]
        return self.fe(frames, return_tensors="pt")["pixel_values"][0]

# =========================
# Evaluation (DDP gather on rank0)
# =========================
@torch.no_grad()
def evaluate_ddp(model: nn.Module, loader: DataLoader, device: torch.device, rank: int, world_size: int):
    model.eval()
    logits_local, trues_local = [], []

    for vids, ys, _ in tqdm(loader, desc="Val", disable=(rank != 0)):
        vids = vids.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        logits = model(pixel_values=vids).logits  # (B,2)
        logits_local.extend(logits.cpu().tolist())
        trues_local.extend(ys.cpu().tolist())

    # gather variable-length lists
    logits_gather, trues_gather = [None] * world_size, [None] * world_size
    dist.all_gather_object(logits_gather, logits_local)
    dist.all_gather_object(trues_gather, trues_local)

    if rank != 0:
        return None

    logits = [x for part in logits_gather for x in part]
    trues = [x for part in trues_gather for x in part]
    logits_t = torch.tensor(logits)  # (N,2)
    probs = torch.softmax(logits_t, dim=-1)[:, 1].numpy()

    thresholds = np.linspace(CFG["THRESH_MIN"], CFG["THRESH_MAX"], CFG["THRESH_STEPS"])
    all_metrics = {}
    best = {"f1": -1.0, "thresh": thresholds[0]}

    for th in thresholds:
        preds = (probs >= th).astype(np.int32).tolist()
        acc = accuracy_score(trues, preds)
        prec = precision_score(trues, preds, zero_division=0)
        rec = recall_score(trues, preds, zero_division=0)
        f1 = f1_score(trues, preds, zero_division=0)
        TP = sum((p == 1 and t == 1) for p, t in zip(preds, trues))
        TN = sum((p == 0 and t == 0) for p, t in zip(preds, trues))
        FP = sum((p == 1 and t == 0) for p, t in zip(preds, trues))
        FN = sum((p == 0 and t == 1) for p, t in zip(preds, trues))

        all_metrics[f"thresh_{th:.2f}"] = OrderedDict(
            acc=acc, precision=prec, recall=rec, f1=f1,
            TP=TP, TN=TN, FP=FP, FN=FN,
            pred_pos=int(sum(preds)), pred_neg=int(len(preds) - sum(preds)),
            true_pos=int(sum(trues)), true_neg=int(len(trues) - sum(trues)),
        )
        if f1 > best["f1"]:
            best["f1"], best["thresh"] = f1, th

    return all_metrics, best

# =========================
# Training
# =========================
def main():
    set_seed(CFG["SEED"])
    torch.backends.cudnn.benchmark = True

    world_size, rank, local_rank, device = get_dist_env()

    # --------- 本地 log + TensorBoard（仅 rank 0）---------
    log_dir = CFG["LOG_DIR"]
    tb_dir = CFG["TENSORBOARD_DIR"] or os.path.join(log_dir, "tensorboard")
    if rank == 0:
        log_file = setup_logging(log_dir, rank)
        os.makedirs(tb_dir, exist_ok=True)
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join(tb_dir, run_name))
        log_info(f"[DDP] world_size={world_size}", rank)
        log_info(f"Log file: {log_file}", rank)
        log_info(f"TensorBoard: {tb_dir} (run: {run_name})", rank)
    else:
        writer = None

    # --------- Scan & split HDF5 episodes ---------
    all_paths = scan_hdf5_episodes(CFG["DATA_ROOT"])
    log_info(f"Total episodes found: {len(all_paths)}", rank)

    if len(all_paths) == 0:
        raise RuntimeError(f"No HDF5 episodes found under {CFG['DATA_ROOT']}")

    train_paths, val_paths = train_val_split(all_paths, CFG["VAL_RATIO"], CFG["SEED"])
    log_info(f"Train episodes: {len(train_paths)}, Val episodes: {len(val_paths)}", rank)

    # --------- Datasets ---------
    tr_ds = HDF5EpisodeWindowDataset(
        episode_paths=train_paths,
        window=CFG["WINDOW"],
        stride=CFG["STRIDE_TRAIN"],
        img_size=CFG["IMG_SIZE"],
        mode="train",
        shuffle_buf=CFG["SHUFFLE_BUF"],
    )
    va_ds = HDF5EpisodeWindowDataset(
        episode_paths=val_paths,
        window=CFG["WINDOW"],
        stride=CFG["STRIDE_VAL"],
        img_size=CFG["IMG_SIZE"],
        mode="val",
        shuffle_buf=0,
    )

    log_info(f"Train windows: {len(tr_ds)}, Val windows: {len(va_ds)}", rank)

    # --------- Distributed samplers ---------
    tr_sampler = torch.utils.data.distributed.DistributedSampler(
        tr_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=CFG["SEED"]
    )
    va_sampler = torch.utils.data.distributed.DistributedSampler(
        va_ds, num_replicas=world_size, rank=rank, shuffle=False
    )

    # --------- DataLoaders ---------
    tr_ld = DataLoader(
        tr_ds,
        batch_size=CFG["BATCH_SIZE"],
        sampler=tr_sampler,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=CFG["PERSISTENT_WORKERS"],
        prefetch_factor=CFG["PREFETCH_FACTOR"],
        drop_last=CFG["DROP_LAST"],
    )
    va_ld = DataLoader(
        va_ds,
        batch_size=CFG["VAL_BATCH_SIZE"],
        sampler=va_sampler,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=CFG["PERSISTENT_WORKERS"],
        prefetch_factor=CFG["PREFETCH_FACTOR"],
        drop_last=False,
    )

    # --------- Model / Optim ---------
    cfg = VideoMAEConfig.from_pretrained(
        CFG["MODEL_NAME"],
        num_frames=CFG["WINDOW"],
        num_labels=CFG["NUM_LABELS"],
    )
    model = VideoMAEForVideoClassification.from_pretrained(CFG["MODEL_NAME"], config=cfg).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])

    # --------- Step-based Train loop ---------
    os.makedirs(CFG["CKPT_DIR"], exist_ok=True) if rank == 0 else None
    global_step, best_f1 = 0, -1.0

    # 用一个"无尽"迭代器；USE_RESAMPLE_TRAIN=True 时循环重建，模拟无限数据流
    tr_iter = iter(tr_ld)

    while global_step < CFG["MAX_STEPS"]:
        try:
            vids, ys, _ = next(tr_iter)
        except StopIteration:
            if CFG["USE_RESAMPLE_TRAIN"]:
                # 重建迭代器，同时重新 shuffle index（DistributedSampler 用新 epoch）
                tr_sampler.set_epoch(global_step)
                tr_iter = iter(tr_ld)
                continue
            else:
                break

        model.train()
        vids = vids.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True)

        logits = model(pixel_values=vids).logits
        loss = criterion(logits, ys)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        global_step += 1
        if rank == 0 and global_step % 20 == 0:
            loss_val = loss.item()
            log_info(f"[step {global_step}] loss={loss_val:.4f}", rank)
            writer.add_scalar("train/loss", loss_val, global_step)

        # ---- step-based eval & checkpoint ----
        if global_step % CFG["EVAL_STEPS"] == 0:
            out = evaluate_ddp(model, va_ld, device, rank, world_size)
            if rank == 0 and out is not None:
                all_metrics, best = out
                log_info(f"[Val @ step {global_step}]", rank)
                for k, v in all_metrics.items():
                    acc = v["acc"]; prec = v["precision"]; rec = v["recall"]; f1 = v["f1"]
                    log_info(f"  {k}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}", rank)
                log_info(f"Best F1={best['f1']:.4f} @ thresh={best['thresh']:.2f}", rank)

                # TensorBoard: 记录 best 阈值下的 val 指标
                writer.add_scalar("val/best_f1", best["f1"], global_step)
                writer.add_scalar("val/best_thresh", best["thresh"], global_step)
                best_metrics = all_metrics.get(f"thresh_{best['thresh']:.2f}", {})
                if best_metrics:
                    writer.add_scalars("val/metrics", {
                        "accuracy": best_metrics["acc"],
                        "precision": best_metrics["precision"],
                        "recall": best_metrics["recall"],
                        "f1": best_metrics["f1"],
                    }, global_step)

                # ---- 保存 step checkpoint ----
                step_pth = os.path.join(
                    CFG["CKPT_DIR"],
                    f"videomae_step{global_step}_f1{best['f1']:.4f}_th{best['thresh']:.2f}.pth"
                )
                torch.save(model.module.state_dict(), step_pth)
                log_info(f"[Checkpoint] saved → {step_pth}", rank)

                # ---- 保存 best checkpoint ----
                if best["f1"] > best_f1:
                    best_f1 = best["f1"]
                    best_thresh = best["thresh"]
                    best_pth = os.path.join(
                        CFG["CKPT_DIR"],
                        f"best_videomae_f1{best_f1:.4f}_th{best_thresh:.2f}.pth"
                    )
                    torch.save(model.module.state_dict(), best_pth)
                    log_info(f"[Best Checkpoint] saved → {best_pth}", rank)
            dist.barrier()

    if rank == 0:
        log_info(f"[Done] total steps={global_step}", rank)
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# torchrun --standalone --nproc_per_node=8 videomae_hdf5.py
