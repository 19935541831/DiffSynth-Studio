"""
Dataset for ODE regression: loads trajectory .pt files (ode_latent, prompt, action_seq).
"""

import os
from typing import Optional

import torch
from torch.utils.data import Dataset


class ODETrajectoryDataset(Dataset):
    """
    Each item is a .pt file with keys: ode_latent (T+1, F, C, H, W), prompt (str), action_seq (T_act, D).
    """

    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = data_dir
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".pt")],
            key=lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else 0,
        )
        if max_samples is not None:
            self.files = self.files[:max_samples]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(path, map_location="cpu", weights_only=True)
        return {
            "ode_latent": data["ode_latent"],
            "prompt": data["prompt"],
            "action_seq": data["action_seq"],
        }
