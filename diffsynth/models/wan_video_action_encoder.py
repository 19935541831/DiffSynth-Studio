# models/wan_video_action_encoder.py
import torch
import torch.nn as nn

class WanActionEncoder(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        dit_dim: int,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.dit_dim = dit_dim

        self.mlp = nn.Sequential(
            nn.Linear(joint_dim, dit_dim),
            nn.GELU(),
            nn.Linear(dit_dim, dit_dim)
        )

    def forward(self, action_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_seq: (B, T, joint_dim)
        Returns:
            action_emb: (B, T, dit_dim) 
        """
        return self.mlp(action_seq)  # (B, T, D)