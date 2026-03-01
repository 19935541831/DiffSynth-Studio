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
        self.packed_joint_dim = joint_dim * 4
        self.dit_dim = dit_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.packed_joint_dim, self.dit_dim),
            nn.SiLU(),
            nn.Linear(dit_dim, dit_dim)
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-4)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, action_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_seq: (B, T, 4*joint_dim)
        Returns:
            action_emb: (B, T, dit_dim) 
        """
        return self.mlp(action_seq)  # (B, T, D)