from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Content-dependent spatial attention pooling for CRANE-v0.

    Learns a spatial weighting over the 8x8 grid, replacing standard GAP.
    """

    def __init__(self, channels: int = 192) -> None:
        super().__init__()
        self.w_p = nn.Parameter(torch.Tensor(channels, 1))
        nn.init.normal_(self.w_p, std=0.02)

    def forward(self, h_attn: torch.Tensor) -> torch.Tensor:
        # h_attn: (B, C, H, W)
        batch, channels, h, w = h_attn.shape
        # H_flat: (B, N, C) where N = H * W
        H_flat = h_attn.reshape(batch, channels, h * w).transpose(1, 2)
        
        # scores: (B, N, 1)
        scores = torch.matmul(H_flat, self.w_p)
        weights = torch.softmax(scores, dim=1)
        
        # h_pool: (B, C)
        h_pool = torch.sum(weights * H_flat, dim=1)
        return h_pool


class ValueHead(nn.Module):
    """Value head for CRANE-v0 using Attention Pooling.

    Data flow:
        h_attn (B, C, 8, 8)
        -> AttentionPooling                  -> (B, C)
        -> Linear(C, 64) + SiLU              -> (B, 64)
        -> Linear(64, 1) + Tanh              -> (B, 1)
    """

    def __init__(self, trunk_channels: int = 192, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pool = AttentionPooling(channels=trunk_channels)
        self.fc1 = nn.Linear(trunk_channels, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h_attn: torch.Tensor) -> torch.Tensor:
        h_pool = self.pool(h_attn)
        z = self.fc2(self.act(self.fc1(h_pool)))
        return torch.tanh(z)


__all__ = ["ValueHead", "AttentionPooling"]
