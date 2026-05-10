from __future__ import annotations

import torch
import torch.nn as nn

from .fusion import RayFusion
from .grid import GridStream
from .ray import DirectedRayStream
from .relation import SerialAttentionStage


class CRANETorso(nn.Module):
    """Serial conv→attention torso for CRANE-v0.

    Data flow:
        X_spatial → Stem → FiLM(s_scalar) → h_0
        
        h_0       → GridStream(12 blocks) ──┐
        X_spatial → DirectedRayStream     ──┤── RayFusion → SerialAttention → f_trunk
    """

    def __init__(
        self,
        input_channels: int = 18,
        width: int = 192,
        board_size: int = 8,
        grid_blocks: int = 12,
        relation_blocks: int = 5,
    ) -> None:
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(),
        )
        
        # FiLM Generator for 5 scalar variables
        self.film_gen = nn.Linear(5, width * 2)
        # Initialize FiLM to identity mapping
        nn.init.constant_(self.film_gen.weight, 0.0)
        nn.init.constant_(self.film_gen.bias[:width], 1.0) # gamma = 1
        nn.init.constant_(self.film_gen.bias[width:], 0.0) # beta = 0

        # Branches
        self.grid = GridStream(width=width, num_blocks=grid_blocks)
        self.ray = DirectedRayStream(in_channels=input_channels, width=width)
        self.ray_fusion = RayFusion(channels=width, board_size=board_size)
        self.attention = SerialAttentionStage(
            embed_dim=width,
            num_heads=8,
            board_size=board_size,
            num_blocks=relation_blocks,
        )

    def forward(self, x_spatial: torch.Tensor, s_scalar: torch.Tensor) -> torch.Tensor:
        # Stem
        h_stem = self.stem(x_spatial)
        
        # FiLM Conditioning
        film_params = self.film_gen(s_scalar)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        h_0 = gamma * h_stem + beta
        
        # Parallel streams
        h_grid = self.grid(h_0)
        h_ray = self.ray(x_spatial)
        
        # Fusion and Attention
        h_fused = self.ray_fusion(h_grid, h_ray)
        f_trunk = self.attention(h_fused)
        
        return f_trunk


__all__ = ["CRANETorso"]
