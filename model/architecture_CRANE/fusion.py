from __future__ import annotations

import torch
import torch.nn as nn


class RayFusion(nn.Module):
    """Gated fusion of GridStream and RayStream outputs for CRANE-v0.

    Combines two spatial feature maps via a learned gate:
        h_fused = h_grid + alpha * sigmoid(g([h_grid; h_ray])) * f([h_grid; h_ray])
    """

    def __init__(self, channels: int = 192, board_size: int = 8) -> None:
        super().__init__()
        in_channels = channels * 2
        
        self.g_conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)
        self.f_conv = nn.Conv2d(in_channels, channels, kernel_size=1, bias=True)
        
        # Initialize gate bias to +1.0 to prevent early collapse
        nn.init.constant_(self.g_conv.bias, 1.0)
        
        # Alpha is a learnable residual scale, initialized to 0.1
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, h_grid: torch.Tensor, h_ray: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([h_grid, h_ray], dim=1)
        f = self.f_conv(fused)
        g = torch.sigmoid(self.g_conv(fused))
        return h_grid + self.alpha * g * f


__all__ = ["RayFusion"]
