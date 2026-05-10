from __future__ import annotations

import torch
import torch.nn as nn

from .head import ValueHead
from .torso import CRANETorso


class CRANEModel(nn.Module):
    """CRANE-v0: Conv-Ray Attention Network for Evaluation.

    A value-only teacher network for chess evaluation.
    Input consists of 18 spatial planes and 5 scalar variables.
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
        self.torso = CRANETorso(
            input_channels=input_channels,
            width=width,
            board_size=board_size,
            grid_blocks=grid_blocks,
            relation_blocks=relation_blocks,
        )
        self.head = ValueHead(
            trunk_channels=width,
            hidden_dim=width // 3, # C -> C/3 as per spec
        )

    def forward(self, x_spatial: torch.Tensor, s_scalar: torch.Tensor) -> dict[str, torch.Tensor]:
        f_trunk = self.torso(x_spatial, s_scalar)
        value = self.head(f_trunk)
        return {
            "value": value,
            "f_trunk": f_trunk,
        }


__all__ = ["CRANEModel"]
