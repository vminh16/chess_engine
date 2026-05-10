from __future__ import annotations

import torch
import torch.nn as nn


def _build_rel_pos_index(size: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij"),
        dim=-1,
    )
    coords_flat = coords.view(-1, 2)
    rel = coords_flat[:, None, :] - coords_flat[None, :, :]
    rel = rel + (size - 1)
    rel_index = rel[..., 0] * (2 * size - 1) + rel[..., 1]
    return rel_index.to(torch.long)


class RelationBlock(nn.Module):
    """Pre-LN Transformer block with relative position bias and SiLU FFN.
    
    FFN Expansion factor is 2.
    """

    def __init__(self, embed_dim: int, num_heads: int, board_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.board_size = board_size
        self.num_tokens = board_size * board_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim, bias=False)
        )

        rel_pos_index = _build_rel_pos_index(board_size)
        self.register_buffer("rel_pos_index", rel_pos_index, persistent=False)
        self.relative_bias = nn.Parameter(
            torch.zeros(num_heads, 2 * board_size - 1, 2 * board_size - 1)
        )

    def _relative_bias(self) -> torch.Tensor:
        bias = self.relative_bias.view(self.num_heads, -1)
        bias = bias[:, self.rel_pos_index]
        return bias.view(self.num_heads, self.num_tokens, self.num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        batch, num_tokens, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self._relative_bias().unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, num_tokens, self.embed_dim)
        out = self.out_proj(out)
        x = residual + out
        
        x = x + self.mlp(self.norm2(x))
        return x


class SerialAttentionStage(nn.Module):
    """Applies N attention blocks to spatial feature maps.

    Input: (B, C, H, W)
    Output: (B, C, H, W)
    """

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 8,
        board_size: int = 8,
        num_blocks: int = 5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.embed_dim = embed_dim
        self.board_size = board_size
        self.blocks = nn.Sequential(
            *[RelationBlock(embed_dim, num_heads, board_size) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → tokens: (B, T, C)
        tokens = x.flatten(2).transpose(1, 2)
        out = self.blocks(tokens)
        batch = out.shape[0]
        return out.transpose(1, 2).reshape(batch, self.embed_dim, self.board_size, self.board_size)


__all__ = ["RelationBlock", "SerialAttentionStage"]
