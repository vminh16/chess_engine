from __future__ import annotations

import torch
import torch.nn as nn


class DGRUCell(nn.Module):
    """Directional GRU cell applied position by position along a ray."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)
        
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)
        
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # x: (B, input_size), h_prev: (B, hidden_size)
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h_prev))
        h_next = (1 - z) * n + z * h_prev
        return h_next


class DirectedRayStream(nn.Module):
    """8-directional GRU scans simulating line-of-sight for sliding pieces.
    
    Scan directions:
    - N: South -> North (r: 7->0)
    - S: North -> South (r: 0->7)
    - E: West -> East (c: 0->7)
    - W: East -> West (c: 7->0)
    - NE: SW -> NE
    - SW: NE -> SW
    - NW: SE -> NW
    - SE: NW -> SE
    """

    def __init__(self, in_channels: int = 18, width: int = 192) -> None:
        super().__init__()
        if width % 8 != 0:
            raise ValueError("width must be divisible by 8 for 8-direction scan.")
        self.width = width
        self.d_ray = width // 8
        
        self.pre = nn.Conv2d(in_channels, width, kernel_size=1, bias=True)
        
        # 8 distinct GRU cells for 8 directions
        self.grus = nn.ModuleList([
            DGRUCell(self.d_ray, self.d_ray) for _ in range(8)
        ])
        
        self.post = nn.Conv2d(width, width, kernel_size=1, bias=True)

    def _scan_orthogonal(self, x: torch.Tensor, dim: int, reverse: bool, gru: DGRUCell) -> torch.Tensor:
        """Scans along rows or columns."""
        b, c, h, w = x.shape
        
        steps = h if dim == 2 else w
        iterable = range(steps - 1, -1, -1) if reverse else range(steps)
        
        # We need to collect outputs in their original spatial coordinates
        out_tensor = torch.zeros_like(x)
        
        if dim == 2:
            # row scan: W independent rays
            h_prev = torch.zeros(b, c, w, device=x.device)
            for i in iterable:
                x_step = x[:, :, i, :] # (B, C, W)
                
                x_step_flat = x_step.transpose(1, 2).reshape(b * w, c)
                h_prev_flat = h_prev.transpose(1, 2).reshape(b * w, c)
                
                h_next_flat = gru(x_step_flat, h_prev_flat)
                h_prev = h_next_flat.view(b, w, c).transpose(1, 2)
                out_tensor[:, :, i, :] = h_prev
        else:
            # col scan: H independent rays
            h_prev = torch.zeros(b, c, h, device=x.device)
            for i in iterable:
                x_step = x[:, :, :, i] # (B, C, H)
                
                x_step_flat = x_step.transpose(1, 2).reshape(b * h, c)
                h_prev_flat = h_prev.transpose(1, 2).reshape(b * h, c)
                
                h_next_flat = gru(x_step_flat, h_prev_flat)
                h_prev = h_next_flat.view(b, h, c).transpose(1, 2)
                out_tensor[:, :, :, i] = h_prev
                
        return out_tensor

    def _scan_diagonal(self, x: torch.Tensor, direction: str, gru: DGRUCell) -> torch.Tensor:
        """Scans along diagonals."""
        b, c, h, w = x.shape
        out_tensor = torch.zeros_like(x)
        
        if direction == 'NE':
            # SouthWest -> NorthEast
            # Lines of c + r = const
            offsets = range(0, h + w - 1)
            for offset in offsets:
                r_start = max(0, offset - w + 1)
                r_end = min(h, offset + 1)
                h_prev = torch.zeros(b, c, device=x.device)
                # To go SW -> NE, r decreases
                for r in range(r_end - 1, r_start - 1, -1):
                    col = offset - r
                    x_step = x[:, :, r, col]
                    h_next = gru(x_step, h_prev)
                    out_tensor[:, :, r, col] = h_next
                    h_prev = h_next
                    
        elif direction == 'SW':
            # NorthEast -> SouthWest
            # Lines of c + r = const
            offsets = range(0, h + w - 1)
            for offset in offsets:
                r_start = max(0, offset - w + 1)
                r_end = min(h, offset + 1)
                h_prev = torch.zeros(b, c, device=x.device)
                # To go NE -> SW, r increases
                for r in range(r_start, r_end):
                    col = offset - r
                    x_step = x[:, :, r, col]
                    h_next = gru(x_step, h_prev)
                    out_tensor[:, :, r, col] = h_next
                    h_prev = h_next
                    
        elif direction == 'NW':
            # SouthEast -> NorthWest
            # Lines of c - r = const
            offsets = range(-(h-1), w)
            for offset in offsets:
                r_start = max(0, -offset)
                r_end = min(h, w - offset)
                h_prev = torch.zeros(b, c, device=x.device)
                # To go SE -> NW, r decreases
                for r in range(r_end - 1, r_start - 1, -1):
                    col = r + offset
                    x_step = x[:, :, r, col]
                    h_next = gru(x_step, h_prev)
                    out_tensor[:, :, r, col] = h_next
                    h_prev = h_next
                    
        elif direction == 'SE':
            # NorthWest -> SouthEast
            # Lines of c - r = const
            offsets = range(-(h-1), w)
            for offset in offsets:
                r_start = max(0, -offset)
                r_end = min(h, w - offset)
                h_prev = torch.zeros(b, c, device=x.device)
                # To go NW -> SE, r increases
                for r in range(r_start, r_end):
                    col = r + offset
                    x_step = x[:, :, r, col]
                    h_next = gru(x_step, h_prev)
                    out_tensor[:, :, r, col] = h_next
                    h_prev = h_next
                    
        return out_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is X_spatial (B, 18, 8, 8)
        h = self.pre(x)
        chunks = torch.chunk(h, 8, dim=1)

        # 0: N (South->North, r: 7->0) -> reverse row scan
        out_n = self._scan_orthogonal(chunks[0], dim=2, reverse=True, gru=self.grus[0])
        # 1: S (North->South, r: 0->7) -> forward row scan
        out_s = self._scan_orthogonal(chunks[1], dim=2, reverse=False, gru=self.grus[1])
        # 2: E (West->East, c: 0->7) -> forward col scan
        out_e = self._scan_orthogonal(chunks[2], dim=3, reverse=False, gru=self.grus[2])
        # 3: W (East->West, c: 7->0) -> reverse col scan
        out_w = self._scan_orthogonal(chunks[3], dim=3, reverse=True, gru=self.grus[3])
        
        # Diagonals
        out_ne = self._scan_diagonal(chunks[4], 'NE', self.grus[4])
        out_sw = self._scan_diagonal(chunks[5], 'SW', self.grus[5])
        out_nw = self._scan_diagonal(chunks[6], 'NW', self.grus[6])
        out_se = self._scan_diagonal(chunks[7], 'SE', self.grus[7])

        out = torch.cat([out_n, out_s, out_e, out_w, out_ne, out_sw, out_nw, out_se], dim=1)
        out = self.post(out)
        return out


__all__ = ["DirectedRayStream", "DGRUCell"]
