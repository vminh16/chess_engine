import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth applied per-sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention with a lighter bottleneck.
    Reduction=8 is kept because reduction=32 was too aggressive at hidden_dim=256.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Mish(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h_prime, x_w_prime = torch.split(y, [h, w], dim=2)
        x_w_prime = x_w_prime.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(x_h_prime))
        a_w = torch.sigmoid(self.conv_w(x_w_prime))
        return x * a_h * a_w


class DFGBlock(nn.Module):
    """
    Dual-Focus block with local and dilated remote branches.
    Attend-before-fuse is kept because it avoids the zero-gamma dead-start issue.
    """

    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"DFGBlock: channels ({channels}) must be divisible by 2")

        split_channels = channels // 2
        self.local_conv = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.Mish(inplace=True),
        )
        self.remote_conv = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.Mish(inplace=True),
        )
        self.coord_attn = CoordinateAttention(channels, reduction=8)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        x_local, x_remote = torch.chunk(x, 2, dim=1)
        x_local = self.local_conv(x_local)
        x_remote = self.remote_conv(x_remote)

        x_concat = torch.cat([x_local, x_remote], dim=1)
        x_attn = self.coord_attn(x_concat)
        x_fused = self.fusion(x_attn)
        return self.drop_path(x_fused) + identity
