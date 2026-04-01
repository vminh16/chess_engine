import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGainValueHead(nn.Module):
    """
    Value head that fixes the main bottleneck of ContextGatedHead.

    - Residual gain branch: can amplify and suppress features in a bounded way.
    - Spatial score branch: reads local/tactical structure from the 8x8 tensor.
    - Global score branch: adds a direct signed scalar from pooled context.
    """

    def __init__(self, in_channels=128, hidden_dim=64, gain_limit=0.5, output_mode="tanh"):
        super().__init__()
        self.output_mode = output_mode
        self.gain_limit = float(gain_limit)
        pooled_dim = in_channels * 2
        global_hidden = max(4, hidden_dim // 2)

        self.gain_mlp = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, in_channels),
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(inplace=True),
        )

        self.spatial_score = nn.Sequential(
            nn.Linear(hidden_dim * 64, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.global_score = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, global_hidden),
            nn.Mish(inplace=True),
            nn.Linear(global_hidden, 1),
        )

    def forward_logits(self, x):
        b, c, _, _ = x.shape
        g_avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        g_max = F.adaptive_max_pool2d(x, 1).view(b, c)
        pooled = torch.cat([g_avg, g_max], dim=1)

        gain = 1.0 + self.gain_limit * torch.tanh(self.gain_mlp(pooled))
        x_mod = x * gain.view(b, c, 1, 1)

        spatial_feat = self.spatial_conv(x_mod).flatten(1)
        spatial_score = self.spatial_score(spatial_feat)
        global_score = self.global_score(pooled)
        return spatial_score + global_score

    def forward(self, x):
        logits = self.forward_logits(x)
        if self.output_mode == "tanh":
            return torch.tanh(logits)
        if self.output_mode == "linear":
            return logits
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")
