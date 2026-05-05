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


class SimplifiedGlobalHead(nn.Module):
    """Global-pooling-only value head that eliminates the spatial flatten bottleneck.

    Rationale (from empirical analysis):
    - The ``ResidualGainValueHead`` spatial branch flattens to ``hidden_dim * 64``
      which creates a massive linear layer that is the primary site of gradient
      interference (measured head cosine = -0.724, 2.83x worse than backbone).
    - This head removes that bottleneck by using only global-pooled features,
      relying on the backbone to have already integrated all spatial information
      (which it does — RF > 8 from just 2 blocks).
    """

    def __init__(self, in_channels=128, hidden_dim=128, dropout=0.1,
                 output_mode="tanh"):
        super().__init__()
        self.output_mode = output_mode
        pooled_dim = in_channels * 2  # avg + max pool concat

        self.mlp = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_logits(self, x):
        b, c, _, _ = x.shape
        g_avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        g_max = F.adaptive_max_pool2d(x, 1).view(b, c)
        pooled = torch.cat([g_avg, g_max], dim=1)
        return self.mlp(pooled)

    def forward(self, x):
        logits = self.forward_logits(x)
        if self.output_mode == "tanh":
            return torch.tanh(logits)
        if self.output_mode == "linear":
            return logits
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")


class RegimeSeparatedHead(nn.Module):
    """Value head that separates magnitude and sign prediction.

    Rationale (from empirical analysis):
    - Center vs mid-band gradient interference is strongest in the head because
      a single scalar output must encode both *direction* (sign) and *magnitude*.
    - By splitting these into separate linear layers that share a common feature
      trunk, the sign-gradient and magnitude-gradient can partially decouple,
      reducing the measured -0.724 head cosine interference.
    - Combined logit: ``z = magnitude_logit * tanh(sign_logit)``.
    - Output: ``tanh(z)`` which stays in [-1, 1].
    """

    def __init__(self, in_channels=128, hidden_dim=128, dropout=0.1,
                 output_mode="tanh"):
        super().__init__()
        self.output_mode = output_mode
        pooled_dim = in_channels * 2

        self.shared = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Dropout(dropout),
        )

        # Magnitude branch: how far from zero
        self.magnitude = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Sign branch: which side is winning
        self.sign = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_logits(self, x):
        b, c, _, _ = x.shape
        g_avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        g_max = F.adaptive_max_pool2d(x, 1).view(b, c)
        pooled = torch.cat([g_avg, g_max], dim=1)

        shared_feat = self.shared(pooled)
        mag_logit = self.magnitude(shared_feat)
        sign_logit = self.sign(shared_feat)
        # Combined logit: magnitude * direction
        return mag_logit * torch.tanh(sign_logit)

    def forward(self, x):
        logits = self.forward_logits(x)
        if self.output_mode == "tanh":
            return torch.tanh(logits)
        if self.output_mode == "linear":
            return logits
        raise ValueError(f"Unsupported output_mode: {self.output_mode}")
