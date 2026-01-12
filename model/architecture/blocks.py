import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---  DropPath (Stochastic Depth) ---
class DropPath(nn.Module):
    """
    Ngẫu nhiên 'bỏ rơi' toàn bộ luồng dữ liệu của block trong khi training.
    Giúp mạng không bị phụ thuộc vào bất kỳ block đơn lẻ nào -> Mạnh hơn.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Tạo mask ngẫu nhiên kích thước [Batch, 1, 1, 1]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        output = x.div(keep_prob) * random_tensor
        return output

# --- Helper Modules ---
class SEBlock(nn.Module):
    # ... (Code cũ giữ nguyên)
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GhostModule(nn.Module):
    # ... (Code cũ giữ nguyên)
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

# --- UPDATED BLOCK 1 ---
class ClassicResBlock(nn.Module):
    def __init__(self, channels, drop_path=0.0): # Thêm tham số drop_path
        super(ClassicResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Tích hợp DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply DropPath trước khi cộng Residual
        out = self.drop_path(out) + residual
        out = self.relu(out)
        return out

# --- UPDATED BLOCK 2 ---
class PhantomBlock(nn.Module):
    def __init__(self, in_chs, out_chs, expand_ratio=1.5, use_se=True, drop_path=0.0): # Thêm drop_path
        super(PhantomBlock, self).__init__()
        hidden_chs = int(in_chs * expand_ratio)
        
        self.ghost_expand = GhostModule(in_chs, hidden_chs, relu=True)
        
        self.dw_conv = nn.Sequential(
            nn.Conv2d(hidden_chs, hidden_chs, 3, 1, 1, groups=hidden_chs, bias=False),
            nn.BatchNorm2d(hidden_chs),
            nn.ReLU(inplace=True)
        )
        
        self.se = SEBlock(hidden_chs) if use_se else nn.Identity()
        self.ghost_proj = GhostModule(hidden_chs, out_chs, relu=False)
        
        self.shortcut = nn.Sequential()
        if in_chs != out_chs:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, out_chs, 1, bias=False),
                nn.BatchNorm2d(out_chs)
            )
            
        # Tích hợp DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.ghost_expand(x)
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.ghost_proj(x)
        
        # Apply DropPath
        x = self.drop_path(x) + residual
        x = self.final_relu(x)
        return x