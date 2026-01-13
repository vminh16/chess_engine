import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DropPath (Stochastic Depth) ---
# Cần thiết để train các mạng sâu, ngăn chặn overfitting bằng cách ngẫu nhiên bỏ qua toàn bộ block
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Shape: (batch_size, 1, 1, 1) để broadcast cho toàn bộ feature map của sample đó
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# --- Coordinate Attention Module (Optimized) ---
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (Improved):
    Concatenate H và W features trước khi Convolution để tạo sự tương tác thông tin
    giữa chiều dọc và chiều ngang.
    """
    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, channels // reduction)

        # Convolution 1x1 chia sẻ cho cả H và W sau khi concat
        # Giúp giảm số lượng tham số và tăng tính tương tác
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Mish(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        
        # 1. Pooling
        x_h = self.pool_h(x) # (N, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # (N, C, 1, W) -> (N, C, W, 1)

        # 2. Concat & Shared Transformation (Interaction)
        y = torch.cat([x_h, x_w], dim=2) # (N, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        # 3. Split
        x_h_prime, x_w_prime = torch.split(y, [h, w], dim=2)
        x_w_prime = x_w_prime.permute(0, 1, 3, 2) # Trả về (N, C, 1, W)

        # 4. Excitation (Generate Gates)
        a_h = torch.sigmoid(self.conv_h(x_h_prime))
        a_w = torch.sigmoid(self.conv_w(x_w_prime))

        # 5. Reweight
        return x * a_h * a_w

# --- Dual-Focus Gated Block (DFG-Block) ---
class DFGBlock(nn.Module):
    """
    Dual-Focus Gated Block:
    Sử dụng Standard Convolution thay vì Group Conv để tối đa hóa khả năng học (Capacity).
    """
    def __init__(self, channels, drop_path=0.0):
        super(DFGBlock, self).__init__()
        
        if channels % 2 != 0:
            raise ValueError(f"DFGBlock: channels ({channels}) must be divisible by 2")
        
        split_channels = channels // 2
        
        # --- Dual Transformation ---
        # NOTE: Đã bỏ `groups=split_channels`.
        # Đây giờ là Standard Convolution (groups=1 mặc định).
        
        # Nhánh 1: Local (Cận chiến) - Conv 3x3, Dilation=1
        self.local_conv = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3, 
                     padding=1, dilation=1, bias=False), 
            nn.BatchNorm2d(split_channels),
            nn.Mish(inplace=True)
        )
        
        # Nhánh 2: Remote (Tầm xa) - Conv 3x3, Dilation=2
        self.remote_conv = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3,
                     padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.Mish(inplace=True)
        )
        
        # --- Fusion ---
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True)
        )
        
        self.coord_attn = CoordinateAttention(channels, reduction=32)
        
        # DropPath đã được định nghĩa ở trên
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Split
        x_local, x_remote = torch.chunk(x, 2, dim=1)
        
        # Transform
        x_local = self.local_conv(x_local)
        x_remote = self.remote_conv(x_remote)
        
        # Fuse
        x_fused = torch.cat([x_local, x_remote], dim=1)
        x_fused = self.fusion(x_fused)
        
        # Attend
        x_attn = self.coord_attn(x_fused)
        
        # Residual
        x_out = self.drop_path(x_attn) + identity
        
        return x_out