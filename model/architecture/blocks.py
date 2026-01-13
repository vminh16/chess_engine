import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Coordinate Attention Module ---
class CoordinateAttention(nn.Module):
    """
    Coordinate Attention: Thay thế SEBlock, tập trung vào thông tin vị trí không gian.
    Pooling theo chiều dọc (H) và ngang (W) riêng biệt để mã hóa thông tin tọa độ.
    """
    def __init__(self, channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.channels = channels
        reduction_channels = max(8, channels // reduction)  # Tối thiểu 8 channels
        
        # Pooling theo chiều cao (H) - Global Average Pooling theo chiều ngang
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # (H, 1)
        # Pooling theo chiều rộng (W) - Global Average Pooling theo chiều cao
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # (1, W)
        
        # Convolution để mã hóa thông tin vị trí
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            nn.Mish(inplace=True)
        )
        self.conv_w = nn.Sequential(
            nn.Conv2d(channels, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            nn.Mish(inplace=True)
        )
        
        # Excitation: Sinh ra gate weights
        self.conv_h_gate = nn.Sequential(
            nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.conv_w_gate = nn.Sequential(
            nn.Conv2d(reduction_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor với shape (N, C, H, W)
        Returns:
            Output tensor với shape (N, C, H, W)
        """
        identity = x
        
        # Pooling theo chiều cao (H): (N, C, H, W) -> (N, C, H, 1)
        x_h = self.pool_h(x)
        # Pooling theo chiều rộng (W): (N, C, H, W) -> (N, C, 1, W)
        x_w = self.pool_w(x)
        x_w = x_w.permute(0, 1, 3, 2)  # (N, C, 1, W) -> (N, C, W, 1) để match với x_h
        
        # Encode thông tin vị trí
        y_h = self.conv_h(x_h)  # (N, reduction_channels, H, 1)
        y_w = self.conv_w(x_w)  # (N, reduction_channels, W, 1)
        
        # Excitation: Tạo gate weights
        gate_h = self.conv_h_gate(y_h)  # (N, C, H, 1)
        gate_w = self.conv_w_gate(y_w)  # (N, C, W, 1)
        gate_w = gate_w.permute(0, 1, 3, 2)  # (N, C, W, 1) -> (N, C, 1, W)
        
        # Áp dụng gates: Nhân với gate_H và gate_W
        # Gate_H: "Hàng nào đang bị phong tỏa"
        # Gate_W: "Cột nào đang mở"
        out = identity * gate_h * gate_w
        
        return out

# --- Dual-Focus Gated Block (DFG-Block) ---
class DFGBlock(nn.Module):
    """
    Dual-Focus Gated Block: Block mới thay thế ClassicResBlock và PhantomBlock.
    
    Luồng xử lý:
    1. Split: Chia input thành 2 nhánh channels bằng nhau
    2. Dual Transformation:
       - Local: Conv 3x3 (Dilation=1) - Cận chiến, mối quan hệ liền kề
       - Remote: Conv 3x3 (Dilation=2) - Tầm xa, đường chéo dài, kiểm soát hàng/cột
    3. Fusion: Concatenate + Conv 1x1 + BN + Mish
    4. Coordinate Attention: Thay SEBlock, tập trung vào vị trí không gian
    5. Residual Connection
    """
    def __init__(self, channels, drop_path=0.0):
        """
        Args:
            channels: Số channels đầu vào và đầu ra (giữ nguyên)
            drop_path: Tỉ lệ drop path cho stochastic depth
        """
        super(DFGBlock, self).__init__()
        
        # Kiểm tra channels phải chia hết cho 2
        if channels % 2 != 0:
            raise ValueError(f"DFGBlock: channels ({channels}) must be divisible by 2 for split operation")
        
        self.channels = channels
        split_channels = channels // 2
        
        # --- Split: Chia thành 2 nhánh channels bằng nhau ---
        # Không cần layer, sẽ split trong forward()
        
        # --- Dual Transformation ---
        # Nhánh 1: Local (Cận chiến) - Conv 3x3 với Dilation=1
        self.local_conv = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3, 
                     padding=1, dilation=1, groups=split_channels, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.Mish(inplace=True)
        )
        
        # Nhánh 2: Remote (Tầm xa) - Conv 3x3 với Dilation=2
        # Dilation=2: Trường nhìn tương đương 5x5, không tăng tham số
        self.remote_conv = nn.Sequential(
            nn.Conv2d(split_channels, split_channels, kernel_size=3,
                     padding=2, dilation=2, groups=split_channels, bias=False),
            nn.BatchNorm2d(split_channels),
            nn.Mish(inplace=True)
        )
        
        # --- Fusion: Hợp nhất hai nhánh ---
        # Concatenate: (N, split_channels, H, W) + (N, split_channels, H, W) -> (N, channels, H, W)
        # Conv 1x1 để trộn thông tin và phục hồi số kênh ban đầu
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Mish(inplace=True)
        )
        
        # --- Coordinate Attention ---
        # Thay SEBlock, tập trung vào thông tin vị trí không gian
        self.coord_attn = CoordinateAttention(channels, reduction=32)
        
        # --- DropPath ---
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor với shape (N, channels, H, W)
        Returns:
            Output tensor với shape (N, channels, H, W)
        """
        identity = x
        
        # --- Split: Chia thành 2 nhánh channels bằng nhau ---
        x_local, x_remote = torch.chunk(x, 2, dim=1)  # Mỗi nhánh: (N, channels//2, H, W)
        
        # --- Dual Transformation ---
        # Nhánh Local: Cận chiến, mối quan hệ liền kề
        x_local = self.local_conv(x_local)
        
        # Nhánh Remote: Tầm xa, đường chéo dài, kiểm soát hàng/cột
        x_remote = self.remote_conv(x_remote)
        
        # --- Fusion: Hợp nhất ---
        x_fused = torch.cat([x_local, x_remote], dim=1)  # (N, channels, H, W)
        x_fused = self.fusion(x_fused)
        
        # --- Coordinate Attention ---
        # "Cột nào đang mở", "Hàng nào đang bị phong tỏa"
        x_attn = self.coord_attn(x_fused)
        
        # --- Residual Connection ---
        x_out = self.drop_path(x_attn) + identity
        
        return x_out