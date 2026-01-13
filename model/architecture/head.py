import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoupledHead(nn.Module):
    def __init__(self, in_channels=128, spatial_channels=32, dropout_rate=0.2):
        """
        Decoupled Head với dynamic spatial dimension.
        Spatial dimension được tính động từ input shape trong forward().
        Không hardcode spatial_dim, tính toán hoàn toàn động.
        
        Args:
            in_channels: Số channels đầu vào
            spatial_channels: Số channels sau spatial conv
            dropout_rate: Tỉ lệ dropout
        """
        super(DecoupledHead, self).__init__()
        
        self.in_channels = in_channels
        self.spatial_channels = spatial_channels
        
        # ---  Mish Activation ---
        # Dùng Mish thay cho ReLU ở nhánh này giúp giữ lại các tín hiệu nhỏ (negative values) tốt hơn
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels, 1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.Mish(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_dim = in_channels  # Global dim luôn = in_channels
        
        # ---  Dropout ---
        # Ngăn chặn overfitting ở lớp cuối cùng
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # FC layer: Tạo với max expected dimension (8x8 cho chess board)
        # Trong forward sẽ slice weight nếu cần cho các input size khác
        # Expected: spatial_channels * 8 * 8 + in_channels
        max_expected_spatial_dim = spatial_channels * 8 * 8
        max_expected_total_dim = max_expected_spatial_dim + self.global_dim
        self.fc = nn.Linear(max_expected_total_dim, 1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass với dynamic spatial dimension calculation.
        Tính spatial_dim động từ input shape, không hardcode.
        
        Args:
            x: Input tensor với shape (N, C, H, W)
        
        Returns:
            Output tensor với shape (N, 1)
        """
        # Spatial path
        s = self.spatial_conv(x)  # Shape: (N, spatial_channels, H, W)
        s_vec = torch.flatten(s, 1)  # Shape: (N, spatial_channels * H * W)
        
        # Global path
        g = self.global_pool(x)  # Shape: (N, in_channels, 1, 1)
        g_vec = torch.flatten(g, 1)  # Shape: (N, in_channels)
        
        # Concatenate - tính total_dim động từ actual shape
        combined = torch.cat([s_vec, g_vec], dim=1)  # Shape: (N, dynamic_total_dim)
        total_dim = combined.shape[1]
        
        # Apply Dropout
        combined = self.dropout(combined)
        
        # Linear layer: Slice weight và bias nếu total_dim < max_expected
        # Điều này cho phép model hoạt động với các input size khác nhau
        if total_dim <= self.fc.in_features:
            # Slice weight và bias để match actual dimension
            weight = self.fc.weight[:, :total_dim]  # Shape: (1, total_dim)
            bias = self.fc.bias  # Shape: (1,)
            output = F.linear(combined, weight, bias)
        else:
            # Nếu total_dim > expected, pad input (không nên xảy ra với chess board 8x8)
            # Nhưng để an toàn, vẫn xử lý
            padding = torch.zeros(combined.shape[0], 
                                 total_dim - self.fc.in_features,
                                 device=combined.device, 
                                 dtype=combined.dtype)
            combined_padded = torch.cat([combined[:, :self.fc.in_features], padding], dim=1)
            output = self.fc(combined_padded)
        
        return torch.tanh(output)