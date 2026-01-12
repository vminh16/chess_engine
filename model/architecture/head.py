import torch
import torch.nn as nn

class DecoupledHead(nn.Module):
    def __init__(self, in_channels=128, spatial_channels=32, dropout_rate=0.2): # Thêm dropout
        super(DecoupledHead, self).__init__()
        
        # ---  Mish Activation ---
        # Dùng Mish thay cho ReLU ở nhánh này giúp giữ lại các tín hiệu nhỏ (negative values) tốt hơn
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, spatial_channels, 1, bias=False),
            nn.BatchNorm2d(spatial_channels),
            nn.Mish(inplace=True) # Thay đổi tại đây
        )
        self.spatial_dim = spatial_channels * 8 * 8
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_dim = in_channels
        
        total_dim = self.spatial_dim + self.global_dim
        
        # ---  Dropout ---
        # Ngăn chặn overfitting ở lớp cuối cùng
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.fc = nn.Linear(total_dim, 1)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        s = self.spatial_conv(x)
        s_vec = torch.flatten(s, 1)
        
        g = self.global_pool(x)
        g_vec = torch.flatten(g, 1)
        
        combined = torch.cat([s_vec, g_vec], dim=1)
        
        # Apply Dropout trước khi vào lớp Linear cuối
        combined = self.dropout(combined)
        
        return torch.tanh(self.fc(combined))