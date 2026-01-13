import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextGatedHead(nn.Module):
    """
    Context-Gated Decoupled Head:
    Tách biệt luồng xử lý nhưng có sự tương tác (Gated).
    Nhánh Global quyết định 'vùng nào quan trọng' để nhánh Spatial tập trung.
    """
    def __init__(self, in_channels=128, hidden_dim=64):
        super(ContextGatedHead, self).__init__()
        
        # --- 1. Global Context Branch (Strategy) ---
        # Input: (B, C, 8, 8) -> Output: (B, C, 1, 1) Gate
        # Dùng Linear thay vì Conv1x1 ở đây để giảm tính toán sau khi pool
        self.global_fc = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim), # *2 do concat Avg + Max
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, in_channels),
            nn.Sigmoid() # Gate range [0, 1]
        )
        
        # --- 2. Spatial Detail Branch (Tactics) ---
        # Input: (B, C, 8, 8) -> Output: Flatten feature
        # Dùng Depthwise Separable Conv để tiết kiệm tham số
        self.spatial_conv = nn.Sequential(
            # Depthwise: Xử lý không gian cho từng kênh riêng biệt
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, 
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            # Pointwise: Trộn thông tin giữa các kênh
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(inplace=True)
        )
        
        # --- 3. Final Evaluation ---
        # Input: hidden_dim * 8 * 8
        self.fc_out = nn.Linear(hidden_dim * 64, 1) # 64 = 8*8 resolution

    def forward(self, x):
        b, c, _, _ = x.shape
        
        # --- A. Extract Global Context ---
        # AvgPool: Đánh giá tổng quan vật chất. MaxPool: Bắt các tín hiệu cực trị (mối đe dọa)
        g_avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        g_max = F.adaptive_max_pool2d(x, 1).view(b, c)
        
        # Tạo Gate vector
        gate = self.global_fc(torch.cat([g_avg, g_max], dim=1))
        gate = gate.view(b, c, 1, 1) # Reshape để broadcast
        
        # --- B. Gating Interaction ---
        # Lọc thông tin: Chỉ giữ lại tín hiệu không gian phù hợp với ngữ cảnh
        x_gated = x * gate 
        
        # --- C. Spatial Processing ---
        s_feat = self.spatial_conv(x_gated)
        
        # --- D. Final Prediction ---
        out = s_feat.flatten(1) # Flatten từ dim 1 trở đi -> (B, hidden_dim*64)
        return torch.tanh(self.fc_out(out))