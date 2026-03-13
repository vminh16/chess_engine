import torch
import torch.nn as nn
from .blocks import DFGBlock

# Import head đã định nghĩa ở trên
from .head import ContextGatedHead

class DGRNChessNet(nn.Module):
    """
    Dilated-Gated ResNet (DGRN) for Chess Engines.
    Đặc điểm: Giữ nguyên resolution 8x8, tầm nhìn đa dạng (Dual-Focus).
    """
    def __init__(self, num_blocks=12, hidden_dim=128, input_channels=18, drop_path_rate=0.1):
        super(DGRNChessNet, self).__init__()
        
        # --- Stem: Entry Point ---
        # Chuyển đổi input thô (18 kênh) sang không gian đặc trưng
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(inplace=True)
        )
        
        # --- Backbone: Deep Thinking ---
        # Tính toán tỉ lệ drop path tăng dần tuyến tính cho từng block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        
        self.blocks = nn.Sequential(*[
            DFGBlock(channels=hidden_dim, drop_path=dpr[i])
            for i in range(num_blocks)
        ])
        
        # --- Head: Evaluation ---
        self.head = ContextGatedHead(in_channels=hidden_dim, hidden_dim=hidden_dim // 2)
        
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số khoa học để hội tụ nhanh."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal tốt cho Mish/ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # BN weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-Gamma Initialization cho Residual Blocks
        # Giúp block khởi đầu như một hàm Identity, huấn luyện mượt hơn.
        # Tìm BN cuối cùng trong phần Fusion của mỗi DFGBlock
        for m in self.modules():
            if isinstance(m, DFGBlock):
                # BN cuối cùng của DFGBlock nằm trong self.fusion[1]
                nn.init.constant_(m.fusion[1].weight, 0)

    def forward(self, x):
        # Đảm bảo input đúng shape cho ONNX (B, C, H, W)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    def predict(self, x, device='cpu'):
        """Hàm helper để infer nhanh 1 sample (không dùng khi export ONNX)."""
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 3: # Thêm batch dim nếu thiếu
                x = x.unsqueeze(0)
            
            x = x.to(device)
            output = self.forward(x)
        return output.item()