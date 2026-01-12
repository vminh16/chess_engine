import torch
import torch.nn as nn
from .blocks import ClassicResBlock, PhantomBlock
from .head import DecoupledHead

class PhantomChessNet(nn.Module):
    def __init__(self, drop_path_rate=0.2): # Tỉ lệ drop path tối đa
        super(PhantomChessNet, self).__init__()

        # Tính toán tỉ lệ drop cho từng block (tăng tuyến tính từ 0 đến drop_path_rate)
        # Tổng số blocks = 3 (Stg1) + 4 (Stg2) + 2 (Stg3a) + 2 (Stg3c) = 11 blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 11)]

        self.stem = nn.Sequential(
            nn.Conv2d(18, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 1: (Foundation) - DropPath thấp
        self.stage1 = nn.Sequential(
            ClassicResBlock(64, drop_path=dpr[0]),
            ClassicResBlock(64, drop_path=dpr[1]),
            ClassicResBlock(64, drop_path=dpr[2])
        )

        # Stage 2: (Backbone) - DropPath trung bình
        self.stage2 = nn.Sequential(
            PhantomBlock(64, 128, expand_ratio=2, drop_path=dpr[3]),
            PhantomBlock(128, 128, expand_ratio=1.5, drop_path=dpr[4]),
            PhantomBlock(128, 128, expand_ratio=1.5, drop_path=dpr[5]),
            PhantomBlock(128, 128, expand_ratio=1.5, drop_path=dpr[6])
        )

        # Stage 3: (Deep Thinking) - DropPath cao nhất
        self.stage3_expand = nn.Sequential(
            PhantomBlock(128, 256, expand_ratio=2, drop_path=dpr[7]),
            PhantomBlock(256, 256, expand_ratio=1.5, drop_path=dpr[8])
        )

        self.stage3_compress = nn.Sequential(
            nn.Conv2d(256, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.Mish(inplace=True) # Dùng Mish ở đây cho "mượt"
        )

        self.stage3_refine = nn.Sequential(
            ClassicResBlock(128, drop_path=dpr[9]),
            ClassicResBlock(128, drop_path=dpr[10])
        )

        self.head = DecoupledHead(in_channels=128, spatial_channels=32, dropout_rate=0.2)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # ---  Zero-Gamma Initialization ---
        # Tìm các lớp BN cuối cùng trong mỗi Block và set weight=0
        # Mục đích: Biến block thành Identity ngay khi bắt đầu train -> Hội tụ cực nhanh
        for m in self.modules():
            if isinstance(m, ClassicResBlock):
                nn.init.constant_(m.bn2.weight, 0) # BN2 là lớp cuối của Classic
            elif isinstance(m, PhantomBlock):
                
                pass 

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3_expand(x)
        x = self.stage3_compress(x)
        x = self.stage3_refine(x)
        x = self.head(x)
        return x

    def predict(self, x, device='cpu'):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 3: x = x.unsqueeze(0)
            x = x.to(device)
            output = self.forward(x)
        return output.item()
        
    def load_model(self, path):
        """Load model state"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()