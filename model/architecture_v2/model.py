import torch
import torch.nn as nn

from .blocks import DFGBlock
from .head import RegimeSeparatedHead, ResidualGainValueHead, SimplifiedGlobalHead


class DGRNChessNetV2(nn.Module):
    """
    DGRN backbone with a safer value head.

    The backbone stays close to the current model.
    The readout is changed because that is where the main bottleneck lives.
    """

    def __init__(
        self,
        num_blocks=12,
        hidden_dim=128,
        input_channels=18,
        drop_path_rate=0.1,
        head_hidden_dim=None,
        gain_limit=0.5,
        head_type="residual_gain",
        head_dropout=0.1,
        output_mode="tanh",
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = str(head_type).strip().lower()
        head_hidden_dim = hidden_dim // 2 if head_hidden_dim is None else int(head_hidden_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.Mish(inplace=True),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.Sequential(*[DFGBlock(channels=hidden_dim, drop_path=dpr[i]) for i in range(num_blocks)])

        if self.head_type == "residual_gain":
            self.head = ResidualGainValueHead(
                in_channels=hidden_dim,
                hidden_dim=head_hidden_dim,
                gain_limit=gain_limit,
                output_mode=output_mode,
            )
        elif self.head_type == "simplified_global":
            self.head = SimplifiedGlobalHead(
                in_channels=hidden_dim,
                hidden_dim=head_hidden_dim,
                dropout=float(head_dropout),
                output_mode=output_mode,
            )
        elif self.head_type == "regime_separated":
            self.head = RegimeSeparatedHead(
                in_channels=hidden_dim,
                hidden_dim=head_hidden_dim,
                dropout=float(head_dropout),
                output_mode=output_mode,
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, DFGBlock):
                nn.init.constant_(m.fusion[1].weight, 0)

        if isinstance(self.head, ResidualGainValueHead):
            # Start the residual-gain branch exactly at identity.
            nn.init.constant_(self.head.gain_mlp[-1].weight, 0)
            nn.init.constant_(self.head.gain_mlp[-1].bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def forward_logits(self, x):
        return self.head.forward_logits(self.forward_features(x))

    def forward(self, x):
        return self.head(self.forward_features(x))

    def predict(self, x, device="cpu"):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x.to(device)
            output = self.forward(x)
        return output.item()

    def save_model(self, path):
        payload = {
            "state_dict": self.state_dict(),
            "output_mode": self.output_mode,
            "head_type": self.head_type,
            "arch": "DGRNChessNetV2",
        }
        torch.save(payload, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.output_mode = checkpoint.get("output_mode", self.output_mode)
            self.head.output_mode = self.output_mode
            self.load_state_dict(checkpoint["state_dict"])
        else:
            self.load_state_dict(checkpoint)
        self.eval()


class DGRNChessNet(DGRNChessNetV2):
    """Compatibility alias with the tuned default width/depth."""

    def __init__(self, output_mode="tanh"):
        super().__init__(
            num_blocks=20,
            hidden_dim=256,
            input_channels=18,
            drop_path_rate=0.1,
            gain_limit=0.5,
            output_mode=output_mode,
        )
