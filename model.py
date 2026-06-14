
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
  

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7,
                 stride: int = 1, dropout: float = 0.2):
        super().__init__()

        pad = kernel_size // 2  # "same" padding

        self.bn1   = nn.BatchNorm1d(in_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad, bias=False)

        self.bn2   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               stride=1, padding=pad, bias=False)

        self.drop  = nn.Dropout(p=dropout)

        # Skip-connection projection (needed when channels or stride change)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)

        return out + identity


class ResNet1D(nn.Module):
    

    def __init__(
        self,
        n_leads: int    = 12,
        n_classes: int  = 5,
        base_ch: int    = 64,
        layers: list    = None,
        kernel_size: int = 7,
        dropout: float  = 0.2,
    ):
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]   # ResNet-18-style depth

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_ch, kernel_size=15, stride=2, padding=7, bias=False),  # /2
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),                              # /2
        )  # output: (B, base_ch, 1250)

        ch = base_ch
        self.stage1 = self._make_stage(ch,      ch,      layers[0], stride=1, ks=kernel_size, dp=dropout)
        self.stage2 = self._make_stage(ch,      ch*2,    layers[1], stride=2, ks=kernel_size, dp=dropout)
        self.stage3 = self._make_stage(ch*2,    ch*4,    layers[2], stride=2, ks=kernel_size, dp=dropout)
        self.stage4 = self._make_stage(ch*4,    ch*8,    layers[3], stride=2, ks=kernel_size, dp=dropout)
        # After stage4: (B, base_ch*8, ~157)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),       # (B, base_ch*8, 1)
            nn.Flatten(),                   # (B, base_ch*8)
            nn.Dropout(p=dropout),
            nn.Linear(ch * 8, n_classes),  # raw logits
        )

        # Weight init
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks, stride, ks, dp):
        blocks = [ResBlock1D(in_ch, out_ch, ks, stride=stride, dropout=dp)]
        for _ in range(1, n_blocks):
            blocks.append(ResBlock1D(out_ch, out_ch, ks, stride=1, dropout=dp))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 12, 5000)
        returns logits : (B, n_classes)
        """
        x = self.stem(x)      # (B, 64,  1250)
        x = self.stage1(x)    # (B, 64,  1250)
        x = self.stage2(x)    # (B, 128,  625)
        x = self.stage3(x)    # (B, 256,  313)
        x = self.stage4(x)    # (B, 512,  157)
        x = self.head(x)      # (B, n_classes)
        return x


def build_model(n_classes: int = 5,
                n_leads: int = 12,
                variant: str = "resnet18") -> ResNet1D:
   
    configs = {
        "resnet18": dict(base_ch=64,  layers=[2, 2, 2, 2]),
        "resnet34": dict(base_ch=64,  layers=[3, 4, 6, 3]),
        "small":    dict(base_ch=32,  layers=[1, 1, 1, 1]),
    }
    cfg = configs[variant]
    return ResNet1D(n_leads=n_leads, n_classes=n_classes, **cfg)


if __name__ == "__main__":
    model = build_model(n_classes=5)
    dummy = torch.randn(4, 12, 5000)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expected: (4, 5)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable:,}")