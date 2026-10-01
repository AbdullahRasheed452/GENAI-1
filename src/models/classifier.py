from torch import nn

CHANNEL_CONFIGS = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "large": (64, 128, 256, 512),
}


def block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CorruptionClassifier(nn.Module):
    def __init__(self, channels="medium", dropout=0.3, num_classes=4):
        super().__init__()
        layers, in_ch = [], 3
        for width in CHANNEL_CONFIGS[channels]:
            layers.append(block(in_ch, width))
            in_ch = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.features(x)).flatten(1))
