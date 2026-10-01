import torch
from torch import nn

IMAGE_SIZE = 128
BOTTLENECK_SIZE = 8
STAGES = 4


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, base=64, latent_dim=256, dropout=0.1):
        super().__init__()
        channels = [base * 2 ** i for i in range(STAGES)]
        layers, in_ch = [], 3
        for ch in channels:
            layers += [conv_block(in_ch, ch, stride=2), conv_block(ch, ch)]
            in_ch = ch
        self.features = nn.Sequential(*layers)
        self.to_latent = nn.Linear(channels[-1] * BOTTLENECK_SIZE ** 2, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.dropout(self.to_latent(h))


class Decoder(nn.Module):
    def __init__(self, base=64, latent_dim=256):
        super().__init__()
        channels = [base * 2 ** i for i in reversed(range(STAGES))]
        self.start_channels = channels[0]
        self.from_latent = nn.Linear(latent_dim, channels[0] * BOTTLENECK_SIZE ** 2)
        layers, in_ch = [], channels[0]
        for ch in channels:
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(in_ch, ch), conv_block(ch, ch)]
            in_ch = ch
        self.blocks = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Conv2d(in_ch, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, z):
        h = self.from_latent(z).view(-1, self.start_channels, BOTTLENECK_SIZE, BOTTLENECK_SIZE)
        return self.out(self.blocks(h))


class DenoisingAutoencoder(nn.Module):
    def __init__(self, base=64, latent_dim=256, dropout=0.1):
        super().__init__()
        self.base = base
        self.latent_dim = latent_dim
        self.encoder = Encoder(base, latent_dim, dropout)
        self.decoder = Decoder(base, latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decoder(self.encoder(x))
