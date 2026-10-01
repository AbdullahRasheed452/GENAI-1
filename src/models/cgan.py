import torch
from torch import nn

NUM_STYLES = 3


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_ch)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """U-Net for 128 by 128 images. Style embedding enters at the input and at the bottleneck."""

    def __init__(self, base=64, embed_dim=16, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(NUM_STYLES, embed_dim)
        c = [base, base * 2, base * 4, base * 8, base * 8, base * 8, base * 8]
        self.downs = nn.ModuleList()
        in_ch = 3 + embed_dim
        for i, ch in enumerate(c):
            self.downs.append(Down(in_ch, ch, norm=i not in (0, len(c) - 1)))
            in_ch = ch
        self.fuse = nn.Sequential(nn.Conv2d(c[-1] + embed_dim, c[-1], 1), nn.ReLU(inplace=True))
        self.ups = nn.ModuleList()
        in_ch = c[-1]
        for i, out_ch in enumerate([c[5], c[4], c[3], c[2], c[1], c[0]]):
            self.ups.append(Up(in_ch, out_ch, dropout if i < 3 else 0.0))
            in_ch = out_ch * 2
        self.final = nn.Sequential(nn.ConvTranspose2d(in_ch, 3, 4, stride=2, padding=1), nn.Tanh())

    def forward(self, photo, style):
        e = self.embed(style)
        spatial = e[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        h = torch.cat([photo, spatial], dim=1)
        skips = []
        for down in self.downs:
            h = down(h)
            skips.append(h)
        h = self.fuse(torch.cat([h, e[:, :, None, None]], dim=1))
        for up, skip in zip(self.ups, skips[:-1][::-1]):
            h = up(h, skip)
        return self.final(h)


class PatchDiscriminator(nn.Module):
    """PatchGAN with a 70 by 70 receptive field. Sees photo, style, and a real or generated sketch."""

    def __init__(self, base=64, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(NUM_STYLES, embed_dim)
        self.net = nn.Sequential(
            nn.Conv2d(6 + embed_dim, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, style, sketch):
        e = self.embed(style)
        spatial = e[:, :, None, None].expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.net(torch.cat([photo, sketch, spatial], dim=1))
