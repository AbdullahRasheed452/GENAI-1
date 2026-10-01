import torch
from pytorch_msssim import ssim
from torch import nn


class RestorationLoss(nn.Module):
    """alpha * L1 + (1 - alpha) * (1 - SSIM), inputs in the range 0 to 1."""

    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        structure = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
        return self.alpha * l1 + (1.0 - self.alpha) * structure
