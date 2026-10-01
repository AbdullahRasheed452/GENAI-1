import torch
from pytorch_msssim import ssim

PSNR_CAP = 100.0


def psnr(pred, target):
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    value = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
    return value.clamp_max(PSNR_CAP)


def ssim_per_image(pred, target):
    return ssim(pred, target, data_range=1.0, size_average=False)


def absolute_error_map(pred, target):
    return (pred - target).abs().mean(1)
