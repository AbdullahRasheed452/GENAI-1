import torch
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset
from src.evaluation.metrics import psnr, ssim_per_image
from src.training.losses import RestorationLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(8, 3, 128, 128, device=device)

    loss = RestorationLoss(alpha=0.8)
    print("loss(x, x) =", float(loss(x, x)))
    print("loss(x, noise) =", round(float(loss(x, torch.rand_like(x))), 3))
    for alpha in (1.0, 0.0):
        value = RestorationLoss(alpha)(x, torch.rand_like(x))
        print(f"alpha={alpha}: {float(value):.3f}")

    pred = (x + 0.1).clamp(0, 1) * 0 + (x * 0.5 + 0.2)
    pred = pred.clone().requires_grad_(True)
    RestorationLoss(0.8)(pred, x).backward()
    print("gradient reaches the prediction:", bool(pred.grad.abs().sum() > 0))

    flat = torch.full((2, 3, 128, 128), 0.5, device=device)
    print("PSNR for a constant error of 0.1:", [round(float(v), 2) for v in psnr(flat + 0.1, flat)], "(expected 20.0)")
    print("PSNR of identical images:", float(psnr(flat, flat)[0]), "(capped)")
    print("SSIM of identical images:", round(float(ssim_per_image(x, x)[0]), 4))

    val = ManifestDataset("val")
    loader = DataLoader(val, batch_size=64, num_workers=2)
    scores = {name: {"psnr": [], "ssim": []} for name in CLASSES}
    for corrupted, clean, labels, severity, idx in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        p, s = psnr(corrupted, clean).cpu(), ssim_per_image(corrupted, clean).cpu()
        for i, label in enumerate(labels.tolist()):
            scores[CLASSES[label]]["psnr"].append(float(p[i]))
            scores[CLASSES[label]]["ssim"].append(float(s[i]))

    print("\nbaseline: corrupted input vs clean target, validation manifest")
    for name in CLASSES:
        n = len(scores[name]["ssim"])
        mean_psnr = sum(scores[name]["psnr"]) / n
        mean_ssim = sum(scores[name]["ssim"]) / n
        print(f"{name:12s} n={n}  PSNR {mean_psnr:6.2f} dB  SSIM {mean_ssim:.4f}")


if __name__ == "__main__":
    main()
