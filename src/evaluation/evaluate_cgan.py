import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from src.models.cgan import UNetGenerator
from src.training.train_cgan import FS2KDataset

STYLE_NAMES = ("Style 1", "Style 2", "Style 3")


@torch.no_grad()
def score(g, dataset, device):
    loader = DataLoader(dataset, batch_size=64, num_workers=2)
    l1s, ssims, psnrs = [], [], []
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        a, b = (g(photo, style) + 1) / 2, (sketch + 1) / 2
        l1s.append((a - b).abs().flatten(1).mean(1).cpu())
        ssims.append(ssim(a, b, data_range=1.0, size_average=False).cpu())
        mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
        psnrs.append((10 * torch.log10(1.0 / mse)).cpu())
    return {"l1": torch.cat(l1s).numpy(), "ssim": torch.cat(ssims).numpy(), "psnr": torch.cat(psnrs).numpy()}


def show(t):
    return ((t.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def generate(g, photo, style, device):
    return g(photo[None].to(device), torch.tensor([style], device=device))[0].cpu()


def plot_cases(g, dataset, indices, scores, path, device):
    fig, axes = plt.subplots(3, len(indices), figsize=(2.2 * len(indices), 6.8))
    axes = np.array(axes).reshape(3, len(indices))
    for c, i in enumerate(indices):
        photo, sketch, style = dataset[i]
        fake = generate(g, photo, style, device)
        for r, img in enumerate((photo, fake, sketch)):
            axes[r, c].imshow(show(img))
        axes[0, c].set_title(f"{STYLE_NAMES[style]}\nSSIM {scores['ssim'][i]:.2f}", fontsize=9)
    for r, label in enumerate(("photo", "generated", "ground truth")):
        axes[r, 0].set_ylabel(label)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_style_control(g, dataset, indices, path, device):
    fig, axes = plt.subplots(len(indices), 5, figsize=(11, 2.3 * len(indices)))
    for r, i in enumerate(indices):
        photo, sketch, _ = dataset[i]
        images = [photo] + [generate(g, photo, s, device) for s in range(3)] + [sketch]
        for c, img in enumerate(images):
            axes[r, c].imshow(show(img))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    for c, title in enumerate(("photo", "Style 1", "Style 2", "Style 3", "ground truth")):
        axes[0, c].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    g = UNetGenerator(cfg["base"], cfg["embed_dim"], cfg["dropout"]).to(device)
    g.load_state_dict(ckpt["generator"])
    g.eval()
    print("checkpoint epoch", ckpt["epoch"] + 1, "validation objective", round(ckpt["metrics"]["objective"], 4))

    test = FS2KDataset("test")
    scores = score(g, test, device)
    styles = test.style
    rows = []
    groups = [("all", np.ones(len(styles), dtype=bool))] + [(STYLE_NAMES[k], styles == k) for k in range(3)]
    for name, mask in groups:
        row = {"group": name, "n": int(mask.sum()), "l1": float(scores["l1"][mask].mean()),
               "ssim": float(scores["ssim"][mask].mean()), "psnr": float(scores["psnr"][mask].mean())}
        rows.append(row)
        print(f"{name:8s} n={row['n']:5d}  L1 {row['l1']:.4f}  SSIM {row['ssim']:.3f}  PSNR {row['psnr']:.2f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cgan_results.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(out / "cgan_per_image_scores.npz", styles=styles, **scores)

    rng = np.random.default_rng(0)
    picks = [int(i) for k in range(3) for i in rng.choice(np.flatnonzero(styles == k), 4, replace=False)]
    plot_cases(g, test, picks, scores, out / "cgan_examples.png", device)
    failures = [int(i) for i in np.argsort(scores["ssim"])[:4]]
    plot_cases(g, test, failures, scores, out / "cgan_failures.png", device)
    plot_style_control(g, test, [picks[0], picks[4], picks[8], picks[2]], out / "cgan_style_control.png", device)
    print("failure cases:")
    for i in failures:
        print(f"  test index {i}  style {STYLE_NAMES[styles[i]]}  SSIM {scores['ssim'][i]:.3f}  L1 {scores['l1'][i]:.3f}")
    print(f"saved results and figures to {out}")


if __name__ == "__main__":
    main()
