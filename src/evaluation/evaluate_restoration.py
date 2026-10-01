import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset
from src.evaluation.metrics import absolute_error_map, psnr, ssim_per_image
from src.models.autoencoder import DenoisingAutoencoder

KINDS = CLASSES[1:]
SEVERITIES = ("low", "medium", "high")


@torch.no_grad()
def score_all(restore_fn, dataset, device, batch_size=256):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    out = {"psnr": [], "ssim": [], "base_psnr": [], "base_ssim": []}
    for corrupted, clean, *_ in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        pred = restore_fn(corrupted)
        out["psnr"].append(psnr(pred, clean).cpu())
        out["ssim"].append(ssim_per_image(pred, clean).cpu())
        out["base_psnr"].append(psnr(corrupted, clean).cpu())
        out["base_ssim"].append(ssim_per_image(corrupted, clean).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summarize(scores, entries):
    kinds = np.array([e["type"] for e in entries])
    sevs = np.array([e["severity"] for e in entries])
    groups = [("clean", "none", kinds == "clean")]
    for kind in KINDS:
        for sev in SEVERITIES:
            groups.append((kind, sev, (kinds == kind) & (sevs == sev)))
        groups.append((kind, "all", kinds == kind))
    groups.append(("all corrupted", "all", kinds != "clean"))
    rows = []
    for kind, sev, mask in groups:
        rows.append(
            {
                "condition": kind,
                "severity": sev,
                "n": int(mask.sum()),
                "psnr": float(scores["psnr"][mask].mean()),
                "ssim": float(scores["ssim"][mask].mean()),
                "input_psnr": None if kind == "clean" else float(scores["base_psnr"][mask].mean()),
                "input_ssim": float(scores["base_ssim"][mask].mean()),
            }
        )
    return rows


def print_table(rows):
    print(f"{'condition':15s}{'severity':9s}{'n':>7s}{'PSNR':>8s}{'SSIM':>8s}{'in PSNR':>9s}{'in SSIM':>9s}")
    for r in rows:
        in_psnr = "n/a" if r["input_psnr"] is None else f"{r['input_psnr']:.2f}"
        print(f"{r['condition']:15s}{r['severity']:9s}{r['n']:7d}{r['psnr']:8.2f}{r['ssim']:8.3f}{in_psnr:>9s}{r['input_ssim']:9.3f}")


def pick_examples(entries, seed=0):
    rng = np.random.default_rng(seed)
    kinds = np.array([e["type"] for e in entries])
    sevs = np.array([e["severity"] for e in entries])

    def one(kind, sev):
        return int(rng.choice(np.flatnonzero((kinds == kind) & (sevs == sev))))

    picks = [one("clean", "none") for _ in range(3)]
    for kind in KINDS:
        for sev in SEVERITIES:
            picks.append(one(kind, sev))
    return picks


def pick_failures(scores, entries):
    kinds = np.array([e["type"] for e in entries])
    delta = scores["ssim"] - scores["base_ssim"]
    picks = []
    for kind in KINDS:
        idx = np.flatnonzero(kinds == kind)
        picks.append(int(idx[np.argmin(delta[idx])]))
    corrupted = np.flatnonzero(kinds != "clean")
    order = corrupted[np.argsort(scores["ssim"][corrupted])]
    picks.append(int(next(i for i in order if i not in picks)))
    return picks


def plot_cases(dataset, restore_fn, indices, path, device, scores):
    items = [dataset[i] for i in indices]
    corrupted = torch.stack([it[0] for it in items]).to(device)
    clean = torch.stack([it[1] for it in items]).to(device)
    with torch.no_grad():
        pred = restore_fn(corrupted)
    err = absolute_error_map(pred, clean).cpu().numpy()

    def show(t):
        return t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    n = len(indices)
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9.2))
    axes = np.array(axes).reshape(4, n)
    for c, i in enumerate(indices):
        e = dataset.entries[i]
        axes[0, c].imshow(show(clean[c]))
        axes[1, c].imshow(show(corrupted[c]))
        axes[2, c].imshow(show(pred[c]))
        axes[3, c].imshow(err[c], cmap="inferno", vmin=0, vmax=0.5)
        axes[0, c].set_title(f"{e['type']} {e['severity']}\nSSIM {scores['ssim'][i]:.2f}", fontsize=9)
    for r, label in enumerate(("clean target", "input", "restored", "abs error")):
        axes[r, 0].set_ylabel(label)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_evaluation(restore_fn, name, out_dir, device):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = ManifestDataset("test")
    scores = score_all(restore_fn, test, device)
    rows = summarize(scores, test.entries)
    print_table(rows)
    (out / f"{name}_results.json").write_text(json.dumps(rows, indent=1))
    np.savez_compressed(out / f"{name}_per_image_scores.npz", **scores)

    examples = pick_examples(test.entries)
    plot_cases(test, restore_fn, examples[:6], out / f"{name}_examples_1.png", device, scores)
    plot_cases(test, restore_fn, examples[6:], out / f"{name}_examples_2.png", device, scores)

    failures = pick_failures(scores, test.entries)
    plot_cases(test, restore_fn, failures, out / f"{name}_failures.png", device, scores)
    print("\nfailure cases:")
    for i in failures:
        e = test.entries[i]
        print(f"  {e['image']:28s} {e['type']:12s} {e['severity']:7s} "
              f"model SSIM {scores['ssim'][i]:.3f}  input SSIM {scores['base_ssim'][i]:.3f}")
    print(f"\nsaved results and figures to {out}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--name", default="task1")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = DenoisingAutoencoder(cfg["base"], cfg["latent_dim"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    run_evaluation(model, args.name, args.out_dir, device)


if __name__ == "__main__":
    main()
