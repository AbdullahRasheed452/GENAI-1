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
from src.evaluation.evaluate_restoration import run_evaluation
from src.models.moe import load_moe

SEV = {0: "none", 1: "low", 2: "medium", 3: "high"}
BRANCHES = ["identity", "salt_pepper", "blur", "occlusion"]


@torch.no_grad()
def collect(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, num_workers=2)
    ws, labels, sevs = [], [], []
    for corrupted, _, label, severity, _ in loader:
        _, w = model(corrupted.to(device))
        ws.append(w.cpu())
        labels.append(label)
        sevs.append(severity)
    return torch.cat(ws).numpy(), torch.cat(labels).numpy(), torch.cat(sevs).numpy()


def group_table(w, labels, sevs):
    groups = [("clean", "none", labels == 0)]
    for k in (1, 2, 3):
        for s in (1, 2, 3):
            groups.append((CLASSES[k], SEV[s], (labels == k) & (sevs == s)))
    rows = []
    print(f"{'true condition':14s}{'severity':9s}" + "".join(f"{b:>13s}" for b in BRANCHES))
    for name, sev, mask in groups:
        mean = w[mask].mean(0)
        rows.append({"condition": name, "severity": sev, "n": int(mask.sum()), "mean_weights": mean.tolist()})
        print(f"{name:14s}{sev:9s}" + "".join(f"{v:13.3f}" for v in mean))
    return rows


def plot_heatmap(rows, path):
    matrix = np.array([r["mean_weights"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(BRANCHES, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r['condition']} {r['severity']}" for r in rows])
    ax.set_xlabel("branch")
    ax.set_ylabel("true condition")
    for i in range(matrix.shape[0]):
        for j in range(4):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(image, label="mean routing weight")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_histograms(w, labels, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for k, ax in enumerate(axes):
        ax.hist(w[labels == k, k], bins=30, range=(0, 1), color="tab:blue")
        ax.set_title(f"true: {CLASSES[k]}")
        ax.set_xlabel(f"weight on {BRANCHES[k]} branch")
        ax.set_yscale("log")
    axes[0].set_ylabel("images (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def plot_weight_cases(model, dataset, indices, w, path, device):
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.4))
    axes = np.array(axes).reshape(2, n)
    for c, i in enumerate(indices):
        corrupted, _, label, severity, _ = dataset[i]
        restored, _ = model(corrupted[None].to(device))
        for r, img in enumerate((corrupted, restored[0].cpu())):
            axes[r, c].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        e = dataset.entries[i]
        axes[0, c].set_title(f"{e['type']} {e['severity']}\nid {w[i, 0]:.2f} sp {w[i, 1]:.2f}\nbl {w[i, 2]:.2f} oc {w[i, 3]:.2f}", fontsize=8)
    axes[0, 0].set_ylabel("input")
    axes[1, 0].set_ylabel("restored")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def expert_health(w, labels):
    top = w.argmax(1)
    report = {}
    print("\nexpert health")
    print(f"{'branch':13s}{'mean weight':>13s}{'top choice':>12s}{'weight on other types':>23s}  flag")
    for j, name in enumerate(BRANCHES):
        mean = float(w[:, j].mean())
        top_share = float((top == j).mean())
        other = float(w[labels != j, j].mean())
        flag = "ok"
        if mean < 0.05 or top_share < 0.01:
            flag = "INACTIVE"
        elif other > 0.5:
            flag = "DOMINATES OTHER TYPES"
        report[name] = {"mean_weight": mean, "top_choice_share": top_share, "mean_weight_on_other_types": other, "flag": flag}
        print(f"{name:13s}{mean:13.3f}{top_share:12.3f}{other:23.3f}  {flag}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_moe(args.checkpoint, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print("checkpoint epoch", ckpt["epoch"] + 1, "validation objective", round(ckpt["metrics"]["objective"], 4), "temperature", ckpt["config"]["temperature"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test = ManifestDataset("test")
    w, labels, sevs = collect(model, test, device)
    print("\nmean routing weights by true condition and severity")
    rows = group_table(w, labels, sevs)
    health = expert_health(w, labels)
    plot_heatmap(rows, out / "task3_weight_heatmap.png")
    plot_histograms(w, labels, out / "task3_weight_histograms.png")

    top = w.max(1)
    dominant = np.flatnonzero(top > 0.9)
    spread = np.argsort(top)[:4]
    print(f"\nimages where one expert has more than 0.9 of the weight: {len(dominant)} of {len(w)}")
    print(f"images where the largest weight is below 0.7: {int((top < 0.7).sum())} of {len(w)}")
    rng = np.random.default_rng(0)
    dom_picks = [int(rng.choice(np.flatnonzero((labels == k) & (top > 0.9)))) for k in range(4) if ((labels == k) & (top > 0.9)).any()]
    plot_weight_cases(model, test, dom_picks, w, out / "task3_dominant_cases.png", device)
    plot_weight_cases(model, test, [int(i) for i in spread], w, out / "task3_spread_cases.png", device)
    print("most spread-out cases:")
    for i in spread:
        e = test.entries[int(i)]
        print(f"  {e['image']:26s} {e['type']:12s} {e['severity']:7s} weights {np.round(w[i], 2).tolist()}")

    (out / "task3_routing_weights.json").write_text(json.dumps({"groups": rows, "expert_health": health}, indent=1))
    print("\nrestoration quality on the test manifest")
    run_evaluation(lambda x: model(x)[0], "task3", out, device)


if __name__ == "__main__":
    main()
