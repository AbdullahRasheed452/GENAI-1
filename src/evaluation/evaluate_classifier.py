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
from src.evaluation.classification_metrics import confusion_matrix, summarize
from src.models.classifier import CorruptionClassifier

SEVERITY_NAMES = {1: "low", 2: "medium", 3: "high"}


@torch.no_grad()
def predict(model, dataset, device):
    loader = DataLoader(dataset, batch_size=256, num_workers=2)
    probs, labels, severities = [], [], []
    for corrupted, _, label, severity, _ in loader:
        probs.append(torch.softmax(model(corrupted.to(device)), dim=1).cpu())
        labels.append(label)
        severities.append(severity)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(severities).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = CorruptionClassifier(cfg["channels"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    probs, labels, severities = predict(model, ManifestDataset("test"), device)
    pred = probs.argmax(1)
    cm = confusion_matrix(labels, pred)
    s = summarize(cm)

    print(f"test entries: {len(labels)}")
    print(f"accuracy {s['accuracy']:.4f}  macro precision {s['macro_precision']:.4f}  "
          f"macro recall {s['macro_recall']:.4f}  macro F1 {s['macro_f1']:.4f}")
    for k, name in enumerate(CLASSES):
        print(f"  {name:12s} precision {s['precision'][k]:.4f}  recall {s['recall'][k]:.4f}  F1 {s['f1'][k]:.4f}  (n={cm[k].sum()})")

    print("\nrecall by corruption and severity")
    by_severity = {}
    for k, name in enumerate(CLASSES[1:], start=1):
        for sev, sev_name in SEVERITY_NAMES.items():
            mask = (labels == k) & (severities == sev)
            by_severity[f"{name}_{sev_name}"] = float((pred[mask] == k).mean())
            print(f"  {name:12s} {sev_name:7s} {by_severity[f'{name}_{sev_name}']:.4f}")

    normalized = cm / cm.sum(1, keepdims=True)
    print("\nnormalized confusion matrix (rows are true classes)")
    print(np.round(normalized, 3))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{normalized[i, j]:.3f}", ha="center", va="center",
                    color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(image)
    fig.tight_layout()
    fig.savefig(out / "classifier_confusion_matrix.png", dpi=150)

    results = {
        "accuracy": s["accuracy"],
        "macro_precision": s["macro_precision"],
        "macro_recall": s["macro_recall"],
        "macro_f1": s["macro_f1"],
        "per_class": {name: {"precision": float(s["precision"][k]), "recall": float(s["recall"][k]), "f1": float(s["f1"][k])} for k, name in enumerate(CLASSES)},
        "recall_by_severity": by_severity,
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_normalized": normalized.tolist(),
    }
    (out / "classifier_results.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out / "classifier_test_predictions.npz", probs=probs, labels=labels, severities=severities)
    print(f"\nsaved results and figure to {out}")


if __name__ == "__main__":
    main()
