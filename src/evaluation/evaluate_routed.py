import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset
from src.evaluation.evaluate_restoration import pick_examples, plot_cases, print_table, summarize
from src.evaluation.metrics import psnr, ssim_per_image
from src.models.autoencoder import DenoisingAutoencoder
from src.models.classifier import CorruptionClassifier

KINDS = CLASSES[1:]


def load_autoencoder(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = DenoisingAutoencoder(cfg["base"], cfg["latent_dim"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = CorruptionClassifier(cfg["channels"], cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def route(x, labels, specialists):
    out = x.clone()
    for k, kind in enumerate(CLASSES):
        if k == 0:
            continue
        mask = labels == k
        if mask.any():
            out[mask] = specialists[kind](x[mask])
    return out


@torch.no_grad()
def score_mode(dataset, device, classifier, specialists, mode):
    loader = DataLoader(dataset, batch_size=256, num_workers=2)
    out = {"psnr": [], "ssim": [], "base_psnr": [], "base_ssim": [], "pred": []}
    for corrupted, clean, label, _, _ in loader:
        corrupted, clean, label = corrupted.to(device), clean.to(device), label.to(device)
        pred = classifier(corrupted).argmax(1)
        restored = route(corrupted, label if mode == "oracle" else pred, specialists)
        out["psnr"].append(psnr(restored, clean).cpu())
        out["ssim"].append(ssim_per_image(restored, clean).cpu())
        out["base_psnr"].append(psnr(corrupted, clean).cpu())
        out["base_ssim"].append(ssim_per_image(corrupted, clean).cpu())
        out["pred"].append(pred.cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--specialist_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = load_classifier(args.classifier, device)
    specialists = {k: load_autoencoder(Path(args.specialist_dir) / f"specialist_{k}.pt", device) for k in KINDS}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test = ManifestDataset("test")
    labels = np.array([CLASSES.index(e["type"]) for e in test.entries])
    results = {}
    for mode in ("oracle", "predicted"):
        scores = score_mode(test, device, classifier, specialists, mode)
        results[mode] = scores
        print(f"\n=== {mode} routing ===")
        rows = summarize(scores, test.entries)
        print_table(rows)
        (out / f"routed_{mode}_results.json").write_text(json.dumps(rows, indent=1))

    pred = results["predicted"]["pred"]
    wrong = np.flatnonzero(pred != labels)
    drop = results["oracle"]["ssim"] - results["predicted"]["ssim"]
    print(f"\nmisrouted entries: {len(wrong)} of {len(labels)}")
    if len(wrong):
        print(f"mean SSIM change caused by misrouting: {-drop[wrong].mean():+.3f}")
        worst = wrong[np.argsort(-drop[wrong])][:4]
        print("routing failure cases:")
        for i in worst:
            e = test.entries[i]
            print(f"  {e['image']:26s} true {e['type']:12s} {e['severity']:7s} routed to {CLASSES[pred[i]]:12s} "
                  f"oracle SSIM {results['oracle']['ssim'][i]:.3f}  predicted SSIM {results['predicted']['ssim'][i]:.3f}")
        def predicted_restore(x):
            return route(x, classifier(x).argmax(1), specialists)

        plot_cases(test, predicted_restore, [int(i) for i in worst], out / "routed_failures.png", device, results["predicted"])

    def predicted_restore(x):
        return route(x, classifier(x).argmax(1), specialists)

    examples = pick_examples(test.entries)
    plot_cases(test, predicted_restore, examples[:6], out / "routed_examples_1.png", device, results["predicted"])
    plot_cases(test, predicted_restore, examples[6:], out / "routed_examples_2.png", device, results["predicted"])
    np.savez_compressed(out / "routed_per_image_scores.npz", **{f"{m}_{k}": v for m, s in results.items() for k, v in s.items()})
    print(f"\nsaved results and figures to {out}")


if __name__ == "__main__":
    main()
