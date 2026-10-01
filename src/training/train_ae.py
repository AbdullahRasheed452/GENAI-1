import argparse
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset, PetTrainDataset
from src.evaluation.metrics import psnr, ssim_per_image
from src.models.autoencoder import DenoisingAutoencoder
from src.training.losses import RestorationLoss

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "task1_universal_ae"

DEFAULTS = {
    "base": 32,
    "latent_dim": 256,
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 64,
    "alpha": 0.8,
    "epochs": 10,
    "seed": 42,
    "num_workers": 2,
    "condition": "random",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    l1s, ssims, psnrs, labels = [], [], [], []
    for corrupted, clean, label, _, _ in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        pred = model(corrupted)
        l1s.append((pred - clean).abs().flatten(1).mean(1).cpu())
        ssims.append(ssim_per_image(pred, clean).cpu())
        psnrs.append(psnr(pred, clean).cpu())
        labels.append(label)
    l1, s, p, lab = (torch.cat(v) for v in (l1s, ssims, psnrs, labels))
    result = {"l1": l1.mean().item(), "ssim": s.mean().item(), "psnr": p.mean().item()}
    for k, name in enumerate(CLASSES):
        mask = lab == k
        if not mask.any():
            continue
        result[f"{name}_ssim"] = s[mask].mean().item()
        result[f"{name}_psnr"] = p[mask].mean().item()
    result["objective"] = 0.5 * result["l1"] + 0.5 * (1.0 - result["ssim"])
    return result


def train(config, tracking_uri=None, checkpoint=None, trial=None, run_name=None):
    cfg = {**DEFAULTS, **config}
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PetTrainDataset(condition=cfg["condition"], seed=cfg["seed"])
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_ds = ManifestDataset("val")
    if cfg["condition"] != "random":
        keep = [i for i, e in enumerate(val_ds.entries) if e["type"] == cfg["condition"]]
        val_ds = Subset(val_ds, keep)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=cfg["num_workers"])

    model = DenoisingAutoencoder(cfg["base"], cfg["latent_dim"], cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    loss_fn = RestorationLoss(cfg["alpha"])

    experiment = EXPERIMENT if cfg["condition"] == "random" else f"task2_specialist_{cfg['condition']}"
    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment)
    best = float("inf")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        for epoch in range(cfg["epochs"]):
            started = time.time()
            train_ds.set_epoch(epoch)
            model.train()
            total, batches = 0.0, 0
            for corrupted, clean, _ in loader:
                corrupted, clean = corrupted.to(device), clean.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(corrupted), clean)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            scheduler.step()

            metrics = evaluate(model, val_loader, device)
            metrics["train_loss"] = total / batches
            mlflow.log_metrics(metrics, step=epoch)

            if metrics["objective"] < best:
                best = metrics["objective"]
                if checkpoint:
                    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics}, checkpoint)

            print(
                f"epoch {epoch + 1}/{cfg['epochs']}  {time.time() - started:.0f}s  "
                f"train {metrics['train_loss']:.4f}  val objective {metrics['objective']:.4f}  "
                f"SSIM {metrics['ssim']:.3f}  PSNR {metrics['psnr']:.2f}"
            )

            if trial is not None:
                import optuna

                trial.report(metrics["objective"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        mlflow.log_metric("best_objective", best)
    return best


def main():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "ae_universal.pt"))
    parser.add_argument("--tracking_uri", default=None)
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()
    cfg = {key: getattr(args, key) for key in DEFAULTS}
    best = train(cfg, args.tracking_uri, args.checkpoint, run_name=args.run_name)
    print(f"best validation objective: {best:.4f}")


if __name__ == "__main__":
    main()
