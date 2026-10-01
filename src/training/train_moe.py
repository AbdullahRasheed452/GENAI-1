import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from pytorch_msssim import ssim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset, PetTrainDataset
from src.evaluation.metrics import psnr, ssim_per_image
from src.models.moe import build_moe
from src.training.train_classifier import BalancedBatchSampler

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "task3_soft_moe"

DEFAULTS = {
    "lr": 5e-5,
    "warm_lr": 1e-4,
    "temperature": 1.0,
    "alpha": 0.8,
    "lambda_ce": 0.1,
    "lambda_bal": 0.01,
    "warmup_epochs": 2,
    "epochs": 15,
    "batch_size": 16,
    "seed": 42,
    "num_workers": 2,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    l1s, ssims, psnrs, weights, labels = [], [], [], [], []
    for corrupted, clean, label, _, _ in loader:
        corrupted, clean = corrupted.to(device), clean.to(device)
        y, w = model(corrupted)
        l1s.append((y - clean).abs().flatten(1).mean(1).cpu())
        ssims.append(ssim_per_image(y, clean).cpu())
        psnrs.append(psnr(y, clean).cpu())
        weights.append(w.cpu())
        labels.append(label)
    l1, s, p, w, lab = (torch.cat(v) for v in (l1s, ssims, psnrs, weights, labels))
    branch_mean = w.mean(0)
    result = {
        "l1": l1.mean().item(),
        "ssim": s.mean().item(),
        "psnr": p.mean().item(),
        "gate_acc": (w.argmax(1) == lab).float().mean().item(),
    }
    for j, name in enumerate(CLASSES):
        result[f"mean_weight_{name}"] = branch_mean[j].item()
        result[f"weight_on_true_{name}"] = w[lab == j, j].mean().item()
    result["objective"] = 0.5 * result["l1"] + 0.5 * (1.0 - result["ssim"])
    result["collapsed"] = float(branch_mean.max() > 0.6 or branch_mean.min() < 0.05)
    return result


def make_optimizer(model, joint, cfg):
    for p in model.experts.parameters():
        p.requires_grad = joint
    params = model.parameters() if joint else model.gate.parameters()
    return torch.optim.Adam(params, lr=cfg["lr"] if joint else cfg["warm_lr"])


def weights_text(m):
    return "[" + " ".join(f"{m[f'mean_weight_{name}']:.2f}" for name in CLASSES) + "]"


def train(config, tracking_uri=None, checkpoint=None, trial=None, run_name=None):
    cfg = {**DEFAULTS, **config}
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_moe(cfg["task2_dir"], cfg["temperature"], device)
    train_ds = PetTrainDataset(condition="balanced", seed=cfg["seed"])
    sampler = BalancedBatchSampler(train_ds, cfg["batch_size"], cfg["seed"])
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg["num_workers"])
    val_loader = DataLoader(ManifestDataset("val"), batch_size=128, num_workers=cfg["num_workers"])

    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    best = float("inf")
    total = cfg["warmup_epochs"] + cfg["epochs"]

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        initial = evaluate(model, val_loader, device)
        mlflow.log_metrics({f"init_{k}": v for k, v in initial.items()})
        print(
            f"initial (no training)  val objective {initial['objective']:.4f}  SSIM {initial['ssim']:.3f}  "
            f"gate acc {initial['gate_acc']:.3f}  mean weights {weights_text(initial)}"
        )

        phase, optimizer, scheduler = None, None, None
        for epoch in range(total):
            started = time.time()
            joint = epoch >= cfg["warmup_epochs"]
            if joint != phase:
                optimizer = make_optimizer(model, joint, cfg)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if joint else None
                phase = joint

            train_ds.set_epoch(epoch)
            model.train()
            for expert in model.experts:
                expert.eval()
            totals, batches = 0.0, 0
            for corrupted, clean, labels in loader:
                corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)
                optimizer.zero_grad()
                y, w, logits = model(corrupted, return_all=True)
                rec = F.l1_loss(y, clean)
                structure = 1.0 - ssim(y, clean, data_range=1.0, size_average=True)
                ce = F.cross_entropy(logits / cfg["temperature"], labels)
                balance = ((w.mean(0) - 0.25) ** 2).sum()
                loss = cfg["alpha"] * rec + (1.0 - cfg["alpha"]) * structure + cfg["lambda_ce"] * ce + cfg["lambda_bal"] * balance
                loss.backward()
                optimizer.step()
                totals += loss.item()
                batches += 1
            if scheduler is not None:
                scheduler.step()

            if joint:
                assert any(p.grad is not None for p in model.experts.parameters()), "experts got no gradients in joint training"
            else:
                assert all(p.grad is None for p in model.experts.parameters()), "experts must stay frozen during warm-up"

            metrics = evaluate(model, val_loader, device)
            metrics["train_loss"] = totals / batches
            mlflow.log_metrics(metrics, step=epoch)

            if joint and metrics["objective"] < best:
                best = metrics["objective"]
                if checkpoint:
                    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics,
                         "gate_cfg": model.gate_cfg, "expert_cfgs": model.expert_cfgs},
                        checkpoint,
                    )

            print(
                f"{'joint' if joint else 'warm '} epoch {epoch + 1}/{total}  {time.time() - started:.0f}s  "
                f"train {metrics['train_loss']:.4f}  val objective {metrics['objective']:.4f}  SSIM {metrics['ssim']:.3f}  "
                f"gate acc {metrics['gate_acc']:.3f}  mean weights {weights_text(metrics)}"
            )

            if trial is not None and joint:
                import optuna

                trial.report(metrics["objective"], epoch)
                if metrics["collapsed"] or trial.should_prune():
                    raise optuna.TrialPruned()
        mlflow.log_metric("best_objective", best)
    return best


def main():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parser.add_argument("--task2_dir", required=True)
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--from_study", action="store_true")
    args = parser.parse_args()

    cfg = {key: getattr(args, key) for key in DEFAULTS}
    cfg["task2_dir"] = args.task2_dir
    storage = Path(args.storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    if args.from_study:
        cfg.update(json.loads((storage / "task3_best.json").read_text())["params"])
    print("configuration:", cfg)

    local = Path(tempfile.gettempdir()) / "moe_best.pt"
    best = train(cfg, tracking_uri=f"sqlite:///{storage / 'mlflow.db'}", checkpoint=str(local), run_name="moe_run")
    target = storage / "moe_final.pt"
    if target.exists():
        target.unlink()
    shutil.copyfile(local, target)
    saved = torch.load(target, map_location="cpu")
    print(f"best validation objective: {best:.4f}")
    print(f"checkpoint copied: epoch {saved['epoch'] + 1}, validation objective {saved['metrics']['objective']:.4f}")
    assert abs(saved["metrics"]["objective"] - best) < 1e-6, "copied checkpoint does not match the best model"


if __name__ == "__main__":
    main()
