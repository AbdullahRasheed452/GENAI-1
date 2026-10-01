import argparse
import json
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.corruptions import CLASSES
from src.data.datasets import ManifestDataset, PetTrainDataset
from src.evaluation.classification_metrics import confusion_matrix, summarize
from src.models.classifier import CorruptionClassifier

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "task2_classifier"

DEFAULTS = {
    "channels": "medium",
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 10,
    "seed": 42,
    "num_workers": 2,
}


class BalancedBatchSampler:
    """Every batch holds the same number of images from each of the four classes."""

    def __init__(self, dataset, batch_size, seed):
        assert batch_size % len(CLASSES) == 0, "batch size must be divisible by 4"
        self.dataset = dataset
        self.per_class = batch_size // len(CLASSES)
        self.seed = seed

    def __len__(self):
        return (len(self.dataset) // len(CLASSES)) // self.per_class

    def __iter__(self):
        kinds = self.dataset.balanced_kinds
        rng = np.random.default_rng([self.seed, 7, self.dataset.epoch])
        pools = [rng.permutation(np.flatnonzero(kinds == k)) for k in range(len(CLASSES))]
        for b in range(len(self)):
            lo, hi = b * self.per_class, (b + 1) * self.per_class
            batch = np.concatenate([pool[lo:hi] for pool in pools])
            yield rng.permutation(batch).tolist()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for corrupted, _, label, _, _ in loader:
        logits_all.append(model(corrupted.to(device)).cpu())
        labels_all.append(label)
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    cm = confusion_matrix(labels.numpy(), logits.argmax(1).numpy())
    s = summarize(cm)
    result = {
        "val_ce": nn.functional.cross_entropy(logits, labels).item(),
        "accuracy": s["accuracy"],
        "macro_precision": s["macro_precision"],
        "macro_recall": s["macro_recall"],
        "macro_f1": s["macro_f1"],
    }
    for k, name in enumerate(CLASSES):
        result[f"{name}_f1"] = float(s["f1"][k])
    return result


def train(config, tracking_uri=None, checkpoint=None, trial=None, run_name=None):
    cfg = {**DEFAULTS, **config}
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = PetTrainDataset(condition="balanced", seed=cfg["seed"])
    sampler = BalancedBatchSampler(train_ds, cfg["batch_size"], cfg["seed"])
    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=cfg["num_workers"])
    val_loader = DataLoader(ManifestDataset("val"), batch_size=128, num_workers=cfg["num_workers"])

    model = CorruptionClassifier(cfg["channels"], cfg["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    best = -1.0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        for epoch in range(cfg["epochs"]):
            started = time.time()
            train_ds.set_epoch(epoch)
            model.train()
            total, batches = 0.0, 0
            for corrupted, _, labels in loader:
                corrupted, labels = corrupted.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(corrupted), labels)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            scheduler.step()

            metrics = evaluate(model, val_loader, device)
            metrics["train_loss"] = total / batches
            mlflow.log_metrics(metrics, step=epoch)

            if metrics["macro_f1"] > best:
                best = metrics["macro_f1"]
                if checkpoint:
                    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics}, checkpoint)

            print(
                f"epoch {epoch + 1}/{cfg['epochs']}  {time.time() - started:.0f}s  "
                f"train loss {metrics['train_loss']:.4f}  val CE {metrics['val_ce']:.4f}  "
                f"acc {metrics['accuracy']:.4f}  macro F1 {metrics['macro_f1']:.4f}"
            )

            if trial is not None:
                import optuna

                trial.report(metrics["macro_f1"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        mlflow.log_metric("best_macro_f1", best)
    return best


def main():
    parser = argparse.ArgumentParser()
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=type(value), default=value)
    parser.add_argument("--from_study", default=None)
    parser.add_argument("--checkpoint", default=str(ROOT / "outputs" / "classifier.pt"))
    args = parser.parse_args()

    if args.from_study:
        out = Path(args.from_study)
        params = json.loads((out / "task2_classifier_best.json").read_text())["params"]
        cfg = {**params, "epochs": args.epochs}
        print("final configuration:", cfg)
        best = train(cfg, f"sqlite:///{out / 'mlflow.db'}", str(out / "classifier_final.pt"), run_name="final_classifier")
    else:
        cfg = {key: getattr(args, key) for key in DEFAULTS}
        best = train(cfg, checkpoint=args.checkpoint, run_name="classifier_run")
    print(f"best validation macro F1: {best:.4f}")


if __name__ == "__main__":
    main()
