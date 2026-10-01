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
from PIL import Image
from pytorch_msssim import ssim
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.cgan import PatchDiscriminator, UNetGenerator

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "fs2k_cache"
EXPERIMENT = "task4_face_sketch_cgan"

DEFAULTS = {
    "lr_g": 2e-4,
    "lr_d": 2e-4,
    "batch_size": 16,
    "base": 64,
    "dropout": 0.3,
    "embed_dim": 16,
    "lambda_l1": 100.0,
    "epochs": 30,
    "seed": 42,
    "num_workers": 2,
    "sample_every": 10,
}


def to_tensor(array):
    return torch.from_numpy(array).permute(2, 0, 1).float().div(127.5).sub(1.0)


class FS2KDataset(Dataset):
    """Returns (photo, sketch, style) in the range -1 to 1. Flips are applied to both images together."""

    def __init__(self, split, augment=False, cache_dir=CACHE):
        cache_dir = Path(cache_dir)
        self.photo = np.load(cache_dir / f"{split}_photo.npy")
        self.sketch = np.load(cache_dir / f"{split}_sketch.npy")
        self.style = np.load(cache_dir / f"{split}_style.npy")
        self.augment = augment

    def __len__(self):
        return len(self.style)

    def __getitem__(self, i):
        photo, sketch = to_tensor(self.photo[i]), to_tensor(self.sketch[i])
        if self.augment and random.random() < 0.5:
            photo, sketch = photo.flip(-1), sketch.flip(-1)
        return photo, sketch, int(self.style[i])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(g, loader, device):
    g.eval()
    l1s, ssims, psnrs = [], [], []
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        a, b = (g(photo, style) + 1) / 2, (sketch + 1) / 2
        l1s.append((a - b).abs().flatten(1).mean(1).cpu())
        ssims.append(ssim(a, b, data_range=1.0, size_average=False).cpu())
        mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
        psnrs.append((10 * torch.log10(1.0 / mse)).cpu())
    result = {
        "val_l1": torch.cat(l1s).mean().item(),
        "val_ssim": torch.cat(ssims).mean().item(),
        "val_psnr": torch.cat(psnrs).mean().item(),
    }
    result["objective"] = 0.5 * result["val_l1"] + 0.5 * (1.0 - result["val_ssim"])
    return result


@torch.no_grad()
def save_samples(g, dataset, indices, device, path):
    g.eval()
    items = [dataset[i] for i in indices]
    photo = torch.stack([it[0] for it in items]).to(device)
    sketch = torch.stack([it[1] for it in items])
    style = torch.tensor([it[2] for it in items]).to(device)
    fake = g(photo, style).cpu()

    def tile(t):
        return ((t.clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()

    size = photo.shape[-1]
    canvas = Image.new("RGB", (size * len(indices), size * 3))
    for c in range(len(indices)):
        for r, batch in enumerate((photo.cpu(), fake, sketch)):
            canvas.paste(Image.fromarray(tile(batch[c])), (c * size, r * size))
    canvas.save(path)


def train(config, tracking_uri=None, checkpoint=None, trial=None, run_name=None, sample_dir=None):
    cfg = {**DEFAULTS, **config}
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, val_ds = FS2KDataset("train", augment=True), FS2KDataset("val")
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=cfg["num_workers"])
    sample_idx = [int(np.flatnonzero(val_ds.style == s)[j]) for s in range(3) for j in range(2)]

    g = UNetGenerator(cfg["base"], cfg["embed_dim"], cfg["dropout"]).to(device)
    d = PatchDiscriminator(cfg["base"], cfg["embed_dim"]).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()

    mlflow.set_tracking_uri(tracking_uri or f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT)
    best = float("inf")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        for epoch in range(cfg["epochs"]):
            started = time.time()
            g.train()
            d.train()
            sums = {"train_d_real": 0.0, "train_d_fake": 0.0, "train_g_adv": 0.0, "train_g_rec": 0.0}
            batches = 0
            for photo, sketch, style in loader:
                photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
                fake = g(photo, style)

                opt_d.zero_grad()
                real_logits = d(photo, style, sketch)
                fake_logits = d(photo, style, fake.detach())
                d_real = bce(real_logits, torch.ones_like(real_logits))
                d_fake = bce(fake_logits, torch.zeros_like(fake_logits))
                (0.5 * (d_real + d_fake)).backward()
                opt_d.step()

                opt_g.zero_grad()
                g_logits = d(photo, style, fake)
                g_adv = bce(g_logits, torch.ones_like(g_logits))
                g_rec = l1(fake, sketch)
                (g_adv + cfg["lambda_l1"] * g_rec).backward()
                opt_g.step()

                for key, value in zip(sums, (d_real, d_fake, g_adv, g_rec)):
                    sums[key] += value.item()
                batches += 1

            metrics = {k: v / batches for k, v in sums.items()}
            metrics.update(evaluate(g, val_loader, device))
            mlflow.log_metrics(metrics, step=epoch)

            if metrics["objective"] < best:
                best = metrics["objective"]
                if checkpoint:
                    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"generator": g.state_dict(), "config": cfg, "epoch": epoch, "metrics": metrics}, checkpoint)

            if sample_dir and ((epoch + 1) % cfg["sample_every"] == 0 or epoch + 1 == cfg["epochs"]):
                Path(sample_dir).mkdir(parents=True, exist_ok=True)
                path = Path(sample_dir) / f"samples_epoch_{epoch + 1:03d}.png"
                save_samples(g, val_ds, sample_idx, device, path)
                mlflow.log_artifact(str(path), artifact_path="samples")

            print(
                f"epoch {epoch + 1}/{cfg['epochs']}  {time.time() - started:.0f}s  "
                f"D real {metrics['train_d_real']:.3f} fake {metrics['train_d_fake']:.3f}  "
                f"G adv {metrics['train_g_adv']:.3f} rec {metrics['train_g_rec']:.3f}  "
                f"val L1 {metrics['val_l1']:.4f} SSIM {metrics['val_ssim']:.3f} objective {metrics['objective']:.4f}"
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
    parser.add_argument("--storage_dir", default=None)
    parser.add_argument("--from_study", default=None)
    args = parser.parse_args()

    cfg = {key: getattr(args, key) for key in DEFAULTS}
    if args.from_study:
        storage = Path(args.from_study)
        cfg.update(json.loads((storage / "task4_best.json").read_text())["params"])
    else:
        storage = Path(args.storage_dir) if args.storage_dir else ROOT / "outputs" / "cgan"
    storage.mkdir(parents=True, exist_ok=True)
    print("configuration:", cfg)

    local = Path(tempfile.gettempdir()) / "cgan_best.pt"
    best = train(
        cfg,
        tracking_uri=f"sqlite:///{storage / 'mlflow.db'}",
        checkpoint=str(local),
        run_name="cgan_run",
        sample_dir=str(storage / "samples"),
    )
    target = storage / "cgan_final.pt"
    if target.exists():
        target.unlink()
    shutil.copyfile(local, target)
    saved = torch.load(target, map_location="cpu")
    print(f"best validation objective: {best:.4f}")
    print(f"checkpoint copied: epoch {saved['epoch'] + 1}, validation objective {saved['metrics']['objective']:.4f}")
    assert abs(saved["metrics"]["objective"] - best) < 1e-6, "copied checkpoint does not match the best model"


if __name__ == "__main__":
    main()
