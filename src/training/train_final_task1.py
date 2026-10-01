import argparse
import json
import shutil
from pathlib import Path

import torch

from src.training.train_ae import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    out = Path(args.storage_dir)
    best = json.loads((out / "task1_best.json").read_text())["params"]
    cfg = {**best, "epochs": args.epochs}
    print("final configuration:", cfg)

    local = Path("/tmp/ae_universal_final.pt")
    value = train(
        cfg,
        tracking_uri=f"sqlite:///{out / 'mlflow.db'}",
        checkpoint=str(local),
        run_name="final_training",
    )
    print(f"best validation objective: {value:.4f}")

    target = out / "ae_universal_final.pt"
    if target.exists():
        target.unlink()
    shutil.copyfile(local, target)

    saved = torch.load(target, map_location="cpu")
    print(f"checkpoint on Drive: epoch {saved['epoch'] + 1}, validation objective {saved['metrics']['objective']:.4f}")
    assert abs(saved["metrics"]["objective"] - value) < 1e-6, "Drive copy does not match the best model"


if __name__ == "__main__":
    main()
