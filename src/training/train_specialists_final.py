import argparse
import json
import shutil
from pathlib import Path

from src.training.train_ae import train

KINDS = ("salt_pepper", "blur", "occlusion")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    out = Path(args.storage_dir)
    best = json.loads((out / "task2_specialists_best.json").read_text())["params"]
    for kind in KINDS:
        cfg = {**best, "epochs": args.epochs, "condition": kind, "seed": 42}
        print(f"\n=== specialist: {kind} ===\n{cfg}")
        value = train(
            cfg,
            tracking_uri=f"sqlite:///{out / 'mlflow.db'}",
            checkpoint=f"/tmp/specialist_{kind}.pt",
            run_name=f"final_{kind}",
        )
        print(f"{kind}: best validation objective {value:.4f}")
        target = out / f"specialist_{kind}.pt"
        if target.exists():
            target.unlink()
        shutil.copyfile(f"/tmp/specialist_{kind}.pt", target)
        print(f"{kind}: saved to {target}")


if __name__ == "__main__":
    main()

