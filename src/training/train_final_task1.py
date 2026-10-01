import argparse
import json
from pathlib import Path

from src.training.train_ae import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    out = Path(args.storage_dir)
    best = json.loads((out / "task1_best.json").read_text())["params"]
    cfg = {**best, "epochs": args.epochs}
    print("final configuration:", cfg)
    value = train(
        cfg,
        tracking_uri=f"sqlite:///{out / 'mlflow.db'}",
        checkpoint=str(out / "ae_universal_final.pt"),
        run_name="final_training",
    )
    print(f"best validation objective: {value:.4f}")


if __name__ == "__main__":
    main()
