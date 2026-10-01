import argparse
import json
from pathlib import Path

import optuna

from src.training.train_cgan import train

STUDY_NAME = "task4_cgan"


def make_objective(epochs, tracking_uri):
    def objective(trial):
        cfg = {
            "lr_g": trial.suggest_float("lr_g", 5e-5, 1e-3, log=True),
            "lr_d": trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "base": trial.suggest_categorical("base", [32, 48, 64]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "embed_dim": trial.suggest_categorical("embed_dim", [8, 16, 32]),
            "lambda_l1": trial.suggest_float("lambda_l1", 10.0, 200.0, log=True),
            "epochs": epochs,
        }
        return train(cfg, tracking_uri=tracking_uri, trial=trial, run_name=f"trial_{trial.number}")

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--n_trials", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    out = Path(args.storage_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{out / 'optuna_task4.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=3),
    )
    finished = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(args.n_trials - finished, 0)
    print(f"{finished} trials already finished, running up to {remaining} more")
    study.optimize(make_objective(args.epochs, f"sqlite:///{out / 'mlflow.db'}"), n_trials=remaining, timeout=args.timeout)

    states = [t.state.name for t in study.trials]
    print({s: states.count(s) for s in sorted(set(states))})
    print("best objective:", study.best_value)
    print("best params:", study.best_params)
    (out / "task4_best.json").write_text(json.dumps({"value": study.best_value, "params": study.best_params}, indent=1))
    study.trials_dataframe().to_csv(out / "task4_trials.csv", index=False)


if __name__ == "__main__":
    main()
