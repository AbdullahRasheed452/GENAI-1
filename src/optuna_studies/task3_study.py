import argparse
import json
from pathlib import Path

import optuna

from src.training.train_moe import train

STUDY_NAME = "task3_moe"


def make_objective(task2_dir, epochs, tracking_uri):
    def objective(trial):
        cfg = {
            "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
            "temperature": trial.suggest_categorical("temperature", [0.5, 1.0, 2.0]),
            "lambda_ce": trial.suggest_float("lambda_ce", 0.01, 1.0, log=True),
            "lambda_bal": trial.suggest_float("lambda_bal", 1e-3, 0.1, log=True),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
            "warmup_epochs": 1,
            "epochs": epochs,
            "task2_dir": task2_dir,
        }
        return train(cfg, tracking_uri=tracking_uri, trial=trial, run_name=f"trial_{trial.number}")

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task2_dir", required=True)
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--n_trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    out = Path(args.storage_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{out / 'optuna_task3.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2),
    )
    finished = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(args.n_trials - finished, 0)
    print(f"{finished} trials already finished, running up to {remaining} more")
    study.optimize(
        make_objective(args.task2_dir, args.epochs, f"sqlite:///{out / 'mlflow.db'}"),
        n_trials=remaining,
        timeout=args.timeout,
    )

    states = [t.state.name for t in study.trials]
    print({s: states.count(s) for s in sorted(set(states))})
    print("best objective:", study.best_value)
    print("best params:", study.best_params)
    (out / "task3_best.json").write_text(json.dumps({"value": study.best_value, "params": study.best_params}, indent=1))
    study.trials_dataframe().to_csv(out / "task3_trials.csv", index=False)


if __name__ == "__main__":
    main()
