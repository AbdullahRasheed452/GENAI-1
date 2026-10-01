import argparse
import json
from pathlib import Path

import optuna

from src.training.train_ae import train

KINDS = ("salt_pepper", "blur", "occlusion")
STUDY_NAME = "task2_specialists"


def make_objective(epochs, tracking_uri):
    def objective(trial):
        shared = {
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "latent_dim": trial.suggest_categorical("latent_dim", [256, 512, 1024]),
            "base": trial.suggest_categorical("base", [32, 48, 64]),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95),
            "epochs": epochs,
        }
        scores = []
        for kind in KINDS:
            score = train({**shared, "condition": kind}, tracking_uri=tracking_uri, run_name=f"trial_{trial.number}_{kind}")
            scores.append(score)
            trial.report(sum(scores) / len(scores), len(scores) - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return sum(scores) / len(scores)

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_dir", required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    out = Path(args.storage_dir)
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{out / 'optuna_task2_specialists.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
    )
    finished = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(args.n_trials - finished, 0)
    print(f"{finished} trials already finished, running up to {remaining} more")
    study.optimize(make_objective(args.epochs, f"sqlite:///{out / 'mlflow.db'}"), n_trials=remaining, timeout=args.timeout)

    states = [t.state.name for t in study.trials]
    print({s: states.count(s) for s in sorted(set(states))})
    print("best objective:", study.best_value)
    print("best params:", study.best_params)
    (out / "task2_specialists_best.json").write_text(json.dumps({"value": study.best_value, "params": study.best_params}, indent=1))
    study.trials_dataframe().to_csv(out / "task2_specialists_trials.csv", index=False)


if __name__ == "__main__":
    main()
