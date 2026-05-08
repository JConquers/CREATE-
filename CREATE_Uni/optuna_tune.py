#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for CREATE-Uni validation mode.

This file keeps the single-run training code in train.py untouched. Each trial
calls train.main() with a generated set of command-line arguments and returns
the best validation NDCG@10 from that run.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch

from . import train as train_module


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning for CREATE-Uni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="beauty",
        choices=["beauty", "office_products", "ml1m", "movielens_1m"],
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs/optuna")
    parser.add_argument("--study_name", type=str, default="create_uni_val")
    parser.add_argument("--storage", type=str, default=None, help="Optional Optuna storage URL, e.g. sqlite:///optuna.db")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early_stopping_rounds", type=int, default=10)

    parser.add_argument("--seq_encoder", type=str, default="sasrec", choices=["sasrec", "bert4rec"])
    parser.add_argument("--graph_type", type=str, default="hypergraph", choices=["hypergraph", "bipartite"])
    parser.add_argument(
        "--graph_conv_type",
        type=str,
        default="UniGCN",
        choices=["UniGCN", "UniGIN", "UniSAGE", "UniGAT", "LightGCN"],
    )
    parser.add_argument("--max_sequence_length", type=int, default=50)
    parser.add_argument("--local_coef", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)

    return parser.parse_args()


def build_train_argv(args, trial) -> List[str]:
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    seq_head_choices = [h for h in [1, 2, 4, 8] if embedding_dim % h == 0]

    params: Dict[str, object] = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "output_dir": str(Path(args.output_dir) / args.study_name / f"trial_{trial.number:04d}"),
        "run_mode": "val",
        "seq_encoder": args.seq_encoder,
        "graph_type": args.graph_type,
        "graph_conv_type": args.graph_conv_type,
        "embedding_dim": embedding_dim,
        "graph_n_layers": trial.suggest_categorical("graph_n_layers", [1, 2, 3]),
        "seq_n_layers": trial.suggest_categorical("seq_n_layers", [1, 2, 3]),
        "seq_heads": trial.suggest_categorical("seq_heads", seq_head_choices),
        "max_sequence_length": args.max_sequence_length,
        "session_length": trial.suggest_categorical(
            "session_length",
            [86400, 604800, 2592000, 7776000],
        ),
        "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "batch_size": args.batch_size,
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3],
        ),
        "num_epochs": args.num_epochs,
        "early_stopping_rounds": args.early_stopping_rounds,
        "warmup_epochs": 0,
        "seed": args.seed + trial.number,
        "local_coef": args.local_coef,
        "global_coef": trial.suggest_categorical("global_coef", [0.1, 0.25, 0.4, 0.6, 1.0]),
        "barlow_twins_coef": trial.suggest_categorical(
            "barlow_twins_coef",
            [0.0, 0.05, 0.1, 0.2, 0.5],
        ),
        "barlow_lambda": trial.suggest_float("barlow_lambda", 0.0, 1.0),
        "num_workers": args.num_workers,
        "device": args.device,
        "log_interval": args.log_interval,
    }

    argv = ["CREATE_Uni.train"]
    for key, value in params.items():
        argv.extend([f"--{key}", str(value)])
    argv.extend(["--eval_k", "10"])
    return argv


def objective_factory(args):
    def objective(trial) -> float:
        old_argv = sys.argv[:]
        sys.argv = build_train_argv(args, trial)
        try:
            best_metrics = train_module.main()
            score = float(best_metrics.get("val/ndcg@10", 0.0))
            trial.set_user_attr("best_epoch", int(best_metrics.get("best_epoch", 0)))
            trial.set_user_attr("best_metrics", best_metrics)
            return score
        finally:
            sys.argv = old_argv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def main():
    args = parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. Install it with: pip install optuna"
        ) from exc

    output_dir = Path(args.output_dir) / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=args.storage is not None,
    )
    study.optimize(objective_factory(args), n_trials=args.n_trials, timeout=args.timeout)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_epoch": study.best_trial.user_attrs.get("best_epoch"),
        "best_metrics": study.best_trial.user_attrs.get("best_metrics"),
        "best_trial_number": study.best_trial.number,
    }
    with open(output_dir / "best_trial.json", "w") as f:
        json.dump(best, f, indent=2)

    study.trials_dataframe().to_csv(output_dir / "trials.csv", index=False)

    print("=" * 60)
    print(f"Best validation NDCG@10: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best epoch: {best['best_epoch']}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Saved Optuna results to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
