#!/usr/bin/env python3
"""Plot robustness metrics for CREATE vs CREATE-Uni."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare robustness results for CREATE and CREATE-Uni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--create_json", type=str, required=True, help="Path to CREATE json")
    parser.add_argument(
        "--create_uni_json",
        type=str,
        required=True,
        help="Path to CREATE-Uni json",
    )
    parser.add_argument("--num_epochs", type=int, required=True, help="Total epochs per run")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--edge_percents",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
        help="Edge removal percentages",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(data: Dict, key: str) -> List[float]:
    if key in data:
        return data[key]
    raise KeyError(f"Missing key '{key}' in {data}")


def get_time_list(data: Dict) -> List[float]:
    if "training_time" in data:
        return data["training_time"]
    if "train_time" in data:
        return data["train_time"]
    raise KeyError("Missing 'training_time' in json")


def to_avg_epoch_times(times: List[float], epochs: int) -> List[float]:
    if epochs <= 0:
        raise ValueError("num_epochs must be > 0")
    return [t / float(epochs) for t in times]


def plot_series(ax, x, y_create, y_uni, title, ylabel):
    ax.plot(x, y_uni, color="orange", marker="o", label="CREATE-Uni")
    ax.plot(x, y_create, color="blue", marker="o", label="CREATE")
    ax.set_title(title)
    ax.set_xlabel("% edges removed")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_data = load_json(args.create_json)
    create_uni_data = load_json(args.create_uni_json)

    x = args.edge_percents

    ndcg_create = get_metric(create_data, "ndcg@10")
    ndcg_uni = get_metric(create_uni_data, "ndcg@10")

    precision_create = get_metric(create_data, "precision@10")
    precision_uni = get_metric(create_uni_data, "precision@10")

    recall_create = get_metric(create_data, "recall@10")
    recall_uni = get_metric(create_uni_data, "recall@10")

    time_create = to_avg_epoch_times(get_time_list(create_data), args.num_epochs)
    time_uni = to_avg_epoch_times(get_time_list(create_uni_data), args.num_epochs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    plot_series(axes[0, 0], x, ndcg_create, ndcg_uni, "NDCG@10", "NDCG@10")
    plot_series(axes[0, 1], x, precision_create, precision_uni, "Precision@10", "Precision@10")
    plot_series(axes[1, 0], x, recall_create, recall_uni, "Recall@10", "Recall@10")
    plot_series(
        axes[1, 1],
        x,
        time_create,
        time_uni,
        "Avg Time per Epoch",
        "Seconds",
    )

    fig.suptitle("Robustness Comparison", y=0.98)
    fig.tight_layout()

    out_path = output_dir / "robustness_compare.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
