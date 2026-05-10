#!/usr/bin/env python3
"""Standalone dataset statistics for CREATE/CREATE-Uni experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class SplitPaths:
    def __init__(self, train: Path, validation: Path, test: Path, train_validation: Path | None = None):
        self.train = train
        self.validation = validation
        self.test = test
        self.train_validation = train_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics for CREATE/CREATE-Uni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset name; can be repeated",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--session_length", type=int, default=86400)
    return parser.parse_args()


def normalize_datasets(values: List[str]) -> List[str]:
    datasets: List[str] = []
    for value in values:
        parts = [v.strip() for v in value.split(",") if v.strip()]
        datasets.extend(parts if parts else [value])
    return datasets


def _ensure_train_val(train_path: Path, val_path: Path, train_val_path: Path) -> Path:
    if train_val_path.exists():
        return train_val_path
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_df.to_csv(train_val_path, index=False)
    return train_val_path


def resolve_split_paths(dataset: str, data_dir: str, output_dir: Path) -> SplitPaths:
    data_dir_path = Path(data_dir)
    dataset_dir = data_dir_path / dataset

    train_path = dataset_dir / "train.csv"
    val_path = dataset_dir / "validation.csv"
    test_path = dataset_dir / "test.csv"
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_val_path = dataset_dir / "train_validation.csv"
        return SplitPaths(
            train=train_path,
            validation=val_path,
            test=test_path,
            train_validation=train_val_path if train_val_path.exists() else None,
        )

    global_split_dir = data_dir_path / "global_split" / dataset
    train_path = global_split_dir / "train.csv"
    val_path = global_split_dir / "validation.csv"
    test_path = global_split_dir / "test.csv"
    if train_path.exists() and val_path.exists() and test_path.exists():
        train_val_path = global_split_dir / "train_validation.csv"
        return SplitPaths(
            train=train_path,
            validation=val_path,
            test=test_path,
            train_validation=train_val_path if train_val_path.exists() else None,
        )

    csv_path = data_dir_path / f"{dataset}.csv"
    if csv_path.exists():
        split_dir = data_dir_path / "global_split" / dataset
        train_path = split_dir / "train.csv"
        val_path = split_dir / "validation.csv"
        test_path = split_dir / "test.csv"
        train_val_path = split_dir / "train_validation.csv"
        return SplitPaths(
            train=train_path,
            validation=val_path,
            test=test_path,
            train_validation=train_val_path if train_val_path.exists() else None,
        )

    # Dataset loaders fallback
    dataset_loaders_dir = Path(__file__).resolve().parent / "dataset_loaders"
    if dataset_loaders_dir.exists() and dataset in {"beauty", "office_products", "ml1m", "movielens_1m"}:
        import sys

        sys.path.insert(0, str(dataset_loaders_dir.parent))
        if dataset == "beauty":
            from dataset_loaders.beauty_dataset import BeautyDataset

            loader = BeautyDataset(root=str(dataset_dir))
        elif dataset == "office_products":
            from dataset_loaders.office_products_dataset import OfficeProductsDataset

            loader = OfficeProductsDataset(root=str(dataset_dir))
        else:
            from dataset_loaders.movielens_1m_dataset import MovieLens1MDataset

            loader = MovieLens1MDataset(root=str(dataset_dir))

        data = loader.load()

        train_df = pd.DataFrame({
            "user_id": data["train_user"].cpu().numpy(),
            "item_id": data["train_item"].cpu().numpy(),
            "timestamp": data["train_time"].cpu().numpy() if "train_time" in data else None,
        })
        val_target_df = pd.DataFrame({
            "user_id": data["val_user"].cpu().numpy(),
            "item_id": data["val_item"].cpu().numpy(),
            "timestamp": data["val_time"].cpu().numpy() if "val_time" in data else None,
        })
        train_val_df = (
            pd.concat([train_df, val_target_df], ignore_index=True)
            .sort_values(["user_id", "timestamp"])
            .reset_index(drop=True)
        )

        train_history = {}
        for u, it, t in zip(
            data["train_user"].cpu().numpy(),
            data["train_item"].cpu().numpy(),
            data["train_time"].cpu().numpy() if "train_time" in data else [0] * len(data["train_user"]),
        ):
            train_history.setdefault(int(u), []).append((int(it), float(t)))
        for u in train_history:
            train_history[u].sort(key=lambda x: x[1])

        val_rows = {"user_id": [], "item_id": [], "timestamp": []}
        for u, it, t in zip(
            data["val_user"].cpu().numpy(),
            data["val_item"].cpu().numpy(),
            data["val_time"].cpu().numpy() if "val_time" in data else [0] * len(data["val_user"]),
        ):
            u = int(u)
            for hist_item, hist_t in train_history.get(u, []):
                val_rows["user_id"].append(u)
                val_rows["item_id"].append(hist_item)
                val_rows["timestamp"].append(hist_t)
            val_rows["user_id"].append(u)
            val_rows["item_id"].append(int(it))
            val_rows["timestamp"].append(float(t))
        val_df = pd.DataFrame(val_rows)

        val_lookup = {}
        for u, it, t in zip(
            data["val_user"].cpu().numpy(),
            data["val_item"].cpu().numpy(),
            data["val_time"].cpu().numpy() if "val_time" in data else [0] * len(data["val_user"]),
        ):
            val_lookup[int(u)] = (int(it), float(t))

        test_rows = {"user_id": [], "item_id": [], "timestamp": []}
        for u, it, t in zip(
            data["test_user"].cpu().numpy(),
            data["test_item"].cpu().numpy(),
            data["test_time"].cpu().numpy() if "test_time" in data else [0] * len(data["test_user"]),
        ):
            u = int(u)
            for hist_item, hist_t in train_history.get(u, []):
                test_rows["user_id"].append(u)
                test_rows["item_id"].append(hist_item)
                test_rows["timestamp"].append(hist_t)
            if u in val_lookup:
                test_rows["user_id"].append(u)
                test_rows["item_id"].append(val_lookup[u][0])
                test_rows["timestamp"].append(val_lookup[u][1])
            test_rows["user_id"].append(u)
            test_rows["item_id"].append(int(it))
            test_rows["timestamp"].append(float(t))
        test_df = pd.DataFrame(test_rows)

        temp_dir = output_dir / "temp_data"
        temp_dir.mkdir(parents=True, exist_ok=True)

        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "validation.csv"
        test_path = temp_dir / "test.csv"
        train_val_path = temp_dir / "train_validation.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        train_val_df.to_csv(train_val_path, index=False)

        return SplitPaths(
            train=train_path,
            validation=val_path,
            test=test_path,
            train_validation=train_val_path,
        )

    raise FileNotFoundError(
        f"Could not resolve dataset paths for '{dataset}' in {data_dir}. "
        "Expected train/validation/test CSVs or global_split folder."
    )


def gini(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = sorted_vals.size
    cum = np.cumsum(sorted_vals)
    if cum[-1] == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * sorted_vals).sum()) / (n * cum[-1]) - (n + 1) / n


def compute_sequence_stats(counts: pd.Series) -> Dict[str, float]:
    values = counts.to_numpy()
    percentiles = np.percentile(values, [25, 50, 75, 90]) if values.size else [0, 0, 0, 0]
    return {
        "mean": float(np.mean(values)) if values.size else 0.0,
        "median": float(percentiles[1]),
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
        "p25": float(percentiles[0]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
    }


def compute_temporal_stats(df: pd.DataFrame) -> Dict[str, float]:
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return {}

    ts = df["timestamp"].fillna(0).astype(float)
    t_min = float(ts.min())
    t_max = float(ts.max())
    span_sec = max(0.0, t_max - t_min)

    median_deltas = []
    overall_deltas = []
    for _, user_df in df.sort_values(["user_id", "timestamp"]).groupby("user_id"):
        times = user_df["timestamp"].to_numpy(dtype=float)
        if len(times) < 2:
            continue
        diffs = np.diff(times)
        if diffs.size:
            median_deltas.append(float(np.median(diffs)))
            overall_deltas.extend(diffs.tolist())

    return {
        "time_min": t_min,
        "time_max": t_max,
        "time_span_days": span_sec / 86400.0,
        "median_inter_event_sec": float(np.median(median_deltas)) if median_deltas else 0.0,
        "overall_median_inter_event_sec": float(np.median(overall_deltas)) if overall_deltas else 0.0,
    }


def compute_session_stats(df: pd.DataFrame, session_length: int) -> Dict[str, float]:
    if session_length <= 0:
        return {}
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        return {}

    session_sizes = []
    sessions_per_user = []

    for _, user_df in df.sort_values(["user_id", "timestamp"]).groupby("user_id"):
        times = user_df["timestamp"].fillna(0).to_numpy(dtype=float)
        if times.size == 0:
            continue
        anchor = times[0]
        session_ids = ((times - anchor) // session_length).astype(int)
        unique_sessions = np.unique(session_ids)
        sessions_per_user.append(float(unique_sessions.size))
        for sid in unique_sessions:
            session_sizes.append(float((session_ids == sid).sum()))

    if not session_sizes:
        return {}

    percentiles = np.percentile(session_sizes, [50, 90])

    return {
        "avg_session_size": float(np.mean(session_sizes)),
        "median_session_size": float(percentiles[0]),
        "p90_session_size": float(percentiles[1]),
        "avg_sessions_per_user": float(np.mean(sessions_per_user)) if sessions_per_user else 0.0,
    }


def compute_stats(df: pd.DataFrame, session_length: int) -> Dict:
    stats: Dict = {}

    n_users = int(df["user_id"].nunique()) if not df.empty else 0
    n_items = int(df["item_id"].nunique()) if not df.empty else 0
    n_interactions = int(len(df))

    density = (n_interactions / (n_users * n_items)) if n_users > 0 and n_items > 0 else 0.0

    stats["basic"] = {
        "num_users": n_users,
        "num_items": n_items,
        "num_interactions": n_interactions,
        "density": float(density),
        "sparsity": float(1.0 - density),
    }

    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("item_id").size()

    stats["sequence_length"] = compute_sequence_stats(user_counts)
    stats["degree_distribution"] = {
        "user_gini": float(gini(user_counts.to_numpy())) if not user_counts.empty else 0.0,
        "item_gini": float(gini(item_counts.to_numpy())) if not item_counts.empty else 0.0,
    }

    # Popularity bias: top 1% items share of interactions
    if not item_counts.empty:
        top_k = max(1, int(np.ceil(0.01 * n_items)))
        top_share = float(item_counts.sort_values(ascending=False).head(top_k).sum() / n_interactions)
    else:
        top_share = 0.0
    stats["popularity"] = {
        "top_1pct_item_share": top_share,
    }

    # Cold-start slices
    stats["cold_start"] = {
        "pct_users_le_2": float((user_counts <= 2).mean()) if not user_counts.empty else 0.0,
        "pct_items_le_2": float((item_counts <= 2).mean()) if not item_counts.empty else 0.0,
    }

    # Repeat rates
    if not df.empty:
        unique_per_user = df.groupby("user_id")["item_id"].nunique()
        repeat_rate_user = 1.0 - (unique_per_user / user_counts)
        repeat_rate_user = repeat_rate_user.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        stats["repeat"] = {
            "avg_user_repeat_rate": float(repeat_rate_user.mean()),
            "repeat_interaction_share": float((n_interactions - unique_per_user.sum()) / n_interactions),
        }
    else:
        stats["repeat"] = {"avg_user_repeat_rate": 0.0, "repeat_interaction_share": 0.0}

    stats["temporal"] = compute_temporal_stats(df)
    stats["session"] = compute_session_stats(df, session_length)

    return stats


def load_all_splits(dataset: str, data_dir: str, output_dir: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    split_paths = resolve_split_paths(dataset, data_dir, output_dir)
    train_df = pd.read_csv(split_paths.train)
    val_df = pd.read_csv(split_paths.validation)
    test_df = pd.read_csv(split_paths.test)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_sizes = {
        "train": int(len(train_df)),
        "validation": int(len(val_df)),
        "test": int(len(test_df)),
    }
    return all_df, split_sizes


def main() -> None:
    args = parse_args()
    datasets = normalize_datasets(args.dataset)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in datasets:
        all_df, split_sizes = load_all_splits(dataset, args.data_dir, output_dir)
        stats = compute_stats(all_df, args.session_length)
        stats["splits"] = split_sizes
        summary[dataset] = stats

        out_path = output_dir / f"dataset_stats_{dataset}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved stats for {dataset} to {out_path}")

    summary_path = output_dir / "dataset_stats_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
