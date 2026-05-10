#!/usr/bin/env python3
"""
Standalone robustness experiment for graph recommendation models.
"""

from __future__ import annotations

import argparse
import json
import time
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

# Ensure CREATE (src) modules are importable
ROOT = Path(__file__).resolve().parent
CREATE_ROOT = ROOT / "CREATE"
SRC_ROOT = ROOT / "src"
if CREATE_ROOT.exists() and str(CREATE_ROOT) not in sys.path:
    sys.path.insert(0, str(CREATE_ROOT))
if SRC_ROOT.exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_create_modules():
    try:
        from src.dataset import get_dataloaders, get_sparse_graph_layer, _convert_sp_mat_to_sp_tensor
        from src.loss import UnderDogLoss, LocalObjective, GlobalObjective
        from src.metrics import NDCGMetric, RecallMetric
        from src.models.underdog import UnderDogModel
        from src.optimizer import BasicOptimizer
        from src.utils import train as create_train
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import CREATE 'src' package. "
            "Make sure CREATE/src exists for running model=CREATE."
        ) from exc

    return {
        "get_dataloaders": get_dataloaders,
        "get_sparse_graph_layer": get_sparse_graph_layer,
        "_convert_sp_mat_to_sp_tensor": _convert_sp_mat_to_sp_tensor,
        "UnderDogLoss": UnderDogLoss,
        "LocalObjective": LocalObjective,
        "GlobalObjective": GlobalObjective,
        "NDCGMetric": NDCGMetric,
        "RecallMetric": RecallMetric,
        "UnderDogModel": UnderDogModel,
        "BasicOptimizer": BasicOptimizer,
        "create_train": create_train,
    }

# CREATE-Uni imports
from CREATE_Uni.data import create_dataloaders as create_uni_dataloaders
from CREATE_Uni.loss import CREATEUniLoss
from CREATE_Uni.metrics import create_metrics
from CREATE_Uni.models import CREATEUni
from CREATE_Uni.utils import inference as create_uni_inference
from CREATE_Uni.utils import get_bipartite_graph_structure, get_graph_structure
from CREATE_Uni.utils import train as create_uni_train


@dataclass
class SplitPaths:
    train: Path
    validation: Path
    test: Path
    train_validation: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness analysis for CREATE / CREATE-Uni",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required hyperparameters
    parser.add_argument("--model", type=str, required=True, choices=["CREATE", "CREATEUni"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_mode", type=str, required=True, choices=["val", "test"])
    parser.add_argument("--embedding_dim", type=int, required=True)
    parser.add_argument("--graph_n_layers", type=int, required=True)
    parser.add_argument("--seq_n_layers", type=int, required=True)
    parser.add_argument("--seq_heads", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--session_length", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--global_coef", type=float, required=True)
    parser.add_argument("--barlow_twins_coef", type=float, required=True)
    parser.add_argument("--barlow_lambda", type=float, required=True)
    parser.add_argument("--warmup_epochs", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--graph_conv_type", type=str, required=True)
    parser.add_argument("--graph_type", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)

    # Optional args
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_sequence_length", type=int, default=50)
    parser.add_argument("--early_stopping_rounds", type=int, default=10)
    parser.add_argument("--local_coef", type=float, default=1.0)
    parser.add_argument("--seq_encoder", type=str, default="sasrec", choices=["sasrec", "bert4rec"])
    parser.add_argument("--graph_heads", type=int, default=1)

    return parser.parse_args()


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_args(args: argparse.Namespace) -> None:
    if args.model == "CREATE":
        if args.graph_conv_type != "LightGCN":
            raise ValueError("CREATE supports only --graph_conv_type LightGCN")
        if args.graph_type != "bipartite":
            raise ValueError("CREATE supports only --graph_type bipartite")
    if args.graph_conv_type == "LightGCN" and args.graph_type != "bipartite":
        raise ValueError("LightGCN requires bipartite graph_type")


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

    # CREATE-Uni dataset loaders fallback
    dataset_loaders_dir = ROOT / "dataset_loaders"
    if dataset_loaders_dir.exists() and dataset in {"beauty", "office_products", "ml1m", "movielens_1m"}:
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


def build_edge_removal_order(num_edges: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.permutation(num_edges)


def build_create_graph_from_edges(
    edges: List[Tuple[int, int]],
    num_users: int,
    num_items: int,
    device: torch.device,
) -> torch.Tensor:
    create_modules = import_create_modules()
    get_sparse_graph_layer = create_modules["get_sparse_graph_layer"]
    convert_sp = create_modules["_convert_sp_mat_to_sp_tensor"]
    if not edges:
        user_ids = np.array([], dtype=np.int64)
        item_ids = np.array([], dtype=np.int64)
    else:
        user_ids = np.array([e[0] for e in edges], dtype=np.int64)
        item_ids = np.array([e[1] for e in edges], dtype=np.int64)

    import scipy.sparse as sp

    user2item = sp.csr_matrix(
        (
            np.ones(len(user_ids), dtype=np.float32),
            (user_ids, item_ids),
        ),
        shape=(num_users + 2, num_items + 2),
    )
    graph = get_sparse_graph_layer(user2item, num_users + 2, num_items + 2, biparite=True)
    return convert_sp(graph).coalesce().to(device)


def compute_create_metrics(
    dataloader,
    model: torch.nn.Module,
    device: torch.device,
    k: int = 10,
) -> Dict[str, float]:
    model.eval()
    ndcgs = []
    precisions = []
    recalls = []

    with torch.no_grad():
        for batch in dataloader:
            for key, value in batch.items():
                batch[key] = value.to(device)
            preds = model(batch)
            labels = batch["labels.ids"].to(device)

            hits = (preds[:, :k] == labels.unsqueeze(-1)).float()
            recall = hits.sum(dim=1)
            precision = recall / float(k)
            discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=device).float())
            ndcg = (hits * discounts).sum(dim=1)

            ndcgs.extend(ndcg.cpu().tolist())
            precisions.extend(precision.cpu().tolist())
            recalls.extend(recall.cpu().tolist())

    model.train()
    return {
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "precision@10": float(np.mean(precisions)) if precisions else 0.0,
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
    }


def prepare_create_dataloaders(
    args: argparse.Namespace,
    split_paths: SplitPaths,
    run_mode: str,
) -> Tuple[Dict, Dict, Dict]:
    create_modules = import_create_modules()
    get_dataloaders = create_modules["get_dataloaders"]
    if run_mode == "test":
        train_val_path = split_paths.train_validation or (
            split_paths.train.parent / "train_validation.csv"
        )
        train_val_path = _ensure_train_val(split_paths.train, split_paths.validation, train_val_path)
        active_train = train_val_path
        active_val = split_paths.test
        active_test = split_paths.test
    else:
        active_train = split_paths.train
        active_val = split_paths.validation
        active_test = split_paths.validation

    train_df = pd.read_csv(active_train)
    val_df = pd.read_csv(active_val)
    test_df = pd.read_csv(active_test)

    num_users = int(pd.concat([train_df, val_df, test_df])["user_id"].max())
    num_items = int(pd.concat([train_df, val_df, test_df])["item_id"].max())
    dataset_meta = {
        "num_users": num_users,
        "num_items": num_items,
        "max_sequence_length": args.max_sequence_length,
    }

    config = {
        "model_name": "underdog",
        "dataset": {
            "path_to_data_dir": args.data_dir,
            "name": args.dataset,
            "max_sequence_length": args.max_sequence_length,
            "mlm_prob": 0.15,
            "last_mask_prob": 0.0,
        },
        "model": {
            "embedding_dim": args.embedding_dim,
            "num_heads": args.seq_heads,
            "num_layers": args.seq_n_layers,
            "dim_feedforward": args.embedding_dim * 4,
            "dropout": args.dropout,
            "num_hops": args.graph_n_layers,
            "activation": "gelu",
            "layer_norm_eps": 1.0e-9,
            "initializer_range": 0.02,
            "topk_k": 10,
            "seq_encoder": args.seq_encoder,
        },
        "dataloader": {
            "train": {"batch_size": args.batch_size, "drop_last": False, "shuffle": True, "num_workers": 0},
            "validation": {"batch_size": args.batch_size, "drop_last": False, "shuffle": False, "num_workers": 0},
            "test": {"batch_size": args.batch_size, "drop_last": False, "shuffle": False, "num_workers": 0},
        },
        "num_epochs": args.num_epochs,
        "early_stopping_rounds": args.early_stopping_rounds,
        "device": args.device,
    }

    dataloaders = get_dataloaders(config, dataset_meta, active_train, active_val, active_test)

    return dataloaders, dataset_meta, {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "active_train": pd.read_csv(active_train),
    }


def run_create_robustness(args: argparse.Namespace, output_dir: Path) -> Dict[str, List[float]]:
    create_modules = import_create_modules()
    UnderDogModel = create_modules["UnderDogModel"]
    UnderDogLoss = create_modules["UnderDogLoss"]
    LocalObjective = create_modules["LocalObjective"]
    GlobalObjective = create_modules["GlobalObjective"]
    NDCGMetric = create_modules["NDCGMetric"]
    RecallMetric = create_modules["RecallMetric"]
    BasicOptimizer = create_modules["BasicOptimizer"]
    create_train = create_modules["create_train"]
    device = torch.device(args.device)
    split_paths = resolve_split_paths(args.dataset, args.data_dir, output_dir)
    dataloaders, dataset_meta, data_frames = prepare_create_dataloaders(args, split_paths, args.run_mode)

    train_df = data_frames["active_train"]
    seen = set()
    edges = []
    for row in train_df.itertuples(index=False):
        key = (int(row.user_id), int(row.item_id))
        if key not in seen:
            seen.add(key)
            edges.append(key)

    removal_order = build_edge_removal_order(len(edges), args.seed)

    metrics_overall = {"ndcg@10": [], "precision@10": [], "recall@10": []}
    time_log_path = output_dir / "robust_time_output.txt"
    time_log_lines = [
        "model=CREATE (routed to CREATEUni LightGCN/bipartite)",
        f"dataset={args.dataset}",
        f"run_mode={args.run_mode}",
    ]

    total_edges = len(edges)
    for pct in [10, 20, 30, 40, 50]:
        remove_count = int(len(edges) * pct / 100)
        removed_idx = set(removal_order[:remove_count])
        kept_edges = [edge for idx, edge in enumerate(edges) if idx not in removed_idx]
        kept_count = len(kept_edges)

        print(
            "Running robustness step: "
            f"removed={pct}% | original_edges={total_edges} | training_edges={kept_count}"
        )

        graph = build_create_graph_from_edges(
            kept_edges,
            num_users=dataset_meta["num_users"],
            num_items=dataset_meta["num_items"],
            device=device,
        )

        model = UnderDogModel(
            cfg={
                "embedding_dim": args.embedding_dim,
                "num_heads": args.seq_heads,
                "num_layers": args.seq_n_layers,
                "dim_feedforward": args.embedding_dim * 4,
                "dropout": args.dropout,
                "activation": "gelu",
                "layer_norm_eps": 1.0e-9,
                "initializer_range": 0.02,
                "num_hops": args.graph_n_layers,
                "topk_k": 10,
                "seq_encoder": args.seq_encoder,
            },
            num_items=dataset_meta["num_items"],
            num_users=dataset_meta["num_users"],
            max_sequence_length=dataset_meta["max_sequence_length"],
            graph=graph,
        ).to(device)

        loss_function = UnderDogLoss(
            cfg={
                "warmup_epochs": args.warmup_epochs,
                "local_coef": args.local_coef,
                "global_coef": args.global_coef,
                "barlow_twins_coef": args.barlow_twins_coef,
                "barlow_twins_lambda": args.barlow_lambda,
                "local_objective": LocalObjective(),
                "global_objective": GlobalObjective(),
            }
        )
        optimizer = BasicOptimizer.create_from_config(
            {
                "optimizer": {"type": "adamw", "lr": args.lr, "weight_decay": args.weight_decay},
                "clip_grad_threshold": 10.0,
            },
            model=model,
        )

        metrics = {
            "ndcg@10": NDCGMetric(k=10),
            "recall@10": RecallMetric(k=10),
        }

        train_start = time.time()
        history, best_metrics = create_train(
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["validation"],
            test_dataloader=dataloaders["test"],
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            num_epochs=args.num_epochs,
            early_stopping_rounds=args.early_stopping_rounds,
            device=device,
            metrics=metrics,
        )
        train_time = best_metrics.get("train_time", time.time() - train_start)
        epochs_ran = max(1, len(history))
        avg_epoch_time = train_time / epochs_ran

        results = compute_create_metrics(dataloaders["test"], model, device, k=10)

        print(
            f"Removal {pct}% -> NDCG@10={results['ndcg@10']:.6f}, "
            f"Precision@10={results['precision@10']:.6f}, "
            f"Recall@10={results['recall@10']:.6f}"
        )

        metrics_overall["ndcg@10"].append(results["ndcg@10"])
        metrics_overall["precision@10"].append(results["precision@10"])
        metrics_overall["recall@10"].append(results["recall@10"])

        time_log_lines.append(
            f"removed={pct}% | original_edges={total_edges} | "
            f"training_edges={kept_count} | avg_epoch_time_sec={avg_epoch_time:.6f}"
        )

    with time_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(time_log_lines) + "\n")

    return metrics_overall


def prepare_create_uni_paths(args: argparse.Namespace, output_dir: Path) -> SplitPaths:
    split_paths = resolve_split_paths(args.dataset, args.data_dir, output_dir)

    if split_paths.train_validation is None:
        split_paths = SplitPaths(
            train=split_paths.train,
            validation=split_paths.validation,
            test=split_paths.test,
            train_validation=_ensure_train_val(
                split_paths.train,
                split_paths.validation,
                split_paths.train.parent / "train_validation.csv",
            ),
        )

    return split_paths


def run_create_uni_robustness(args: argparse.Namespace, output_dir: Path) -> Dict[str, List[float]]:
    device = torch.device(args.device)
    split_paths = prepare_create_uni_paths(args, output_dir)

    if args.run_mode == "test":
        active_train = split_paths.train_validation
        active_val = split_paths.test
        active_test = split_paths.test
    else:
        active_train = split_paths.train
        active_val = split_paths.validation
        active_test = split_paths.validation

    train_df = pd.read_csv(active_train)
    val_df = pd.read_csv(active_val)
    test_df = pd.read_csv(active_test)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    num_users = int(all_df["user_id"].max()) + 1
    num_items = int(all_df["item_id"].max()) + 1

    dataloaders = create_uni_dataloaders(
        train_path=str(active_train),
        val_path=str(active_val),
        test_path=str(active_test),
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        num_workers=0,
        seq_encoder_type=args.seq_encoder,
        num_items=num_items,
    )

    removal_order = build_edge_removal_order(len(train_df), args.seed)

    metrics_overall = {"ndcg@10": [], "precision@10": [], "recall@10": []}
    time_log_path = output_dir / "robust_time_output.txt"
    time_log_lines = [
        "model=CREATEUni",
        f"dataset={args.dataset}",
        f"run_mode={args.run_mode}",
        f"graph_type={args.graph_type}",
        f"graph_conv_type={args.graph_conv_type}",
    ]

    total_edges = len(train_df)
    for pct in [10, 20, 30, 40, 50]:
        remove_count = int(len(train_df) * pct / 100)
        removed_idx = set(removal_order[:remove_count])
        kept_df = train_df.drop(index=train_df.index[list(removed_idx)]).reset_index(drop=True)
        kept_count = len(kept_df)

        print(
            "Running robustness step: "
            f"removed={pct}% | original_edges={total_edges} | training_edges={kept_count}"
        )

        user_ids = torch.tensor(kept_df["user_id"].values, dtype=torch.long, device=device)
        item_ids = torch.tensor(kept_df["item_id"].values, dtype=torch.long, device=device)
        timestamps = None
        if "timestamp" in kept_df.columns and kept_df["timestamp"].notnull().any():
            timestamps = torch.tensor(
                kept_df["timestamp"].fillna(0).values,
                dtype=torch.float,
                device=device,
            )

        model = CREATEUni(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            graph_n_layers=args.graph_n_layers,
            graph_conv_type=args.graph_conv_type,
            graph_heads=args.graph_heads,
            graph_dropout=args.dropout,
            seq_n_layers=args.seq_n_layers,
            seq_heads=args.seq_heads,
            seq_dropout=args.dropout,
            max_sequence_length=args.max_sequence_length,
            seq_encoder_type=args.seq_encoder,
            use_graph=True,
            use_sequence=True,
        ).to(device)

        if args.graph_type == "bipartite" or args.graph_conv_type == "LightGCN":
            edge_index, degV_inv_sqrt = get_bipartite_graph_structure(
                user_ids=user_ids,
                item_ids=item_ids,
                num_users=num_users,
                num_items=num_items,
                device=device,
            )
            model.set_graph_structure(edge_index=edge_index, degV_inv_sqrt=degV_inv_sqrt, is_hypergraph=False)
        else:
            vertex, edges, degV, degE = get_graph_structure(
                user_ids=user_ids,
                item_ids=item_ids,
                timestamps=timestamps,
                num_users=num_users,
                num_items=num_items,
                device=device,
                session_length=args.session_length,
            )
            model.set_graph_structure(vertex, edges, degV, degE, is_hypergraph=True)

        loss_fn = CREATEUniLoss(
            local_coef=args.local_coef,
            global_coef=args.global_coef,
            barlow_twins_coef=args.barlow_twins_coef,
            barlow_lambda=args.barlow_lambda,
            warmup_epochs=args.warmup_epochs,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        metrics = create_metrics(k_values=[10], num_items=num_items)

        train_start = time.time()
        history, best_metrics = create_uni_train(
            train_dataloader=dataloaders["train"],
            val_dataloader=dataloaders["validation"],
            test_dataloader=dataloaders["test"],
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metrics=metrics,
            device=device,
            num_epochs=args.num_epochs,
            early_stopping_rounds=args.early_stopping_rounds,
            log_interval=1,
            warmup_epochs=args.warmup_epochs,
            output_dir=None,
            select_best=(args.run_mode == "val"),
        )
        train_time = best_metrics.get("train_time", time.time() - train_start)
        epochs_ran = max(1, len(history))
        avg_epoch_time = train_time / epochs_ran

        results, _ = create_uni_inference(dataloaders["test"], model, metrics, device)

        print(
            f"Removal {pct}% -> NDCG@10={results['ndcg@10']:.6f}, "
            f"Precision@10={results['precision@10']:.6f}, "
            f"Recall@10={results['recall@10']:.6f}"
        )

        metrics_overall["ndcg@10"].append(results["ndcg@10"])
        metrics_overall["precision@10"].append(results["precision@10"])
        metrics_overall["recall@10"].append(results["recall@10"])

        time_log_lines.append(
            f"removed={pct}% | original_edges={total_edges} | "
            f"training_edges={kept_count} | avg_epoch_time_sec={avg_epoch_time:.6f}"
        )

    with time_log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(time_log_lines) + "\n")

    return metrics_overall


def plot_results(metrics: Dict[str, List[float]], output_path: Path, model_name: str) -> None:
    x = [10, 20, 30, 40, 50]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)

    axes[0].plot(x, metrics["ndcg@10"], marker="o")
    axes[0].set_title("NDCG@10")
    axes[0].set_xlabel("% edges removed")
    axes[0].set_ylabel("NDCG@10")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, metrics["precision@10"], marker="o")
    axes[1].set_title("Precision@10")
    axes[1].set_xlabel("% edges removed")
    axes[1].set_ylabel("Precision@10")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, metrics["recall@10"], marker="o")
    axes[2].set_title("Recall@10")
    axes[2].set_xlabel("% edges removed")
    axes[2].set_ylabel("Recall@10")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Robustness Results: {model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    fix_random_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "CREATE":
        # Route CREATE requests to CREATE-Uni with LightGCN + bipartite graph.
        routed_args = argparse.Namespace(**vars(args))
        routed_args.model = "CREATEUni"
        routed_args.graph_conv_type = "LightGCN"
        routed_args.graph_type = "bipartite"
        metrics = run_create_uni_robustness(routed_args, output_dir)
    else:
        metrics = run_create_uni_robustness(args, output_dir)

    plot_path = output_dir / f"robustness_results_{args.model}.png"
    plot_results(metrics, plot_path, args.model)

    with open(output_dir / f"robustness_results_{args.model}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
