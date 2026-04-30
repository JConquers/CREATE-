#!/usr/bin/env python3
"""
Training script for CREATE-Uni model.

Usage:
    python -m CREATE_Uni.train \
        --dataset beauty \
        --data_dir ./data \
        --output_dir ./outputs \
        --seq_encoder sasrec \
        --embedding_dim 64 \
        --graph_conv_type UniGCN
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from .models import CREATEUni
from .loss import CREATEUniLoss
from .data import create_dataloaders, get_dataset_stats
from .metrics import create_metrics
from .utils import (
    create_logger,
    fix_random_seed,
    train,
    save_metrics,
    get_graph_structure,
    move_batch,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CREATE-Uni model for sequential recommendation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="beauty",
        choices=["beauty", "office_products"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing dataset CSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory for outputs",
    )

    # Model architecture arguments
    parser.add_argument(
        "--seq_encoder",
        type=str,
        default="sasrec",
        choices=["sasrec", "bert4rec"],
        help="Sequence encoder type",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="hypergraph",
        choices=["hypergraph", "bipartite"],
        help="Graph type: hypergraph (UniGNN) or bipartite (LightGCN)",
    )
    parser.add_argument(
        "--graph_conv_type",
        type=str,
        default="UniGCN",
        choices=["UniGCN", "UniGIN", "UniSAGE", "UniGAT", "LightGCN"],
        help="Graph convolution type",
    )
    parser.add_argument(
        "--graph_n_layers",
        type=int,
        default=2,
        help="Number of graph convolution layers",
    )
    parser.add_argument(
        "--graph_heads",
        type=int,
        default=1,
        help="Number of attention heads for UniGAT",
    )
    parser.add_argument(
        "--seq_n_layers",
        type=int,
        default=2,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--seq_heads",
        type=int,
        default=4,
        help="Number of attention heads for transformer",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=50,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--use_graph",
        action="store_true",
        default=True,
        help="Use graph encoder",
    )
    parser.add_argument(
        "--no_graph",
        action="store_false",
        dest="use_graph",
        help="Disable graph encoder",
    )
    parser.add_argument(
        "--use_sequence",
        action="store_true",
        default=True,
        help="Use sequence encoder",
    )
    parser.add_argument(
        "--no_sequence",
        action="store_false",
        dest="use_sequence",
        help="Disable sequence encoder",
    )
    parser.add_argument(
        "--session_length",
        type=int,
        default=86400,
        help="Session length in seconds for hypergraph construction",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for both sequence and graph encoders",
    )
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=50,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Loss arguments
    parser.add_argument(
        "--local_coef",
        type=float,
        default=1.0,
        help="Local objective weight",
    )
    parser.add_argument(
        "--global_coef",
        type=float,
        default=0.6,
        help="Global objective weight",
    )
    parser.add_argument(
        "--barlow_twins_coef",
        type=float,
        default=0.2,
        help="Barlow twins alignment objective weight",
    )
    parser.add_argument(
        "--barlow_lambda",
        type=float,
        default=0.1,
        help="Barlow Twins lambda parameter",
    )

    # Other arguments
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Log every N epochs",
    )
    parser.add_argument(
        "--eval_k",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for evaluation metrics",
    )

    return parser.parse_args()


def get_dataset_paths(args) -> tuple:
    """Get paths to dataset splits."""
    data_dir = Path(args.data_dir) / args.dataset

    # Check if using the dataset_loaders module
    beauty_path = Path(__file__).parent.parent / "dataset_loaders"
    if beauty_path.exists():
        # Use dataset_loaders module
        if args.dataset == "beauty":
            from dataset_loaders.beauty_dataset import BeautyDataset
            dataset = BeautyDataset(root=str(data_dir))
        elif args.dataset == "office_products":
            from dataset_loaders.office_products_dataset import OfficeProductsDataset
            dataset = OfficeProductsDataset(root=str(data_dir))
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        data = dataset.load()
        # Return paths to processed data
        return data

    # Fall back to CSV files
    train_path = data_dir / "train.csv"
    val_path = data_dir / "validation.csv"
    test_path = data_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. "
            f"Please ensure train.csv, validation.csv, and test.csv exist."
        )

    return train_path, val_path, test_path


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.dataset}_{args.seq_encoder}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = create_logger(
        "CREATE-Uni",
        level=logging.INFO,
        log_file=str(output_dir / "training.log"),
    )

    logger.info("=" * 60)
    logger.info("CREATE-Uni Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # Fix random seed
    fix_random_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")

    # Try to use dataset_loaders module
    data_module_path = Path(__file__).parent.parent / "dataset_loaders"
    if data_module_path.exists() and args.dataset in ["beauty", "office_products"]:
        sys.path.insert(0, str(data_module_path.parent))
        if args.dataset == "beauty":
            from dataset_loaders.beauty_dataset import BeautyDataset
            dataset = BeautyDataset(root=str(Path(args.data_dir) / args.dataset))
        else:
            from dataset_loaders.office_products_dataset import OfficeProductsDataset
            dataset = OfficeProductsDataset(root=str(Path(args.data_dir) / args.dataset))

        data = dataset.load()

        # Extract data for CREATE-Uni
        num_users = data["n_users"]
        num_items = data["n_items"]

        # Create temporary CSV files for dataloader
        import pandas as pd

        train_df = pd.DataFrame({
            "user_id": data["train_user"].cpu().numpy(),
            "item_id": data["train_item"].cpu().numpy(),
            "timestamp": data["train_time"].cpu().numpy() if "train_time" in data else None,
        })
        # Build val/test CSVs with full history context.
        # SequenceDataset groups by user_id and Collator splits off the last item
        # as the target, so each user's CSV rows must contain:
        #   val:  [train_items..., val_target]
        #   test: [train_items..., val_item, test_target]
        # The leave-one-out split in BeautyDataset already separated them,
        # so we reconstitute the full sequences here.

        # Index training interactions per user (sorted by timestamp)
        train_history = {}  # user_id -> list of (item, timestamp) sorted by time
        for u, it, t in zip(
            data["train_user"].cpu().numpy(),
            data["train_item"].cpu().numpy(),
            data["train_time"].cpu().numpy() if "train_time" in data else [0] * len(data["train_user"]),
        ):
            train_history.setdefault(int(u), []).append((int(it), float(t)))
        for u in train_history:
            train_history[u].sort(key=lambda x: x[1])

        # Val: training history + val target
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

        # Test: training history + val item + test target
        val_lookup = {}  # user_id -> (item, timestamp)
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
        temp_dir.mkdir()

        train_path = temp_dir / "train.csv"
        val_path = temp_dir / "validation.csv"
        test_path = temp_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
    else:
        # Use CSV files directly
        data_dir = Path(args.data_dir) / args.dataset
        train_path = data_dir / "train.csv"
        val_path = data_dir / "validation.csv"
        test_path = data_dir / "test.csv"

        # Get dataset stats
        stats = get_dataset_stats(str(train_path))
        num_users = stats["n_users"]
        num_items = stats["n_items"]

    logger.info(f"Number of users: {num_users}")
    logger.info(f"Number of items: {num_items}")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_encoder_type=args.seq_encoder,
        num_items=num_items,
    )

    logger.info(
        f"Train batches: {len(dataloaders['train'])}, "
        f"Val batches: {len(dataloaders['validation'])}, "
        f"Test batches: {len(dataloaders['test'])}"
    )

    # Build model
    logger.info("Building model...")
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

        use_graph=args.use_graph,
        use_sequence=args.use_sequence,
        seq_encoder_type=args.seq_encoder,
    )

    # Set up graph structure if using graph encoder
    if args.use_graph and data_module_path.exists() and args.dataset in ["beauty", "office_products"]:
        logger.info("Setting up graph structure...")

        if args.graph_type == "bipartite" or args.graph_conv_type == "LightGCN":
            # LightGCN: simple bipartite graph
            from .utils import get_bipartite_graph_structure
            edge_index, degV_inv_sqrt = get_bipartite_graph_structure(
                user_ids=data["train_user"],
                item_ids=data["train_item"],
                num_users=num_users,
                num_items=num_items,
                device=device,
            )
            model.set_graph_structure(edge_index=edge_index, degV_inv_sqrt=degV_inv_sqrt, is_hypergraph=False)
            logger.info(f"Bipartite graph: {edge_index.shape[1]//2} undirected edges")
        else:
            # UniGNN: session-based hypergraph
            vertex, edges, degV, degE = get_graph_structure(
                user_ids=data["train_user"],
                item_ids=data["train_item"],
                timestamps=data.get("train_time", None),
                num_users=num_users,
                num_items=num_items,
                device=device,
                session_length=args.session_length,
            )
            model.set_graph_structure(vertex, edges, degV, degE)
            logger.info(f"Hypergraph structure: {len(vertex)} nodes in incidence matrix, {edges.max().item() + 1} edges")

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters: {num_params:,}")

    # Build loss function
    loss_fn = CREATEUniLoss(
        local_coef=args.local_coef,
        global_coef=args.global_coef,
        barlow_twins_coef=args.barlow_twins_coef,
        barlow_lambda=args.barlow_lambda,
        warmup_epochs=args.warmup_epochs,
    )

    # Build optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Build metrics
    metrics = create_metrics(k_values=args.eval_k)

    # Train
    logger.info("Starting training...")
    history, best_metrics = train(
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
        log_interval=args.log_interval,
        warmup_epochs=args.warmup_epochs,
        output_dir=str(output_dir),
    )

    # Save results
    logger.info("Saving results...")
    save_metrics(history, str(output_dir))

    # Save config
    config = vars(args)
    config["num_users"] = num_users
    config["num_items"] = num_items
    config["num_parameters"] = num_params
    config["best_metrics"] = best_metrics

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best epoch: {best_metrics.get('best_epoch', 'N/A')}")
    logger.info(f"Best validation NDCG@10: {best_metrics.get('val/ndcg@10', 'N/A'):.4f}")
    logger.info(f"Best test NDCG@10: {best_metrics.get('test/ndcg@10', 'N/A'):.4f}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)

    return best_metrics


if __name__ == "__main__":
    main()
