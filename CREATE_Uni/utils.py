"""
Utilities for CREATE-Uni training and evaluation.
"""

import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

# Set seaborn style for plots
try:
    import seaborn as sns
    sns.set_theme()
except ImportError:
    pass


def create_logger(
    name: str,
    level: int = logging.DEBUG,
    format_str: str = "[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Create a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        format_str: Log format string
        datefmt: Date format string
        log_file: Optional path to log file

    Returns:
        logger: Configured logger
    """
    logging.basicConfig(level=level, format=format_str, datefmt=datefmt)
    logger = logging.getLogger(name)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_str, datefmt))
        logger.addHandler(file_handler)

    return logger


def fix_random_seed(seed: int):
    """
    Fix random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Move all tensors in batch to device.

    Args:
        batch: Dictionary of tensors
        device: Target device

    Returns:
        batch: Modified batch (in-place)
    """
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch


def inference(
    dataloader: DataLoader,
    model: nn.Module,
    metrics: Dict[str, any],
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    is_warmup: bool = False,
) -> Tuple[Dict[str, float], Optional[float]]:
    """
    Run inference on a dataloader.

    Args:
        dataloader: Data loader
        model: Model to evaluate
        metrics: Dictionary of metric instances
        device: Device to run on
        loss_fn: Optional loss function to compute loss

    Returns:
        results: Dictionary of metric values
        avg_loss: Average loss if loss_fn provided
    """
    running_metrics = defaultdict(list)
    total_loss = 0.0
    num_batches = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            move_batch(batch, device)
            outputs = model(batch, return_alignment=False, is_warmup=is_warmup)

            # Compute loss if provided
            if loss_fn is not None:
                loss = loss_fn(batch, outputs)
                total_loss += loss.item()
                num_batches += 1

            # Compute metrics
            prediction_key = "graph_prediction" if is_warmup else "local_prediction"
            if prediction_key in outputs and "labels.ids" in batch:
                predictions = outputs[prediction_key]
                targets = batch["labels.ids"]

                # Handle MLM-style with flattened labels vs standard next-item prediction
                if predictions.dim() == 2 and targets.dim() == 1:
                    if predictions.shape[0] == targets.shape[0]:
                        # Standard case: one label per sample
                        for metric_name, metric_fn in metrics.items():
                            if hasattr(metric_fn, "compute_batch"):
                                values = metric_fn.compute_batch(predictions, targets)
                                running_metrics[metric_name].extend(values)
                            else:
                                running_metrics[metric_name].append(metric_fn(predictions, targets))
                    elif predictions.shape[0] > targets.shape[0]:
                        # MLM case: multiple predictions per sample, need to aggregate
                        # This happens with BERT4Rec where we have num_masked predictions
                        # but only batch_size labels
                        pass  # Skip metrics for MLM case during inference

    # Aggregate metrics
    results = {}
    for metric_name, values in running_metrics.items():
        if values:
            results[metric_name] = float(np.mean(values))
        else:
            results[metric_name] = 0.0

    avg_loss = total_loss / num_batches if num_batches > 0 else None

    model.train()
    return results, avg_loss


def train(
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    metrics: Dict[str, any],
    device: torch.device,
    num_epochs: int = 100,
    early_stopping_rounds: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    log_interval: int = 1,
    warmup_epochs: int = 0,
    output_dir: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Training loop for CREATE-Uni.

    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        test_dataloader: Test data loader
        model: Model to train
        optimizer: Optimizer
        loss_fn: Loss function
        metrics: Evaluation metrics
        device: Device to run on
        num_epochs: Number of epochs
        early_stopping_rounds: Patience for early stopping
        scheduler: Optional learning rate scheduler
        log_interval: Log every N epochs
        warmup_epochs: Number of warmup epochs (only global loss)
        output_dir: Directory to save model checkpoints

    Returns:
        history: List of per-epoch metrics
        best_metrics: Best validation metrics
    """
    logger = create_logger("train", level=logging.INFO)
    logger.info(f"Starting training for {num_epochs} epochs on device: {device}")
    if warmup_epochs > 0:
        logger.info(f"Warmup epochs: {warmup_epochs} (global loss only)")

    train_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    train_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if train_start:
        train_start.record()

    best_epoch = 0
    best_val_metric = 0.0
    best_metrics = {}
    history = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        phase = "Warmup" if epoch < warmup_epochs else "Joint"
        desc = f"[{phase}] Epoch {epoch + 1}/{num_epochs}"

        for batch in tqdm(train_dataloader, desc=desc, mininterval=30, ncols=100):
            move_batch(batch, device)
            # Pass is_warmup to save sequence encoder computation during warmup training phase
            outputs = model(batch, return_alignment=(loss_fn.barlow_twins_coef > 0), is_warmup=(epoch < warmup_epochs))

            loss = loss_fn(batch, outputs, epoch_num=epoch, warmup_epochs=warmup_epochs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0

        # Validation
        val_metrics, val_loss = inference(
            val_dataloader, model, metrics, device, loss_fn, is_warmup=(epoch < warmup_epochs)
        )

        # Test
        test_metrics, test_loss = inference(
            test_dataloader, model, metrics, device, loss_fn, is_warmup=(epoch < warmup_epochs)
        )

        # Prepare epoch metrics
        epoch_metrics = {
            "train/loss": avg_train_loss,
        }
        epoch_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        epoch_metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

        history.append(epoch_metrics)

        # Check for improvement (using NDCG@10 as primary metric)
        val_ndcg = val_metrics.get("ndcg@10", 0.0)
        improved = ""
        if val_ndcg > best_val_metric:
            best_val_metric = val_ndcg
            best_epoch = epoch
            best_metrics = epoch_metrics.copy()
            improved = " ★"

            # Save best model checkpoint
            if output_dir:
                checkpoint_path = Path(output_dir) / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ndcg@10": val_ndcg,
                    "best_metrics": best_metrics,
                }, checkpoint_path)

        # Save checkpoint for all epochs (latest model tracking)
        if output_dir:
            latest_path = Path(output_dir) / "latest_checkpoint.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": epoch_metrics,
            }, latest_path)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        # Log one clean line per epoch
        if (epoch + 1) % log_interval == 0:
            logger.info(
                f"[{phase:6s}] Epoch {epoch + 1:3d}/{num_epochs} | "
                f"loss={avg_train_loss:.4f} | "
                f"val_ndcg@10={val_ndcg:.4f} | "
                f"test_ndcg@10={test_metrics.get('ndcg@10', 0.0):.4f}{improved}"
            )

        # Early stopping
        if epoch - best_epoch >= early_stopping_rounds:
            logger.info(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_rounds} epochs)")
            break

    # Save final model checkpoint
    if output_dir:
        final_checkpoint_path = Path(output_dir) / "final_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_ndcg@10": val_ndcg,
            "best_metrics": best_metrics,
        }, final_checkpoint_path)
        logger.info(f"Saved final model checkpoint to {final_checkpoint_path}")

    if train_start:
        train_end.record()
        torch.cuda.synchronize()
        total_time = train_start.elapsed_time(train_end) / 1000.0
    else:
        total_time = 0.0

    best_metrics["train_time"] = total_time
    best_metrics["best_epoch"] = best_epoch

    logger.info(f"Training completed. Best epoch: {best_epoch}, Best val NDCG@10: {best_val_metric:.4f}")
    logger.info(f"Best metrics: {best_metrics}")

    return history, best_metrics


def save_metrics(history: List[Dict], output_dir: str):
    """
    Save training metrics to JSON and plot learning curves.

    Args:
        history: List of per-epoch metrics
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)

    # Organize metrics by split
    metrics_per_split = defaultdict(lambda: defaultdict(list))
    for epoch_metrics in history:
        for key, value in epoch_metrics.items():
            if "/" in key:
                split, metric_name = key.split("/", 1)
                metrics_per_split[split][metric_name].append(value)

    # Plot metrics
    metric_names = sorted(
        {name for split in metrics_per_split.values() for name in split}
    )

    for metric_name in metric_names:
        plt.figure(figsize=(8, 5))
        for split, split_metrics in metrics_per_split.items():
            if metric_name in split_metrics:
                plt.plot(
                    range(1, len(split_metrics[metric_name]) + 1),
                    split_metrics[metric_name],
                    marker="o",
                    label=split,
                )
        plt.title(metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric_name.replace('@', '_at_')}.png", dpi=150)
        plt.close()

    # Save summary
    summary = {
        "total_epochs": len(history),
        "best_epoch": history[-1].get("best_epoch", len(history) - 1) if history else 0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved metrics and plots to {output_dir}")


def compute_item_distance_matrix(
    df: pd.DataFrame,
    num_items: int,
    num_users: int,
) -> torch.Tensor:
    """
    Compute cosine distance matrix between items.

    Args:
        df: DataFrame with user_id, item_id, rating columns
        num_items: Number of items
        num_users: Number of users

    Returns:
        dist_matrix: Item-item distance matrix (num_items, num_items)
    """
    # Build user-item matrix
    mat = torch.zeros((num_items, num_users), dtype=torch.float32)
    mat[df["item_id"], df["user_id"]] = torch.tensor(
        df["rating"].values, dtype=torch.float32
    )

    # Normalize
    mat = nn.functional.normalize(mat, p=2, dim=1)

    # Compute similarity
    sim = mat @ mat.T

    # Convert to distance
    dist = 1 - sim

    return dist


def get_graph_structure(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    num_users: int,
    num_items: int,
    device: torch.device,
    timestamps: Optional[torch.Tensor] = None,
    session_length: int = 86400,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare graph structure for UniGNN message passing using session hyperedges.

    Args:
        user_ids: User IDs tensor (num_interactions,)
        item_ids: Item IDs tensor (num_interactions,)
        num_users: Number of users
        num_items: Number of items
        device: Device to place tensors on
        timestamps: timestamps of interactions (Optional)
        session_length: Maximum temporal length to bound a user's isolated session.

    Returns:
        vertex: Row indices of incidence matrix (2*num_interactions,)
        edges: Column indices of incidence matrix (2*num_interactions,)
        degV: Vertex degree normalization (num_users+num_items,)
        degE: Edge degree normalization (num_interactions,)
    """
    import scipy.sparse as sp
    from torch_scatter import scatter

    # Build bipartite graph in combined space
    # Users: 0 to num_users-1
    # Items: num_users to num_users+num_items-1
    total_nodes = num_users + num_items
    num_interactions = len(user_ids)

    # Shift item IDs to be in combined space
    items_shifted = item_ids + num_users

    # For UniGNN hypergraph representation:
    # A Session hyperedge connects the user and all items interacted with in that session window.
    vertex_list = []
    edges_list = []

    if timestamps is not None:
        from collections import defaultdict
        user_interactions = defaultdict(list)
        for i in range(len(user_ids)):
            u = user_ids[i].item()
            it = items_shifted[i].item()
            t = timestamps[i].item()
            user_interactions[u].append((it, t))
            
        edge_idx_counter = 0
        for u, history in user_interactions.items():
            history.sort(key=lambda x: x[1])
            if not history: continue
            
            curr_start = history[0][1]
            curr_items = []
            
            for it, t in history:
                if t - curr_start >= session_length:
                    # Push hyperedge
                    vertex_list.append(u)
                    edges_list.append(edge_idx_counter)
                    for c_it in curr_items:
                        vertex_list.append(c_it)
                        edges_list.append(edge_idx_counter)
                    edge_idx_counter += 1
                    
                    curr_start = t
                    curr_items = [it]
                else:
                    curr_items.append(it)
                    
            if curr_items:
                vertex_list.append(u)
                edges_list.append(edge_idx_counter)
                for c_it in curr_items:
                    vertex_list.append(c_it)
                    edges_list.append(edge_idx_counter)
                edge_idx_counter += 1
        num_hyperedges = edge_idx_counter
    else:
        # Fallback if timestamps are missing
        for e_idx in range(num_interactions):
            u = user_ids[e_idx].item()
            i = items_shifted[e_idx].item()
            vertex_list.extend([u, i])
            edges_list.extend([e_idx, e_idx])
        num_hyperedges = num_interactions

    vertex = torch.tensor(vertex_list, dtype=torch.long, device=device)
    edges = torch.tensor(edges_list, dtype=torch.long, device=device)

    # Build sparse incidence matrix H (nodes x edges)
    row = vertex.cpu().numpy()
    col = edges.cpu().numpy()
    data = np.ones(len(row))
    H = sp.csr_matrix((data, (row, col)), shape=(total_nodes, num_hyperedges))

    # Compute degrees
    degV_raw = np.array(H.sum(1)).flatten()  # Node degrees
    degE_raw = np.array(H.sum(0)).flatten()  # Edge degrees (size of each session hyperedge)

    degV = torch.from_numpy(degV_raw).float().to(device)
    degE = torch.from_numpy(degE_raw).float().to(device)

    # Normalize for GCN-style propagation: D^(-0.5)
    degE_inv_sqrt = degE.pow(-0.5)
    degE_inv_sqrt[torch.isinf(degE_inv_sqrt)] = 1.0

    degV_inv_sqrt = degV.pow(-0.5)
    degV_inv_sqrt[torch.isinf(degV_inv_sqrt)] = 1.0

    return vertex, edges, degV_inv_sqrt, degE_inv_sqrt
