#!/usr/bin/env python3
"""
Main training script for CREATE-Pone.

Usage:
    python train.py --dataset beauty --epochs 100 --lr 0.001
    python train.py --dataset books --epochs 100 --embedding_dim 128
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from create_ponue.models.create_pone import CREATEPone
from create_ponue.datasets.beauty_dataset import BeautyDataset
from create_ponue.datasets.books_dataset import BooksDataset


class SequenceDataset(Dataset):
    """Dataset for sequential recommendations."""

    def __init__(self, data, max_seq_len=50):
        self.data = data
        self.max_seq_len = max_seq_len
        # Get unique users from training data
        self.users = torch.unique(data["train_user"]).tolist()
        self.user_sequences = data["user_sequences"]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user_id = self.users[idx]
        sequence = self.user_sequences.get(user_id, [])

        # Ensure sequence contains tensors
        if isinstance(sequence, list):
            sequence = [int(s) for s in sequence]

        # Truncate or pad sequence
        if len(sequence) > self.max_seq_len:
            sequence = sequence[-self.max_seq_len:]
        elif len(sequence) < self.max_seq_len:
            sequence = [0] * (self.max_seq_len - len(sequence)) + sequence

        return {
            "user_id": user_id,
            "sequence": torch.tensor(sequence, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    users = torch.tensor([item["user_id"] for item in batch], dtype=torch.long)
    sequences = torch.stack([item["sequence"] for item in batch])
    attention_mask = sequences != 0

    return {
        "user_indices": users,
        "item_sequence": sequences,
        "attention_mask": attention_mask,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CREATE-Pone model")

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="beauty",
        choices=["beauty", "books"],
        help="Dataset to use (default: beauty)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory to store datasets (default: data)",
    )
    parser.add_argument(
        "--rating_threshold",
        type=int,
        default=4,
        help="Rating threshold for positive/negative split (default: 4)",
    )

    # Model arguments
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--num_graph_layers",
        type=int,
        default=2,
        help="Number of graph encoder layers (default: 2)",
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=2,
        help="Number of transformer layers (default: 2)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of attention heads (default: 2)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=50,
        help="Maximum sequence length (default: 50)",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs (default: 10)",
    )
    parser.add_argument(
        "--num_neg_samples",
        type=int,
        default=1,
        help="Number of negative samples per positive (default: 1)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)",
    )

    # Loss weights
    parser.add_argument(
        "--local_weight",
        type=float,
        default=1.0,
        help="Weight for local loss (default: 1.0)",
    )
    parser.add_argument(
        "--global_weight",
        type=float,
        default=0.1,
        help="Weight for global loss (default: 0.1)",
    )
    parser.add_argument(
        "--align_weight",
        type=float,
        default=0.1,
        help="Weight for alignment loss (default: 0.1)",
    )
    parser.add_argument(
        "--ortho_weight",
        type=float,
        default=0.1,
        help="Weight for orthogonality loss / mu parameter in Barlow Twins (default: 0.1)",
    )
    parser.add_argument(
        "--bt_lambda",
        type=float,
        default=0.1,
        help="Lambda parameter for Barlow Twins off-diagonal regularization (default: 0.1)",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.1,
        help="Weight for contrastive loss in global objective (default: 0.1)",
    )

    # Other arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log interval (default: 10)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)",
    )

    return parser.parse_args()


def load_dataset(args):
    """Load dataset based on arguments."""
    if args.dataset == "beauty":
        dataset = BeautyDataset(root=os.path.join(args.data_dir, "beauty"))
    elif args.dataset == "books":
        dataset = BooksDataset(root=os.path.join(args.data_dir, "books"))
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data = dataset.load(rating_threshold=args.rating_threshold)
    stats = dataset.get_stats(rating_threshold=args.rating_threshold)

    print(f"\nDataset: {args.dataset}")
    print(f"  Users: {stats['n_users']}")
    print(f"  Items: {stats['n_items']}")
    print(f"  Interactions: {stats['n_interactions']}")
    print(f"  Positive edges: {stats['n_positive']}")
    print(f"  Negative edges: {stats['n_negative']}")

    return data, stats


def create_model(stats, args):
    """Create CREATE-Pone model."""
    model = CREATEPone(
        num_users=stats["n_users"],
        num_items=stats["n_items"],
        embedding_dim=args.embedding_dim,
        num_graph_layers=args.num_graph_layers,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        local_weight=args.local_weight,
        global_weight=args.global_weight,
        align_weight=args.align_weight,
        ortho_weight=args.ortho_weight,
        bt_lambda=args.bt_lambda,  # Barlow Twins lambda for off-diagonal regularization
        reg_weight=args.weight_decay,
    )

    return model


def train_epoch(model, dataloader, optimizer, device, epoch, args, edge_index_dict, warmup_mode=False):
    """Train for one epoch.

    Args:
        warmup_mode: If True, only train graph encoder with dual-feedback loss (no local/align/ortho)
    """
    model.train()
    total_loss = 0.0
    loss_dict = {
        "local": 0.0,
        "global": 0.0,
        "dual_feedback": 0.0,
        "pos_bpr": 0.0,
        "neg_bpr": 0.0,
        "contrastive": 0.0,
        "barlow_twins": 0.0,
        "orthogonality": 0.0,
        "align": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        # Get edge indices
        pos_edge_index = edge_index_dict["pos"].to(device)
        neg_edge_index = edge_index_dict["neg"].to(device)

        # Forward pass through graph encoder
        user_pos_emb, item_pos_emb, user_neg_emb, item_neg_emb = model.graph_encoder(
            pos_edge_index,
            neg_edge_index,
        )

        # Forward pass through sequential encoder
        _, user_seq_emb = model.sequential_encoder(
            batch["item_sequence"],
            batch["attention_mask"],
        )

        # Create positive and negative pairs
        users = batch["user_indices"]
        pos_items = batch["item_sequence"][:, -1]  # Last item as positive
        neg_items = torch.randint(0, model.num_items, (len(users),), device=device)

        pos_pairs = torch.stack([users, pos_items], dim=1)
        neg_pairs = torch.stack([users, neg_items], dim=1)

        # Compute losses
        total_loss_batch, losses = model.compute_losses(
            user_seq_emb=user_seq_emb,
            user_pos_emb=user_pos_emb,
            user_neg_emb=user_neg_emb,
            item_pos_emb=item_pos_emb,
            item_neg_emb=item_neg_emb,
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
        )

        # Add regularization
        reg_loss = model.graph_encoder.get_embedding_regularization()

        # In warmup mode: only train with dual-feedback loss (graph encoder only)
        # No local, alignment, orthogonality, or contrastive losses during warmup
        if warmup_mode:
            total_loss_batch = losses["dual_feedback_loss"] + reg_loss
        else:
            total_loss_batch = total_loss_batch + reg_loss

        # Backward pass
        total_loss_batch.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Track losses
        total_loss += total_loss_batch.item()
        for key, value in losses.items():
            # Convert key names: e.g., "barlow_twins_loss" -> "barlow_twins"
            loss_name = key.replace("_loss", "")
            if loss_name in loss_dict:
                loss_dict[loss_name] += value.item()

        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_dict.items()}

    return avg_loss, avg_loss_dict


def evaluate(model, dataloader, device, edge_index_dict, k=10):
    """Evaluate model."""
    import numpy as np

    model.eval()

    pos_edge_index = edge_index_dict["pos"].to(device)
    neg_edge_index = edge_index_dict["neg"].to(device)

    with torch.no_grad():
        _, item_pos_emb, _, _ = model.graph_encoder(pos_edge_index, neg_edge_index)

        ndcg_scores = []
        recall_scores = []

        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            scores = model.predict(
                batch["item_sequence"],
                item_pos_emb,
                batch["attention_mask"],
            )

            # Get target items (last item in sequence)
            targets = batch["item_sequence"][:, -1]

            # Get top-k predictions
            _, topk_items = torch.topk(scores, k=k, dim=1)

            # Compute Recall@K
            hits = (topk_items == targets.unsqueeze(1)).sum(dim=1).float()
            recall_scores.extend(hits.cpu().tolist())

            # Compute NDCG@K (assuming single relevant item)
            for i, target in enumerate(targets):
                hit_positions = (topk_items[i] == target).nonzero()
                if len(hit_positions) > 0:
                    rank = hit_positions[0].item() + 1  # 1-indexed
                    dcg = 1.0 / np.log2(rank + 1)
                    idcg = 1.0 / np.log2(2)  # Ideal DCG for single relevant item
                    ndcg_scores.append(dcg / idcg)
                else:
                    ndcg_scores.append(0.0)

    return {
        f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        f"recall@{k}": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
    }


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    data, stats = load_dataset(args)

    # Create model
    model = create_model(stats, args)
    model = model.to(args.device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create dataloader
    train_dataset = SequenceDataset(data, max_seq_len=args.max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Prepare edge indices
    edge_index_dict = {
        "pos": data["pos_edge_index"],
        "neg": data["neg_edge_index"],
    }

    # Training loop
    print(f"\nStarting training on {args.device}...")
    print(f"Warmup epochs: {args.warmup_epochs} (graph encoder only)")
    print(f"Joint training epochs: {args.epochs - args.warmup_epochs}")
    print(f"Total epochs: {args.epochs}")

    best_ndcg = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Determine if in warmup phase
        warmup_mode = epoch <= args.warmup_epochs

        # Train
        train_loss, train_losses = train_epoch(
            model, train_loader, optimizer, args.device, epoch, args, edge_index_dict, warmup_mode
        )

        # Evaluate
        if epoch % args.log_interval == 0 or epoch == 1:
            eval_metrics = evaluate(model, train_loader, args.device, edge_index_dict)

            print(
                f"Epoch {epoch:3d} | "
                f"Total Loss: {train_loss:.4f} | "
                f"L_local: {train_losses['local']:.4f} | "
                f"L_global: {train_losses['global']:.4f} (L_DF: {train_losses['dual_feedback']:.4f} = L_pos_bpr: {train_losses['pos_bpr']:.4f} + L_neg_bpr: {train_losses['neg_bpr']:.4f}) | "
                f"L_contrastive: {train_losses['contrastive']:.4f} | "
                f"L_align: {train_losses['align']:.4f} (L_barlow_twins: {train_losses['barlow_twins']:.4f} + L_ortho: {train_losses['orthogonality']:.4f}) | "
                f"NDCG@10: {eval_metrics['ndcg@10']:.4f} | "
                f"Recall@10: {eval_metrics['recall@10']:.4f}"
            )

            # Save best model
            if eval_metrics["ndcg@10"] > best_ndcg:
                best_ndcg = eval_metrics["ndcg@10"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": eval_metrics,
                    },
                    os.path.join(args.save_dir, f"create_pone_{args.dataset}_best.pt"),
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete!")
    print(f"Best NDCG@10: {best_ndcg:.4f}")
    print(f"Model saved to {args.save_dir}/create_pone_{args.dataset}_best.pt")


if __name__ == "__main__":
    main()
