"""
Training loop for CREATE-Pone.
Implements two-phase training: warm-up + joint optimization.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class CREATEPoneTrainer:
    """
    Trainer for CREATE-Pone model.

    Two-phase training:
    1. Warm-up: Train only graph encoder with global loss
    2. Joint: End-to-end optimization with all losses
    """

    def __init__(
        self,
        model,
        optimizer,
        device="cuda",
        warmup_epochs=10,
        grad_clip=1.0,
        save_dir="checkpoints",
        save_interval=10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.save_dir = save_dir
        self.save_interval = save_interval
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, dataloader, epoch, warmup_mode=False):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_dict = {"local": 0.0, "global": 0.0, "align": 0.0, "ortho": 0.0}
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", leave=False):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            loss = self._train_batch(batch, epoch, warmup_mode=warmup_mode)
            total_loss += loss
            num_batches += 1

        return total_loss / max(num_batches, 1), {k: v / max(num_batches, 1) for k, v in loss_dict.items()}

    def _train_batch(self, batch, epoch, warmup_mode=False):
        """Train on a single batch."""
        self.optimizer.zero_grad()

        # Forward pass through graph encoder
        user_pos_emb, item_pos_emb, user_neg_emb, item_neg_emb = self.model.graph_encoder(
            batch["pos_edge_index"],
            batch["neg_edge_index"],
        )

        # Forward pass through sequential encoder
        _, user_seq_emb = self.model.sequential_encoder(
            batch["item_sequence"],
            batch.get("attention_mask"),
        )

        # Compute losses
        total_loss, losses = self.model.compute_losses(
            user_seq_emb=user_seq_emb,
            user_pos_emb=user_pos_emb,
            user_neg_emb=user_neg_emb,
            item_pos_emb=item_pos_emb,
            item_neg_emb=item_neg_emb,
            pos_pairs=batch.get("pos_pairs", torch.stack([batch["user_indices"], batch["item_sequence"][:, -1]], dim=1)),
            neg_pairs=batch.get("neg_pairs", torch.stack([batch["user_indices"], torch.randint(0, self.model.num_items, (len(batch["user_indices"]),), device=self.device)], dim=1)),
        )

        # Add regularization
        reg_loss = self.model.graph_encoder.get_embedding_regularization()

        # In warmup mode: only use dual-feedback loss
        if warmup_mode:
            total_loss = losses["dual_feedback_loss"] + reg_loss
        else:
            total_loss = total_loss + reg_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()

        return total_loss.item()

    def evaluate(self, dataloader, k=10):
        """Evaluate model on test data."""
        self.model.eval()
        all_users = []
        all_items = []
        all_scores = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

                # Get item embeddings from graph encoder
                _, item_pos_emb, _, _ = self.model.graph_encoder(
                    batch["pos_edge_index"],
                    batch["neg_edge_index"],
                )

                # Get predictions
                scores = self.model.predict(
                    batch["item_sequence"],
                    item_pos_emb,
                    batch.get("attention_mask"),
                )

                all_users.append(batch["user_indices"])
                all_items.append(batch["target_items"])
                all_scores.append(scores)

        return all_users, all_items, all_scores

    def compute_metrics(self, users, items, scores, k_values=(10, 20, 50)):
        """Compute NDCG@K and Recall@K metrics."""
        metrics = {f"ndcg@{k}": 0.0 for k in k_values}
        metrics.update({f"recall@{k}": 0.0 for k in k_values})
        num_samples = 0

        for user_ids, target_items, score_matrix in zip(users, items, scores):
            batch_size = len(user_ids)
            num_samples += batch_size

            for k in k_values:
                _, topk_items = torch.topk(score_matrix, k=k, dim=1)

                # Compute metrics for each user in batch
                for i in range(batch_size):
                    target = target_items[i].item()
                    preds = topk_items[i]

                    # Check if target is in top-k
                    hits = (preds == target).sum().item()

                    if hits > 0:
                        # Recall@K: binary hit or miss
                        metrics[f"recall@{k}"] += 1.0

                        # NDCG@K: discounted cumulative gain
                        # Find position of first hit
                        hit_positions = torch.where(preds == target)[0]
                        if len(hit_positions) > 0:
                            rank = hit_positions[0].item() + 1  # 1-indexed rank
                            dcg = 1.0 / np.log2(rank + 1)
                            idcg = 1.0 / np.log2(2)  # Ideal DCG for single relevant item
                            metrics[f"ndcg@{k}"] += dcg / idcg

        # Average over all samples
        for key in metrics:
            metrics[key] /= max(num_samples, 1)

        return metrics

    def save_checkpoint(self, epoch, metrics, losses, filename=None):
        """Save a training checkpoint."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "losses": losses,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {checkpoint_path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def train(self, dataloader, val_dataloader, epochs, edge_index_dict, args):
        """
        Full training loop with warmup and joint training phases.

        Args:
            dataloader: Training data loader
            val_dataloader: Validation data loader
            epochs: Total number of epochs
            edge_index_dict: Dictionary with 'pos' and 'neg' edge indices
            args: Training arguments
        """
        print(f"\nStarting training on {self.device}...")
        print(f"Warmup epochs: {self.warmup_epochs} (graph encoder only)")
        print(f"Joint training epochs: {epochs - self.warmup_epochs}")
        print(f"Total epochs: {epochs}")

        best_ndcg = 0.0
        patience = 10
        patience_counter = 0

        for epoch in tqdm(range(1, epochs + 1), desc="Training Progress"):
            # Determine if in warmup phase
            warmup_mode = epoch <= self.warmup_epochs

            # Train
            train_loss, train_losses = self.train_epoch(dataloader, epoch, warmup_mode=warmup_mode)

            # Evaluate
            if epoch % args.log_interval == 0 or epoch == 1:
                eval_metrics = self.compute_metrics(*self.evaluate(val_dataloader))

                print(
                    f"Epoch {epoch:3d} | "
                    f"Total Loss: {train_loss:.4f} | "
                    f"NDCG@10: {eval_metrics['ndcg@10']:.4f} | "
                    f"Recall@10: {eval_metrics['recall@10']:.4f}"
                )

                # Save checkpoint for every logged epoch
                self.save_checkpoint(epoch, eval_metrics, train_losses)

                # Save best model
                if eval_metrics["ndcg@10"] > best_ndcg:
                    best_ndcg = eval_metrics["ndcg@10"]
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "metrics": eval_metrics,
                    }, os.path.join(self.save_dir, "best_model.pt"))
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"\nTraining complete!")
        print(f"Best NDCG@10: {best_ndcg:.4f}")
        print(f"Best model saved to {self.save_dir}/best_model.pt")


class BatchGenerator:
    """
    Generate training batches with positive and negative samples.
    """

    def __init__(self, dataset, batch_size, num_neg_samples=1, max_seq_len=50):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.num_items = dataset["n_items"]
        self.max_seq_len = max_seq_len
        # Pre-build user sequences
        self.user_sequences = dataset.get("user_sequences", {})

    def __iter__(self):
        """Generate batches."""
        indices = torch.randperm(len(self.dataset["train_user"]))

        for start_idx in range(0, len(indices), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]

            users = self.dataset["train_user"][batch_indices]
            pos_items = self.dataset["train_item"][batch_indices]

            # Sample negative items (ensure they're different from positive)
            neg_items = torch.randint(0, self.num_items, (len(batch_indices),))

            batch = {
                "user_indices": users,
                "pos_pairs": torch.stack([users, pos_items], dim=1),
                "neg_pairs": torch.stack([users, neg_items], dim=1),
                "item_sequence": self._build_sequence(users),
                "pos_edge_index": self.dataset["pos_edge_index"],
                "neg_edge_index": self.dataset["neg_edge_index"],
            }

            # Create attention mask
            batch["attention_mask"] = batch["item_sequence"] != 0

            yield batch

    def _build_sequence(self, users):
        """Build item sequences for users."""
        sequences = []
        for user in users:
            user_id = user.item()
            if user_id in self.user_sequences:
                seq = self.user_sequences[user_id]
                if isinstance(seq, list):
                    seq = seq[:self.max_seq_len]
                else:
                    seq = seq.tolist()[:self.max_seq_len]
            else:
                seq = []

            # Pad sequence if needed
            if len(seq) < self.max_seq_len:
                seq = [0] * (self.max_seq_len - len(seq)) + seq
            else:
                seq = seq[-self.max_seq_len:]  # Truncate to max_seq_len

            sequences.append(torch.tensor(seq, dtype=torch.long))

        return torch.stack(sequences)

    def _sample_neg_edges(self):
        """Sample negative edges for negative graph."""
        # For ratings <= 3, create negative edges
        # This is a simplified version
        edge_index = self.dataset["edge_index"]
        return edge_index  # In practice, filter by rating
