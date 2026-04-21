#!/usr/bin/env python3
"""
Kaggle-friendly training script for CREATE++ Pone Variant.

CREATE++ is a hybrid sequential recommendation model that combines:
1. SASRec (Self-Attentive Sequential Recommendation) - captures sequential patterns
2. PoneGNN (Signed Graph Neural Network) - captures user-item graph structure with positive/negative edges

Architecture Overview:
┌─────────────────────────────────────────────────────────────┐
│                    CREATE++ Model                            │
├─────────────────────────────────────────────────────────────┤
│  Input: User item sequence [item_1, item_2, ..., item_t]    │
│                                                               │
│  ┌──────────────┐         ┌──────────────┐                  │
│  │  SASRec      │         │  PoneGNN     │                  │
│  │  Encoder     │         │  Encoder     │                  │
│  │  (Sequential)│         │  (Graph)     │                  │
│  └──────┬───────┘         └──────┬───────┘                  │
│         │                        │                           │
│         └───────────┬────────────┘                           │
│                     ▼                                        │
│              ┌─────────────┐                                 │
│              │   Fusion    │  (concat/sum/gate/mlp)          │
│              └──────┬──────┘                                 │
│                     ▼                                        │
│              ┌─────────────┐                                 │
│              │  Prediction │  → Top-K item recommendations   │
│              └─────────────┘                                 │
└─────────────────────────────────────────────────────────────┘

Training Strategy (Two-Stage):
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Pre-train PoneGNN on signed graph                 │
│  - Positive edges: user-item interactions with rating > 3.5 │
│  - Negative edges: user-item interactions with rating < 3.5 │
│  - Contrastive loss to separate positive/negative embeddings│
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Joint Training                                    │
│  - SASRec learns sequential patterns                        │
│  - PoneGNN refines graph representations                    │
│  - Fusion module combines both signals                      │
│  - BPR loss for ranking + Contrastive loss for graph        │
└─────────────────────────────────────────────────────────────┘

Usage in Kaggle notebook:
    %run train_kaggle.py --dataset beauty --pretrain_epochs 10 --num_epochs 20

For faster iteration, reduce model size:
    %run train_kaggle.py --dataset beauty \\
        --embedding_dim 32 --sasrec_layers 1 --ponegnn_layers 1 \\
        --pretrain_epochs 10 --num_epochs 20 --batch_size 128
"""

import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training

# Import local modules
from dataset_loaders import get_dataset, SequenceDataset, SASRecCollator, build_graph_edges
from models import CREATEPlusPlusModel, PoneGNNEncoder, SASRecEncoder


def setup_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    This ensures that results are deterministic when using the same seed.
    Critical for debugging and comparing different hyperparameter configurations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Use benchmark for faster training (non-deterministic but faster)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class NegativeSampler:
    """
    Negative sampler for BPR (Bayesian Personalized Ranking) training.

    Samples negative items for users based on item popularity distribution.
    Popular items are sampled more frequently (power-law distribution).

    Why negative sampling?
    - In recommendation, we only observe positive interactions (clicks, purchases)
    - We need negative examples to learn what users DON'T like
    - Sampling from popularity distribution is more realistic than uniform
    """

    def __init__(self, num_items: int, user2items: dict, power: float = 0.75):
        """
        Args:
            num_items: Total number of items in the dataset
            user2items: Dict mapping user_id -> list of interacted item_ids
            power: Exponent for popularity distribution (0.75 is standard)
        """
        self.num_items = num_items
        self.user2items = user2items
        self.power = power

        # Compute item popularity distribution
        # Items interacted more often will be sampled more as negatives
        item_counts = np.zeros(num_items)
        for items in user2items.values():
            for item in items:
                item_counts[item] += 1

        # Apply power transformation to smooth the distribution
        self.item_probs = item_counts ** power
        self.item_probs /= self.item_probs.sum()

        # Pre-compute user items as sets for fast lookup
        self.user_items_set = {uid: set(items) for uid, items in user2items.items()}

        # Pre-sample a large buffer of negative candidates for faster sampling
        self._buffer_size = 100000
        self._sample_buffer = np.random.choice(
            num_items, size=self._buffer_size, p=self.item_probs
        )
        self._buffer_pos = 0

    def sample(self, users: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample negative items for a batch of users using vectorized sampling.

        Args:
            users: Tensor of user IDs [batch_size]
            n_samples: Number of negative samples per user

        Returns:
            Tensor of negative item IDs [batch_size, n_samples]
        """
        batch_size = len(users)
        negatives = np.zeros((batch_size, n_samples), dtype=np.int64)
        user_list = users.tolist()

        for i, user_id in enumerate(user_list):
            user_items = self.user_items_set.get(user_id, set())
            n_collected = 0
            attempts = 0

            while n_collected < n_samples and attempts < 100:
                # Get candidates from pre-sampled buffer or sample new one
                if self._buffer_pos + (n_samples - n_collected) > self._buffer_size:
                    self._sample_buffer = np.random.choice(
                        self.num_items, size=self._buffer_size, p=self.item_probs
                    )
                    self._buffer_pos = 0

                candidates = self._sample_buffer[self._buffer_pos:self._buffer_pos + 50]
                self._buffer_pos += 50
                attempts += 1

                for cand in candidates:
                    if cand not in user_items:
                        negatives[i, n_collected] = cand
                        n_collected += 1
                        if n_collected >= n_samples:
                            break

        return torch.from_numpy(negatives).to(users.device)


class RatingBasedSampler:
    """
    Rating-based negative sampler for PoneGNN pre-training.

    Uses explicit ratings to create positive/negative edge labels:
    - Positive edges: interactions with rating > 3.5 (out of 5)
    - Negative edges: interactions with rating < 3.5

    This is specific to datasets with explicit ratings (Amazon Reviews).
    For implicit feedback (clicks only), use NegativeSampler instead.
    """

    def __init__(self, train_df, num_users: int, num_items: int,
                 offset: float = 3.5, k: int = 40):
        """
        Args:
            train_df: DataFrame with user_id, item_id, rating columns
            num_users: Total number of users
            num_items: Total number of items
            offset: Rating threshold for positive/negative split (default 3.5)
            k: Number of negative samples per positive item
        """
        self.num_users = num_users
        self.num_items = num_items
        self.offset = offset
        self.k = k

        # Build user-item-rating mapping
        # Separates items into positive (liked) and negative (disliked)
        self.user_items = {}
        for _, row in train_df.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            rating = float(row['rating'])

            if uid not in self.user_items:
                self.user_items[uid] = {'pos': [], 'neg': [], 'all': []}

            self.user_items[uid]['all'].append(iid)

            # Split by rating threshold
            if rating > offset:
                self.user_items[uid]['pos'].append(iid)
            else:
                self.user_items[uid]['neg'].append(iid)

        # Item index array for efficient negative sampling
        self.tot = np.arange(num_items)

    def generate_negatives(self, epoch: int = 0) -> dict:
        """
        Generate negative samples for all users.

        For each positive item a user liked, samples k negative items
        from the set of items the user hasn't interacted with.

        Args:
            epoch: Current epoch (can be used for dynamic sampling)

        Returns:
            Dict mapping user_id -> array of negative samples [n_pos_items, k]
        """
        negatives = {}
        for user_id, data in self.user_items.items():
            pos_items = data['pos']
            if not pos_items:
                continue

            # Get items user hasn't interacted with as negative candidates
            neg_candidates = np.setdiff1d(self.tot, pos_items)
            if len(neg_candidates) == 0:
                continue

            # Sample k negatives per positive item
            # Shape: [num_positive_items * k]
            neg_samples = np.random.choice(
                neg_candidates,
                size=len(pos_items) * self.k,
                replace=True,
            )
            # Reshape to [num_positive_items, k] for easy indexing
            negatives[user_id] = neg_samples.reshape(len(pos_items), self.k)

        return negatives


def build_dataloaders(dataset, batch_size: int = 2048, max_sequence_length: int = 50):
    """
    Build DataLoaders for sequential recommendation training.

    Creates train/val splits using leave-last-out strategy:
    - Training: All items except the last one
    - Validation: Last item(s) for evaluation

    Args:
        dataset: Dataset object with user_sequences
        batch_size: Batch size for training
        max_sequence_length: Max sequence length for padding/truncation

    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    user_sequences = dataset.get_user_sequences()

    # Split sequences: training uses all but last item, val uses last item(s)
    train_sequences = {}
    val_sequences = {}
    for user_id, items in user_sequences.items():
        if len(items) > 1:
            train_sequences[user_id] = items[:-1]  # All except last
            val_sequences[user_id] = items[-2:]     # Last 2 for validation
        else:
            train_sequences[user_id] = items
            val_sequences[user_id] = items

    train_dataset = SequenceDataset(train_sequences, mode='train')
    val_dataset = SequenceDataset(val_sequences, mode='validation')

    train_collator = SASRecCollator(pad_id=0, mode='train')
    val_collator = SASRecCollator(pad_id=0, mode='validation')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader


def evaluate(model, dataloader, device, top_k: int = 10):
    """
    Evaluate model on validation/test set using Top-K metrics.

    Metrics:
    - Hit Rate @ K (HR@K): Fraction of users where ground truth is in top-K
    - NDCG @ K (Normalized Discounted Cumulative Gain): Accounts for rank position

    Args:
        model: Trained recommendation model
        dataloader: DataLoader with evaluation data
        device: torch device (cpu/cuda)
        top_k: Number of recommendations to consider

    Returns:
        Dict with 'hit_rate' and 'ndcg' scores
    """
    model.eval()

    hits = 0
    ndcgs = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            item_sequences = batch['padded_sequence_ids'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels.ids'].to(device, non_blocking=True)

            # Get model predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(item_sequences, mask, top_k=top_k)
            else:
                output = model(item_sequences, mask)
                scores = output['item_scores']
                _, predictions = torch.topk(scores, k=top_k, dim=-1)

            # Vectorized Hit Rate and NDCG computation
            # Create mask for correct items in predictions
            correct_mask = (predictions == labels.unsqueeze(1))
            hits += correct_mask.any(dim=1).sum().item()

            # Vectorized NDCG: find rank of first correct item
            ranks = correct_mask.float().argmax(dim=1)
            # Only count if there's actually a hit
            has_hit = correct_mask.any(dim=1)
            ndcg_contribs = torch.zeros_like(ranks, dtype=torch.float)
            ndcg_contribs[has_hit] = 1.0 / torch.log2(ranks[has_hit].float() + 2)
            ndcgs += ndcg_contribs.sum().item()

            total += len(labels)

    model.train()
    return {
        'hit_rate': hits / total if total > 0 else 0,
        'ndcg': ndcgs / total if total > 0 else 0,
    }


def train_ponegnn(model, train_df, dataset, device, pretrain_epochs: int = 50,
                  batch_size: int = 256, lr: float = 1e-3, reg: float = 1e-4,
                  eval_every: int = 10, save_path: str = None):
    """
    Stage 1: Pre-train PoneGNN graph encoder on signed graph.

    The PoneGNN encoder learns to separate positive and negative user-item
    interactions in the embedding space using contrastive loss.

    Training loop:
    1. Generate negative samples for each positive interaction
    2. Process in mini-batches for memory efficiency
    3. Compute contrastive loss to push apart positive/negative embeddings
    4. Save checkpoint when best loss improves

    Args:
        model: PoneGNNEncoder model
        train_df: Training DataFrame with ratings
        dataset: Dataset object for statistics
        device: torch device
        pretrain_epochs: Number of pre-training epochs
        batch_size: Batch size
        lr: Learning rate
        reg: L2 regularization
        eval_every: Evaluate and log every N epochs
        save_path: Path to save best checkpoint

    Returns:
        data_p, data_n: Positive and negative graph edge indices
    """
    print("\n" + "=" * 60)
    print("Stage 1: Pre-training PoneGNN Graph Encoder")
    print("=" * 60)

    # Build signed graph edges from ratings
    # Positive edges: rating > 3.5, Negative edges: rating < 3.5
    data_p, data_n = build_graph_edges(
        train_df,
        dataset.num_users,
        dataset.num_items,
        device=device,
    )
    data_p = data_p.to(device)
    data_n = data_n.to(device)

    # Initialize negative sampler
    sampler = RatingBasedSampler(
        train_df,
        dataset.num_users,
        dataset.num_items,
        k=40,  # 40 negatives per positive item
    )

    # Optimizer with cosine annealing for smooth convergence
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    scaler = GradScaler()

    model.train()
    best_loss = float('inf')
    best_state = None

    # Pre-compute edge index on GPU once
    pos_edge = data_p.edge_index
    neg_edge = data_n.edge_index

    # Training loop
    pbar = tqdm(range(1, pretrain_epochs + 1), desc="Pre-training PoneGNN")
    for epoch in pbar:
        # Generate negative samples for all users
        negatives = sampler.generate_negatives(epoch)

        total_loss = 0
        n_batches = 0

        # Collect all training samples for efficient batched processing
        # Pre-allocate arrays for speed
        n_total = sum(len(sampler.user_items[uid]['pos']) for uid in negatives.keys() if uid in sampler.user_items)
        all_users = np.zeros(n_total, dtype=np.int64)
        all_items = np.zeros(n_total, dtype=np.int64)
        all_negs = np.zeros(n_total, dtype=np.int64)
        idx = 0

        for user_id, user_negs in negatives.items():
            if user_id not in sampler.user_items:
                continue
            pos_items = sampler.user_items[user_id]['pos']
            if not pos_items:
                continue
            # Vectorized: get first negative for each positive item
            n_pos = len(pos_items)
            if n_pos <= len(user_negs):
                all_users[idx:idx+n_pos] = user_id
                all_items[idx:idx+n_pos] = pos_items
                all_negs[idx:idx+n_pos] = user_negs[:n_pos, 0]  # Take first negative
                idx += n_pos

        # Trim unused space
        all_users = all_users[:idx]
        all_items = all_items[:idx]
        all_negs = all_negs[:idx]

        # Shuffle indices for stochastic training
        indices = np.random.permutation(len(all_users))

        # Process in mini-batches
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Batch tensor creation (GPU-compatible, zero-copy from numpy)
            batch_users = torch.from_numpy(all_users[batch_indices]).to(device)
            batch_items = torch.from_numpy(all_items[batch_indices]).to(device)
            batch_weights = torch.ones(len(batch_indices), dtype=torch.float, device=device)
            batch_negs = torch.from_numpy(all_negs[batch_indices]).to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with autocast():
                loss = model.compute_loss(
                    batch_users, batch_items, batch_weights, batch_negs,
                    pos_edge,
                    neg_edge,
                    epoch,
                )

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        # Update learning rate
        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Update progress bar with current metrics
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': f'{best_loss:.4f}'})

        # Log progress periodically
        if epoch % eval_every == 0:
            pbar.write(f"Epoch {epoch:3d}/{pretrain_epochs} | Loss: {avg_loss:.4f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            if save_path:
                torch.save(best_state, save_path)
                pbar.write(f"  -> Checkpoint saved (best loss: {best_loss:.4f})")

    print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
    return data_p, data_n


def train_joint(model, train_loader, val_loader, dataset, args, device,
                data_p, data_n):
    """
    Stage 2: Joint training of SASRec + PoneGNN.

    Combines sequential patterns (SASRec) with graph structure (PoneGNN)
    through a fusion module. Optimizes both:
    - BPR loss for sequential recommendation
    - Contrastive loss for graph embeddings

    Args:
        model: CREATEPlusPlusModel with both encoders
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        dataset: Dataset object
        args: Training arguments (epochs, lr, alpha, etc.)
        device: torch device
        data_p, data_n: Positive/negative graph edges from Stage 1

    Returns:
        best_hr, best_ndcg: Best validation metrics achieved
    """
    print("\n" + "=" * 60)
    print("Stage 2: Joint Training of SASRec + PoneGNN")
    print("=" * 60)

    # Negative sampler for BPR loss
    neg_sampler = NegativeSampler(
        dataset.num_items,
        dataset.user2items,
    )

    # Optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Set up GradScaler for mixed precision
    scaler = GradScaler()

    model.train()
    best_hr = 0
    best_ndcg = 0
    best_model_state = None
    checkpoint_path = None

    # Set up checkpoint path if save directory is specified
    if hasattr(args, 'save_dir') and args.save_dir:
        checkpoint_path = os.path.join(args.save_dir, 'joint_training_best.pt')

    # Training loop
    pbar = tqdm(range(1, args.num_epochs + 1), desc="Joint Training")
    for epoch in pbar:
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            item_sequences = batch['padded_sequence_ids'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels.ids'].to(device, non_blocking=True)
            users = batch['user.ids'].to(device, non_blocking=True)

            # Sample negative items for BPR loss
            neg_items = neg_sampler.sample(users, n_samples=1).to(device)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision forward pass
            with autocast():
                loss_dict = model.compute_joint_loss(
                    item_sequences=item_sequences,
                    mask=mask,
                    labels=labels,
                    pos_edge_index=data_p.edge_index,
                    neg_edge_index=data_n.edge_index,
                    negative_samples=neg_items,
                    epoch=epoch,
                    alpha=args.alpha,
                )
                loss = loss_dict['total_loss']

            # Scaled backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Evaluate on validation set periodically
        if epoch % args.eval_every == 0:
            metrics = evaluate(model, val_loader, device, top_k=args.top_k)

            improved = False
            if metrics['hit_rate'] > best_hr:
                best_hr = metrics['hit_rate']
                improved = True
            if metrics['ndcg'] > best_ndcg:
                best_ndcg = metrics['ndcg']
                improved = True

            # Log epoch results
            pbar.write(
                f"Epoch {epoch:3d}/{args.num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"HR@{args.top_k}: {metrics['hit_rate']:.4f} | "
                f"NDCG@{args.top_k}: {metrics['ndcg']:.4f} | "
                f"Best: HR={best_hr:.4f}, NDCG={best_ndcg:.4f}"
            )

            # Save best model checkpoint when metrics improve
            if improved and checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_hr': best_hr,
                    'best_ndcg': best_ndcg,
                    'loss': avg_loss,
                }, checkpoint_path)
                pbar.write(f"  -> Best model saved!")

        # Update main progress bar postfix
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best_hr': f'{best_hr:.4f}'})

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nJoint training complete.")
    print(f"Best HR@{args.top_k}: {best_hr:.4f}")
    print(f"Best NDCG@{args.top_k}: {best_ndcg:.4f}")

    return best_hr, best_ndcg


def train_create_plus_plus(
    dataset_name: str = 'books',
    data_dir: str = './data',
    max_sequence_length: int = 50,
    embedding_dim: int = 64,
    sasrec_heads: int = 4,
    sasrec_layers: int = 2,
    ponegnn_layers: int = 2,
    fusion_type: str = 'concat',
    pretrain_epochs: int = 50,
    num_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    reg: float = 1e-4,
    dropout: float = 0.1,
    contrastive_weight: float = 0.1,
    alpha: float = 0.5,
    top_k: int = 10,
    eval_every: int = 10,
    gpu: int = 0,
    seed: int = 42,
    output_dir: str = './outputs',
    save_checkpoint: bool = True,
):
    """
    Main training function for CREATE++ Pone Variant.

    Orchestrates the complete two-stage training pipeline:
    1. Load and preprocess dataset
    2. Build dataloaders
    3. Stage 1: Pre-train PoneGNN graph encoder
    4. Stage 2: Joint training with SASRec + PoneGNN
    5. Save checkpoints and final model

    Args:
        dataset_name: 'books' or 'beauty'
        data_dir: Path to data directory
        max_sequence_length: Max sequence length for SASRec
        embedding_dim: Embedding dimension for both encoders
        sasrec_heads: Number of attention heads in SASRec
        sasrec_layers: Number of transformer layers in SASRec
        ponegnn_layers: Number of graph conv layers in PoneGNN
        fusion_type: Fusion method ('concat', 'sum', 'gate', 'mlp')
        pretrain_epochs: Stage 1 epochs
        num_epochs: Stage 2 epochs
        batch_size: Batch size
        lr: Learning rate
        reg: L2 regularization
        dropout: Dropout rate
        contrastive_weight: Weight for contrastive loss
        alpha: Balance between sequential and graph loss
        top_k: Top-K for evaluation metrics
        eval_every: Evaluate every N epochs
        gpu: GPU device ID (-1 for CPU)
        seed: Random seed for reproducibility
        output_dir: Directory for checkpoints
        save_checkpoint: Whether to save model checkpoints

    Returns:
        trained_model, dataset, best_metrics
    """
    # Setup device (GPU or CPU)
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Setup output directory for checkpoints
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"create++_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set random seed for reproducibility
    setup_seed(seed)

    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
    )
    train_df, val_df, test_df = dataset.load_data()

    print(f"Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Build dataloaders for sequential training
    train_loader, val_loader = build_dataloaders(
        dataset,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
    )

    # Print training configuration
    print("\n" + "=" * 60)
    print("CREATE++ Pone Variant Training")
    print("=" * 60)

    # Initialize PoneGNN graph encoder
    graph_model = PoneGNNEncoder(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=embedding_dim,
        num_layers=ponegnn_layers,
        reg=reg,
        temperature=1.0,
        contrastive_weight=contrastive_weight,
    )
    graph_model = graph_model.to(device)

    # Stage 1: Pre-train graph encoder
    save_path = os.path.join(run_dir, 'ponegnn_pretrain.pt') if save_checkpoint else None
    data_p, data_n = train_ponegnn(
        graph_model, train_df, dataset, device,
        pretrain_epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=lr,
        reg=reg,
        eval_every=eval_every,
        save_path=save_path,
    )

    # Stage 2: Create joint model and load pre-trained weights
    print("\nCreating CREATE++ model with pre-trained graph encoder...")
    create_model = CREATEPlusPlusModel(
        num_users=dataset.num_users,
        num_items=dataset.num_items,
        embedding_dim=embedding_dim,
        sasrec_heads=sasrec_heads,
        sasrec_layers=sasrec_layers,
        ponegnn_layers=ponegnn_layers,
        max_sequence_length=max_sequence_length,
        fusion_type=fusion_type,
        dropout=dropout,
        reg=reg,
        temperature=1.0,
        contrastive_weight=contrastive_weight,
    )
    create_model = create_model.to(device)

    # Load pre-trained graph encoder weights into joint model
    create_model.graph_encoder.load_state_dict(graph_model.state_dict())
    print("Loaded pre-trained PoneGNN weights.")

    # Training args for joint stage
    class JointArgs:
        def __init__(self):
            self.num_epochs = num_epochs
            self.lr = lr
            self.alpha = alpha
            self.top_k = top_k
            self.eval_every = eval_every
            self.save_dir = run_dir if save_checkpoint else None

    joint_args = JointArgs()

    # Stage 2: Joint training
    best_hr, best_ndcg = train_joint(
        create_model, train_loader, val_loader, dataset,
        joint_args, device, data_p, data_n,
    )

    # Save final model with full configuration
    if save_checkpoint:
        final_path = os.path.join(run_dir, 'create_plus_plus_final.pt')
        checkpoint = torch.load(os.path.join(run_dir, 'joint_training_best.pt')) if os.path.exists(os.path.join(run_dir, 'joint_training_best.pt')) else None

        torch.save({
            'model_state_dict': create_model.state_dict(),
            'graph_encoder_state_dict': graph_model.state_dict(),
            'best_hr': best_hr,
            'best_ndcg': best_ndcg,
            'config': {
                'dataset': dataset_name,
                'embedding_dim': embedding_dim,
                'sasrec_heads': sasrec_heads,
                'sasrec_layers': sasrec_layers,
                'ponegnn_layers': ponegnn_layers,
                'fusion_type': fusion_type,
                'pretrain_epochs': pretrain_epochs,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'lr': lr,
                'reg': reg,
                'dropout': dropout,
                'contrastive_weight': contrastive_weight,
                'alpha': alpha,
                'seed': seed,
            },
            'best_joint_checkpoint': checkpoint,
        }, final_path)
        print(f"\nFinal model saved to: {final_path}")

    print("\n" + "=" * 60)
    print("CREATE++ Training Complete!")
    print(f"Output directory: {run_dir}")
    print(f"Best HR@{top_k}: {best_hr:.4f}")
    print(f"Best NDCG@{top_k}: {best_ndcg:.4f}")
    print("=" * 60)

    return create_model, dataset, {'hit_rate': best_hr, 'ndcg': best_ndcg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CREATE++ Pone Variant Training for Kaggle',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--dataset', type=str, default='books',
                        choices=['books', 'beauty'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--max_sequence_length', type=int, default=50,
                        help='Maximum sequence length')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--sasrec_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--sasrec_layers', type=int, default=2,
                        help='Number of SASRec layers')
    parser.add_argument('--ponegnn_layers', type=int, default=2,
                        help='Number of PoneGNN layers')
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'sum', 'gate', 'mlp'],
                        help='Fusion type')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                        help='Pretraining epochs')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Joint training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--reg', type=float, default=1e-4,
                        help='Regularization')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Contrastive loss weight')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Sequential vs graph loss weight')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Top-K for evaluation')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--no_save', action='store_true',
                        help='Disable checkpoint saving')

    args = parser.parse_args()

    train_create_plus_plus(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_sequence_length=args.max_sequence_length,
        embedding_dim=args.embedding_dim,
        sasrec_heads=args.sasrec_heads,
        sasrec_layers=args.sasrec_layers,
        ponegnn_layers=args.ponegnn_layers,
        fusion_type=args.fusion_type,
        pretrain_epochs=args.pretrain_epochs,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        reg=args.reg,
        dropout=args.dropout,
        contrastive_weight=args.contrastive_weight,
        alpha=args.alpha,
        top_k=args.top_k,
        eval_every=args.eval_every,
        gpu=args.gpu,
        seed=args.seed,
        output_dir=args.output_dir,
        save_checkpoint=not args.no_save,
    )
