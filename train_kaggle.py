#!/usr/bin/env python3
"""
Optimized Kaggle-friendly training script for CREATE++ Pone Variant.

Performance optimizations match official PoneGNN implementation:
- Pre-computed negative sampling for all epochs upfront
- Efficient bipartite_dataset pattern with PyTorch DataLoader
- Tensor-based operations instead of pandas
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
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from dataset_loaders import get_dataset, SequenceDataset, SASRecCollator, build_graph_edges
from models import CREATEPlusPlusModel, PoneGNNEncoder, SASRecEncoder


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# Efficient Bipartite Dataset (matching official PoneGNN implementation)
# ============================================================================

class BipartiteDataset(Dataset):
    """
    Efficient bipartite dataset - negatives generated per epoch (matching official).
    """

    def __init__(self, train_df, neg_dist, offset, num_users, num_items, K):
        """
        Args:
            train_df: DataFrame with user_id, item_id, rating columns (0-based contiguous IDs)
            neg_dist: Negative sampling distribution (item popularity)
            offset: Rating threshold for positive/negative split
            num_users, num_items: Dataset statistics
            K: Number of negative samples per positive
        """
        # Build edges
        self.edge_1 = torch.tensor(train_df['user_id'].values).long()
        self.edge_2 = torch.tensor(train_df['item_id'].values).long() + num_users
        self.edge_3 = torch.tensor(train_df['rating'].values).float() - offset

        self.num_users = num_users
        self.num_v = num_items
        self.K = K
        self.neg_dist = neg_dist

        # Build user->positive_items mapping (done once)
        print('Building user->positive items mapping...')
        self.user_pos_items = {}
        for idx in range(len(self.edge_1)):
            user = self.edge_1[idx].item()
            item = self.edge_2[idx].item() - self.num_users
            if user not in self.user_pos_items:
                self.user_pos_items[user] = set()
            self.user_pos_items[user].add(item)

        print(f'Dataset ready: {len(self.edge_1)} edges, {len(self.user_pos_items)} users')

    def generate_negatives(self, epoch):
        """Generate negatives for current epoch - simple per-edge sampling."""
        num_edges = len(self.edge_1)
        K = self.K
        num_v = self.num_v
        neg_dist = self.neg_dist.numpy() if hasattr(self.neg_dist, 'numpy') else self.neg_dist

        # Pre-sample all negatives at once (matching official)
        all_negs = np.random.choice(
            num_v,
            size=num_edges * K,
            replace=True,
            p=neg_dist
        )

        # Convert to tensor and reshape
        edge_4 = torch.from_numpy(all_negs).long().view(num_edges, K) + self.num_users

        return edge_4

    def set_epoch(self, epoch):
        """Set current epoch's negatives by generating them on-the-fly."""
        self.edge_4 = self.generate_negatives(epoch)

    def __len__(self):
        return len(self.edge_1)

    def __getitem__(self, idx):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u, v, w, negs

    def set_epoch(self, epoch):
        """Generate negatives for this epoch."""
        self.edge_4 = self.generate_negatives(epoch)

    def __len__(self):
        return len(self.edge_1)

    def __getitem__(self, idx):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u, v, w, negs


def create_negative_distribution(train_df, num_items, power=0.75):
    """Create item popularity distribution for negative sampling."""
    item_counts = np.zeros(num_items)
    for item in train_df['item_id'].values:
        if item < num_items:
            item_counts[item] += 1

    # Power-law transformation (0.75 is standard)
    item_probs = item_counts ** power
    item_probs /= item_probs.sum()

    return torch.tensor(item_probs, dtype=torch.float)


# ============================================================================
# Training Functions
# ============================================================================

def train_ponegnn_optimized(model, train_df, dataset, device, pretrain_epochs=50,
                             batch_size=2048, lr=1e-3, K=40, eval_every=10,
                             save_path=None):
    """
    Stage 1: Pre-train PoneGNN graph encoder - matching official PoneGNN performance.
    """
    print("\n" + "=" * 60)
    print("Stage 1: Pre-training PoneGNN Graph Encoder")
    print("=" * 60)

    num_users = dataset.num_users
    num_items = dataset.num_items

    # Build signed graph edges
    print("Building signed graph edges...")
    data_p, data_n = build_graph_edges(train_df, num_users, num_items, device=device)
    data_p = data_p.to(device)
    data_n = data_n.to(device)

    # Create negative sampling distribution
    neg_dist = create_negative_distribution(train_df, num_items)

    # Create dataset (negatives generated per-epoch, matching official)
    training_dataset = BipartiteDataset(
        train_df, neg_dist, 3.5, num_users, num_items, K
    )

    # DataLoader
    dataloader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Optimizer and scheduler (matching official: MultiStepLR)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[20, 200], gamma=0.2)

    # Mixed precision
    scaler = GradScaler()

    model.train()
    best_loss = float('inf')
    best_state = None

    # Pre-load edges to GPU once
    pos_edge = data_p.edge_index
    neg_edge = data_n.edge_index

    print(f"\nStarting {pretrain_epochs} epochs of pre-training...")
    print(f"Batch size: {batch_size}, Batches per epoch: ~{len(training_dataset) // batch_size}")

    for epoch in range(1, pretrain_epochs + 1):
        # Generate negatives for this epoch (fast - just samples + simple rejection)
        st = time.time()
        training_dataset.set_epoch(epoch)
        print(f"  Negative sampling: {time.time() - st:.2f}s")

        total_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{pretrain_epochs}", leave=False)

        for u, v, w, negs in pbar:
            u = u.to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            negs = negs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                loss = model.compute_loss(u, v, w, negs, pos_edge, neg_edge, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict().copy()
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, save_path)
            pbar.write(f"  -> Best checkpoint saved (loss: {best_loss:.4f})")

        # Log periodically
        if epoch % eval_every == 0:
            pbar.write(f"Epoch {epoch:3d}/{pretrain_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
    return data_p, data_n


def train_joint(model, train_loader, val_loader, dataset, args, device, data_p, data_n):
    """Stage 2: Joint training of SASRec + PoneGNN."""
    print("\n" + "=" * 60)
    print("Stage 2: Joint Training of SASRec + PoneGNN")
    print("=" * 60)

    # Negative sampler for BPR
    neg_sampler = NegativeSampler(dataset.num_items, dataset.user2items)

    # Optimizer with cosine annealing
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    scaler = GradScaler()

    model.train()
    best_hr = 0
    best_ndcg = 0
    checkpoint_path = None

    if hasattr(args, 'save_dir') and args.save_dir:
        checkpoint_path = os.path.join(args.save_dir, 'joint_training_best.pt')

    pbar = tqdm(range(1, args.num_epochs + 1), desc="Joint Training")

    for epoch in pbar:
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            item_sequences = batch['padded_sequence_ids'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels.ids'].to(device, non_blocking=True)
            users = batch['user.ids'].to(device, non_blocking=True)

            neg_items = neg_sampler.sample(users, n_samples=1).squeeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)

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

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate periodically
        if epoch % args.eval_every == 0:
            metrics = evaluate(model, val_loader, device, top_k=args.top_k)

            improved = False
            if metrics['hit_rate'] > best_hr:
                best_hr = metrics['hit_rate']
                improved = True
            if metrics['ndcg'] > best_ndcg:
                best_ndcg = metrics['ndcg']
                improved = True

            pbar.write(
                f"Epoch {epoch:3d}/{args.num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"HR@{args.top_k}: {metrics['hit_rate']:.4f} | "
                f"NDCG@{args.top_k}: {metrics['ndcg']:.4f}"
            )

            if improved and checkpoint_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_hr': best_hr,
                    'best_ndcg': best_ndcg,
                }, checkpoint_path)

        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best_hr': f'{best_hr:.4f}'})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nJoint training complete.")
    print(f"Best HR@{args.top_k}: {best_hr:.4f}")
    print(f"Best NDCG@{args.top_k}: {best_ndcg:.4f}")

    return best_hr, best_ndcg


# ============================================================================
# Original Functions (unchanged)
# ============================================================================

class NegativeSampler:
    """Negative sampler for BPR training with pre-sampled buffer."""

    def __init__(self, num_items, user2items, power=0.75):
        self.num_items = num_items
        self.user2items = user2items
        self.power = power

        item_counts = np.zeros(num_items)
        for items in user2items.values():
            for item in items:
                item_counts[item] += 1

        self.item_probs = item_counts ** power
        self.item_probs /= self.item_probs.sum()

        self.user_items_set = {uid: set(items) for uid, items in user2items.items()}

        # Pre-sample buffer
        self._buffer_size = 100000
        self._sample_buffer = np.random.choice(num_items, size=self._buffer_size, p=self.item_probs)
        self._buffer_pos = 0

    def sample(self, users, n_samples=1):
        batch_size = len(users)
        negatives = np.zeros((batch_size, n_samples), dtype=np.int64)
        user_list = users.tolist()

        for i, user_id in enumerate(user_list):
            user_items = self.user_items_set.get(user_id, set())
            n_collected = 0
            attempts = 0

            while n_collected < n_samples and attempts < 100:
                if self._buffer_pos + 50 > self._buffer_size:
                    self._sample_buffer = np.random.choice(self.num_items, size=self._buffer_size, p=self.item_probs)
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


def build_dataloaders(dataset, batch_size=2048, max_sequence_length=50):
    """Build DataLoaders for sequential recommendation."""
    user_sequences = dataset.get_user_sequences()

    train_sequences = {}
    val_sequences = {}
    for user_id, items in user_sequences.items():
        if len(items) > 1:
            train_sequences[user_id] = items[:-1]
            val_sequences[user_id] = items[-2:]
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


def evaluate(model, dataloader, device, top_k=10):
    """Evaluate model on validation/test set."""
    model.eval()

    hits = 0
    ndcgs = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            item_sequences = batch['padded_sequence_ids'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            labels = batch['labels.ids'].to(device, non_blocking=True)

            if hasattr(model, 'predict'):
                predictions = model.predict(item_sequences, mask, top_k=top_k)
            else:
                output = model(item_sequences, mask)
                scores = output['item_scores']
                _, predictions = torch.topk(scores, k=top_k, dim=-1)

            correct_mask = (predictions == labels.unsqueeze(1))
            hits += correct_mask.any(dim=1).sum().item()

            ranks = correct_mask.float().argmax(dim=1)
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


def train_create_plus_plus(
    dataset_name='books',
    data_dir='./data',
    max_sequence_length=50,
    embedding_dim=64,
    sasrec_heads=4,
    sasrec_layers=2,
    ponegnn_layers=2,
    fusion_type='concat',
    pretrain_epochs=50,
    num_epochs=100,
    batch_size=256,
    lr=1e-3,
    reg=1e-4,
    dropout=0.1,
    contrastive_weight=0.1,
    K=40,
    alpha=0.5,
    top_k=10,
    eval_every=10,
    gpu=0,
    seed=42,
    output_dir='./outputs',
    save_checkpoint=True,
):
    """Main training function for CREATE++ Pone Variant."""
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    os.makedirs(output_dir, exist_ok=True)
    run_name = f"create++_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    setup_seed(seed)

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

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        dataset,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
    )

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

    # Stage 1: Pre-train graph encoder (using optimized version)
    save_path = os.path.join(run_dir, 'ponegnn_pretrain.pt') if save_checkpoint else None
    data_p, data_n = train_ponegnn_optimized(
        graph_model, train_df, dataset, device,
        pretrain_epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=lr,
        K=K,
        eval_every=eval_every,
        save_path=save_path,
    )

    # Stage 2: Create joint model
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

    # Load pre-trained weights
    create_model.graph_encoder.load_state_dict(graph_model.state_dict())
    print("Loaded pre-trained PoneGNN weights.")

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

    # Save final model
    if save_checkpoint:
        final_path = os.path.join(run_dir, 'create_plus_plus_final.pt')
        checkpoint_path = os.path.join(run_dir, 'joint_training_best.pt')
        checkpoint = torch.load(checkpoint_path) if os.path.exists(checkpoint_path) else None

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
                'K': K,
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
    parser = argparse.ArgumentParser(description='CREATE++ Pone Variant Training')
    parser.add_argument('--dataset', type=str, default='books', choices=['books', 'beauty'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--max_sequence_length', type=int, default=50)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--sasrec_heads', type=int, default=4)
    parser.add_argument('--sasrec_layers', type=int, default=2)
    parser.add_argument('--ponegnn_layers', type=int, default=2)
    parser.add_argument('--fusion_type', type=str, default='concat', choices=['concat', 'sum', 'gate', 'mlp'])
    parser.add_argument('--pretrain_epochs', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--contrastive_weight', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=40, help='Number of negative samples')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--no_save', action='store_true')

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
        K=args.K,
        alpha=args.alpha,
        top_k=args.top_k,
        eval_every=args.eval_every,
        gpu=args.gpu,
        seed=args.seed,
        output_dir=args.output_dir,
        save_checkpoint=not args.no_save,
    )
