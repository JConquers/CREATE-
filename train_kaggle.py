#!/usr/bin/env python3
"""
Kaggle-friendly training script for CREATE++ Pone variant.

This script is designed to be run directly in a Kaggle notebook.
It handles:
- Automatic data download and preprocessing (with skip if already present)
- Two-stage training: PoneGNN pretraining + SASRec+PoneGNN joint training
- Evaluation and checkpoint saving

Usage in Kaggle notebook:
    %run train_kaggle.py --dataset books --mode joint --num_epochs 100
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

# Import local modules
from dataset_loaders import get_dataset, SequenceDataset, SASRecCollator, build_graph_edges
from models import CREATEPlusPlusModel, PoneGNNEncoder, SASRecEncoder


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NegativeSampler:
    """Negative sampler for BPR training."""

    def __init__(self, num_items: int, user2items: dict, power: float = 0.75):
        self.num_items = num_items
        self.user2items = user2items
        self.power = power

        # Compute item popularity distribution
        item_counts = np.zeros(num_items)
        for items in user2items.values():
            for item in items:
                item_counts[item] += 1
        self.item_probs = item_counts ** self.power
        self.item_probs /= self.item_probs.sum()

    def sample(self, users: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample negative items for users."""
        batch_size = len(users)
        negatives = []

        for user_id in users.tolist():
            user_items = set(self.user2items.get(user_id, []))
            for _ in range(n_samples):
                while True:
                    neg = np.random.choice(
                        self.num_items,
                        size=1,
                        p=self.item_probs,
                    )[0]
                    if neg not in user_items:
                        negatives.append(neg)
                        break
        return torch.tensor(negatives, dtype=torch.long).view(batch_size, n_samples)


class RatingBasedSampler:
    """Rating-based negative sampler for PoneGNN."""

    def __init__(self, train_df, num_users: int, num_items: int, offset: float = 3.5, k: int = 40):
        self.num_users = num_users
        self.num_items = num_items
        self.offset = offset
        self.k = k

        # Build user-item-rating mapping
        self.user_items = {}
        for _, row in train_df.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            rating = float(row['rating'])
            if uid not in self.user_items:
                self.user_items[uid] = {'pos': [], 'neg': [], 'all': []}
            self.user_items[uid]['all'].append(iid)
            if rating > offset:
                self.user_items[uid]['pos'].append(iid)
            else:
                self.user_items[uid]['neg'].append(iid)

        # Item distribution for sampling
        self.tot = np.arange(num_items)

    def generate_negatives(self, epoch: int = 0) -> dict:
        """Generate negative samples for all users."""
        negatives = {}
        for user_id, data in self.user_items.items():
            pos_items = data['pos']
            if not pos_items:
                continue

            # Sample negatives
            neg_candidates = np.setdiff1d(self.tot, pos_items)
            if len(neg_candidates) == 0:
                continue

            # Sample k negatives per positive item
            neg_samples = np.random.choice(
                neg_candidates,
                size=len(pos_items) * self.k,
                replace=True,
            )
            negatives[user_id] = neg_samples.reshape(len(pos_items), self.k)

        return negatives


def build_dataloaders(dataset, batch_size: int = 2048, max_sequence_length: int = 50):
    """Build DataLoaders for training/evaluation."""
    user_sequences = dataset.get_user_sequences()

    # Split for training
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
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=0,
    )

    return train_loader, val_loader


def evaluate(model, dataloader, device, top_k: int = 10):
    """Evaluate model on validation/test set."""
    model.eval()

    hits = 0
    ndcgs = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            item_sequences = batch['padded_sequence_ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels.ids'].to(device)

            # Get predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(item_sequences, mask, top_k=top_k)
            else:
                output = model(item_sequences, mask)
                scores = output['item_scores']
                _, predictions = torch.topk(scores, k=top_k, dim=-1)

            # Compute Hit Rate and NDCG
            for i, label in enumerate(labels):
                if label in predictions[i]:
                    hits += 1

                rank = (predictions[i] == label).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    ndcgs += 1.0 / np.log2(rank.item() + 2)

            total += len(labels)

    model.train()
    return {
        'hit_rate': hits / total if total > 0 else 0,
        'ndcg': ndcgs / total if total > 0 else 0,
    }


def train_ponegnn(model, train_df, dataset, device, pretrain_epochs: int = 50,
                  batch_size: int = 256, lr: float = 1e-3, reg: float = 1e-4,
                  eval_every: int = 10, save_path: str = None):
    """Pre-train PoneGNN graph encoder."""
    print("\n" + "=" * 60)
    print("Stage 1: Pre-training PoneGNN Graph Encoder")
    print("=" * 60)

    # Build graph edges
    data_p, data_n = build_graph_edges(
        train_df,
        dataset.num_users,
        dataset.num_items,
        device=device,
    )
    data_p = data_p.to(device)
    data_n = data_n.to(device)

    # Create sampler
    sampler = RatingBasedSampler(
        train_df,
        dataset.num_users,
        dataset.num_items,
        k=40,
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=pretrain_epochs)

    model.train()
    best_loss = float('inf')
    best_state = None

    pbar = tqdm(range(1, pretrain_epochs + 1), desc="Pre-training PoneGNN")
    for epoch in pbar:
        # Generate negatives
        negatives = sampler.generate_negatives(epoch)

        total_loss = 0
        n_batches = 0

        # Create mini-batches
        users = list(sampler.user_items.keys())
        random.shuffle(users)

        for batch_start in range(0, len(users), batch_size):
            batch_users = users[batch_start:batch_start + batch_size]

            for user_id in batch_users:
                if user_id not in negatives:
                    continue

                user_data = sampler.user_items[user_id]
                pos_items = user_data['pos']
                if not pos_items:
                    continue

                for item_id in pos_items:
                    u = torch.tensor([user_id], dtype=torch.long, device=device)
                    i = torch.tensor([item_id], dtype=torch.long, device=device)
                    w = torch.tensor([1.0], dtype=torch.float, device=device)
                    negs = torch.tensor(
                        [negatives[user_id][pos_items.index(item_id)]],
                        dtype=torch.long,
                        device=device,
                    )

                    optimizer.zero_grad()
                    loss = model.compute_loss(
                        u, i, w, negs,
                        data_p.edge_index,
                        data_n.edge_index,
                        epoch,
                    )
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Update progress bar
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'best': f'{best_loss:.4f}'})

        if epoch % eval_every == 0:
            pbar.write(f"Epoch {epoch:3d}/{pretrain_epochs} | Loss: {avg_loss:.4f}")

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
    """Joint training of SASRec + PoneGNN."""
    print("\n" + "=" * 60)
    print("Stage 2: Joint Training of SASRec + PoneGNN")
    print("=" * 60)

    # Negative sampler
    neg_sampler = NegativeSampler(
        dataset.num_items,
        dataset.user2items,
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    model.train()
    best_hr = 0
    best_ndcg = 0
    best_model_state = None
    checkpoint_path = None

    # Determine checkpoint path from save_path if available
    if hasattr(args, 'save_dir') and args.save_dir:
        checkpoint_path = os.path.join(args.save_dir, 'joint_training_best.pt')

    pbar = tqdm(range(1, args.num_epochs + 1), desc="Joint Training")
    for epoch in pbar:
        total_loss = 0
        n_batches = 0

        batch_pbar = tqdm(train_loader, desc=f"  Batch", leave=False)
        for batch in batch_pbar:
            item_sequences = batch['padded_sequence_ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels.ids'].to(device)
            users = batch['user.ids'].to(device)

            # Sample negatives
            neg_items = neg_sampler.sample(users, n_samples=1).to(device)

            optimizer.zero_grad()

            # Compute joint loss
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
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            batch_pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Evaluate
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
                f"NDCG@{args.top_k}: {metrics['ndcg']:.4f} | "
                f"Best: HR={best_hr:.4f}, NDCG={best_ndcg:.4f}"
            )

            # Save best model checkpoint
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
    Complete two-stage training for CREATE++ Pone variant.

    Stage 1: Pre-train PoneGNN graph encoder
    Stage 2: Joint training of SASRec + PoneGNN with fusion

    Args:
        dataset_name: 'books' or 'beauty'
        data_dir: Path to data directory
        max_sequence_length: Maximum sequence length
        embedding_dim: Embedding dimension for both encoders
        sasrec_heads: Number of attention heads in SASRec
        sasrec_layers: Number of transformer layers in SASRec
        ponegnn_layers: Number of graph convolution layers in PoneGNN
        fusion_type: Type of fusion ('concat', 'sum', 'gate', 'mlp')
        pretrain_epochs: Number of epochs for PoneGNN pretraining
        num_epochs: Number of epochs for joint training
        batch_size: Batch size
        lr: Learning rate
        reg: L2 regularization coefficient
        dropout: Dropout rate
        contrastive_weight: Weight for contrastive loss
        alpha: Weight for sequential vs graph loss in joint training
        top_k: Top-K for evaluation
        eval_every: Evaluate every N epochs
        gpu: GPU device ID (-1 for CPU)
        seed: Random seed
        output_dir: Output directory for checkpoints
        save_checkpoint: Whether to save checkpoints

    Returns:
        trained_model, dataset, best_metrics
    """
    # Setup device
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"create++_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Set random seed
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

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(
        dataset,
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
    )

    # Stage 1: Pre-train PoneGNN
    print("\n" + "=" * 60)
    print("CREATE++ Pone Variant Training")
    print("=" * 60)

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

    # Load pre-trained graph encoder weights
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

    # Joint training
    best_hr, best_ndcg = train_joint(
        create_model, train_loader, val_loader, dataset,
        joint_args, device, data_p, data_n,
    )

    # Save final model
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
