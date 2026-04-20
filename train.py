#!/usr/bin/env python3
"""Joint training script for CREATE++ Pone variant.

Implements two-stage training:
1. Pre-train PoneGNN graph encoder
2. Jointly train SASRec + PoneGNN with fusion

Usage:
    python train.py --dataset books --model ponegnn --mode pretrain
    python train.py --dataset beauty --model create_plus_plus --mode joint
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
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch_geometric.data import Data
from tqdm import tqdm


def setup_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CREATE++ Pone Variant Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='books',
        choices=['books', 'beauty'],
        help='Dataset to use for training',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Path to data directory',
    )
    parser.add_argument(
        '--max_sequence_length',
        type=int,
        default=50,
        help='Maximum sequence length for sequential models',
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='create_plus_plus',
        choices=['ponegnn', 'sasrec', 'create_plus_plus'],
        help='Model to train',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='joint',
        choices=['pretrain', 'joint', 'sequential_only'],
        help='Training mode: pretrain (graph only), joint, or sequential_only',
    )
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=64,
        help='Embedding dimension',
    )
    parser.add_argument(
        '--sasrec_heads',
        type=int,
        default=4,
        help='Number of attention heads in SASRec',
    )
    parser.add_argument(
        '--sasrec_layers',
        type=int,
        default=2,
        help='Number of transformer layers in SASRec',
    )
    parser.add_argument(
        '--ponegnn_layers',
        type=int,
        default=2,
        help='Number of graph convolution layers in PoneGNN',
    )
    parser.add_argument(
        '--fusion_type',
        type=str,
        default='concat',
        choices=['concat', 'sum', 'gate', 'mlp'],
        help='Fusion type for combining embeddings',
    )

    # Training arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2048,
        help='Batch size',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--pretrain_epochs',
        type=int,
        default=100,
        help='Number of pre-training epochs for graph encoder',
    )
    parser.add_argument(
        '--reg',
        type=float,
        default=1e-4,
        help='L2 regularization coefficient',
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='Dropout rate',
    )
    parser.add_argument(
        '--contrastive_weight',
        type=float,
        default=0.1,
        help='Weight for contrastive loss',
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Weight for sequential vs graph loss in joint training',
    )

    # Evaluation arguments
    parser.add_argument(
        '--eval_every',
        type=int,
        default=10,
        help='Evaluate every N epochs',
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Top-K for evaluation metrics',
    )

    # System arguments
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID (-1 for CPU)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory for checkpoints and logs',
    )
    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to load',
    )
    parser.add_argument(
        '--save_checkpoint',
        action='store_true',
        help='Save checkpoints during training',
    )

    return parser.parse_args()


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
        self.item_probs = item_counts ** power
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
                    # Filter out positive items
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
            rating = row['rating']
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


def build_dataloaders(dataset, args, mode: str = 'train'):
    """Build DataLoaders for training/evaluation."""
    from dataset_loaders import SequenceDataset, SASRecCollator

    user_sequences = dataset.get_user_sequences()

    if mode == 'train':
        # Split for training
        train_sequences = {}
        for user_id, items in user_sequences.items():
            if len(items) > 1:
                train_sequences[user_id] = items[:-1]
            else:
                train_sequences[user_id] = items

        val_sequences = {}
        for user_id, items in user_sequences.items():
            if len(items) > 1:
                val_sequences[user_id] = items[-2:]
            else:
                val_sequences[user_id] = items

        train_dataset = SequenceDataset(train_sequences, mode='train')
        val_dataset = SequenceDataset(val_sequences, mode='validation')

        train_collator = SASRecCollator(pad_id=0, mode='train')
        val_collator = SASRecCollator(pad_id=0, mode='validation')

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_collator,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=val_collator,
            num_workers=0,
        )

        return train_loader, val_loader

    return None, None


def evaluate(model, dataloader, device, top_k: int = 10, max_items: int = None):
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
                # For sequential encoder only
                output = model(item_sequences, mask)
                scores = output['item_scores']
                _, predictions = torch.topk(scores, k=top_k, dim=-1)

            # Compute Hit Rate
            for i, label in enumerate(labels):
                if label in predictions[i]:
                    hits += 1

                # Compute NDCG
                rank = (predictions[i] == label).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    ndcgs += 1.0 / np.log2(rank.item() + 2)

            total += len(labels)

    model.train()
    return {
        'hit_rate': hits / total if total > 0 else 0,
        'ndcg': ndcgs / total if total > 0 else 0,
    }


def train_ponegnn(model, train_df, dataset, args, device):
    """Pre-train PoneGNN graph encoder."""
    print("\n" + "=" * 60)
    print("Stage 1: Pre-training PoneGNN Graph Encoder")
    print("=" * 60)

    # Build graph edges
    from dataset_loaders import build_graph_edges
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
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.pretrain_epochs)

    model.train()
    best_loss = float('inf')

    for epoch in range(1, args.pretrain_epochs + 1):
        # Generate negatives
        negatives = sampler.generate_negatives(epoch)

        total_loss = 0
        n_batches = 0

        # Create mini-batches
        users = list(sampler.user_items.keys())
        random.shuffle(users)

        for batch_start in range(0, len(users), args.batch_size):
            batch_users = users[batch_start:batch_start + args.batch_size]

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

        if epoch % args.eval_every == 0:
            print(f"Epoch {epoch:3d}/{args.pretrain_epochs} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            if args.save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, os.path.join(args.output_dir, 'ponegnn_best.pt'))

    print(f"\nPre-training complete. Best loss: {best_loss:.4f}")
    return data_p, data_n


def train_joint(model, train_loader, val_loader, dataset, args, device,
                data_p=None, data_n=None):
    """Joint training of SASRec + PoneGNN."""
    print("\n" + "=" * 60)
    print("Stage 2: Joint Training of SASRec + PoneGNN")
    print("=" * 60)

    # Negative sampler
    neg_sampler = NegativeSampler(
        dataset.num_items,
        dataset.user2items,
    )

    # Optimizer - different learning rates for different components
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    model.train()
    best_hr = 0
    best_ndcg = 0

    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", disable=True):
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
                pos_edge_index=data_p.edge_index if data_p is not None else None,
                neg_edge_index=data_n.edge_index if data_n is not None else None,
                negative_samples=neg_items,
                epoch=epoch,
                alpha=args.alpha,
            )

            loss = loss_dict['total_loss']
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Evaluate
        if epoch % args.eval_every == 0:
            metrics = evaluate(model, val_loader, device, top_k=args.top_k)

            if metrics['hit_rate'] > best_hr:
                best_hr = metrics['hit_rate']
            if metrics['ndcg'] > best_ndcg:
                best_ndcg = metrics['ndcg']

            print(
                f"Epoch {epoch:3d}/{args.num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"HR@{args.top_k}: {metrics['hit_rate']:.4f} | "
                f"NDCG@{args.top_k}: {metrics['ndcg']:.4f}"
            )

            if args.save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                }, os.path.join(args.output_dir, 'create_plus_plus_best.pt'))

    print(f"\nJoint training complete.")
    print(f"Best HR@{args.top_k}: {best_hr:.4f}")
    print(f"Best NDCG@{args.top_k}: {best_ndcg:.4f}")


def train_sequential_only(model, train_loader, val_loader, args, device):
    """Train SASRec encoder only."""
    print("\n" + "=" * 60)
    print("Training SASRec (Sequential Only)")
    print("=" * 60)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    model.train()
    best_hr = 0
    best_ndcg = 0

    for epoch in range(1, args.num_epochs + 1):
        total_loss = 0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}"):
            item_sequences = batch['padded_sequence_ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels.ids'].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(item_sequences, mask)
            seq_scores = output['item_scores']

            # Cross-entropy loss
            loss = F.cross_entropy(seq_scores, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Evaluate
        if epoch % args.eval_every == 0:
            metrics = evaluate(model, val_loader, device, top_k=args.top_k)

            if metrics['hit_rate'] > best_hr:
                best_hr = metrics['hit_rate']
            if metrics['ndcg'] > best_ndcg:
                best_ndcg = metrics['ndcg']

            print(
                f"Epoch {epoch:3d}/{args.num_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"HR@{args.top_k}: {metrics['hit_rate']:.4f} | "
                f"NDCG@{args.top_k}: {metrics['ndcg']:.4f}"
            )

    print(f"\nTraining complete.")
    print(f"Best HR@{args.top_k}: {best_hr:.4f}")
    print(f"Best NDCG@{args.top_k}: {best_ndcg:.4f}")


def main():
    """Main training function."""
    args = parse_args()

    # Setup device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    run_name = f"{args.model}_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    args.output_dir = run_dir

    # Set random seed
    setup_seed(args.seed)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    from dataset_loaders import get_dataset

    dataset = get_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_sequence_length=args.max_sequence_length,
    )
    train_df, val_df, test_df = dataset.load_data()

    print(f"Dataset statistics:")
    stats = dataset.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Build model
    print(f"\nBuilding model: {args.model}")

    if args.model == 'ponegnn':
        from models.encoders import PoneGNNEncoder
        model = PoneGNNEncoder(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            embedding_dim=args.embedding_dim,
            num_layers=args.ponegnn_layers,
            reg=args.reg,
            contrastive_weight=args.contrastive_weight,
        )
        args.mode = 'pretrain'

    elif args.model == 'sasrec':
        from models.encoders import SASRecEncoder
        model = SASRecEncoder(
            num_items=dataset.num_items,
            embedding_dim=args.embedding_dim,
            num_heads=args.sasrec_heads,
            num_layers=args.sasrec_layers,
            dropout=args.dropout,
            max_sequence_length=args.max_sequence_length,
        )
        args.mode = 'sequential_only'

    elif args.model == 'create_plus_plus':
        from models.fusion import CREATEPlusPlusModel
        model = CREATEPlusPlusModel(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            embedding_dim=args.embedding_dim,
            sasrec_heads=args.sasrec_heads,
            sasrec_layers=args.sasrec_layers,
            ponegnn_layers=args.ponegnn_layers,
            max_sequence_length=args.max_sequence_length,
            fusion_type=args.fusion_type,
            dropout=args.dropout,
            reg=args.reg,
            contrastive_weight=args.contrastive_weight,
        )

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(dataset, args)

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"\nLoading checkpoint: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Training based on mode
    if args.mode == 'pretrain' or args.model == 'ponegnn':
        # Stage 1: Pre-train graph encoder
        data_p, data_n = train_ponegnn(model, train_df, dataset, args, device)

        # Optionally proceed to joint training
        if args.model == 'create_plus_plus':
            print("\nProceeding to joint training...")
            # Re-initialize joint model with pre-trained weights
            from models.fusion import CREATEPlusPlusModel
            joint_model = CREATEPlusPlusModel(
                num_users=dataset.num_users,
                num_items=dataset.num_items,
                embedding_dim=args.embedding_dim,
                sasrec_heads=args.sasrec_heads,
                sasrec_layers=args.sasrec_layers,
                ponegnn_layers=args.ponegnn_layers,
                max_sequence_length=args.max_sequence_length,
                fusion_type=args.fusion_type,
                dropout=args.dropout,
                reg=args.reg,
                contrastive_weight=args.contrastive_weight,
            )
            joint_model = joint_model.to(device)

            # Load pre-trained graph encoder weights
            graph_encoder_state = model.state_dict()
            joint_model.graph_encoder.load_state_dict(graph_encoder_state)

            train_joint(joint_model, train_loader, val_loader, dataset, args, device, data_p, data_n)

    elif args.mode == 'joint':
        if args.model != 'create_plus_plus':
            print("Joint mode only available for create_plus_plus model")
            return

        # Stage 1: Pre-train graph encoder first
        print("\n" + "=" * 60)
        print("Stage 1: Pre-training PoneGNN Graph Encoder (for joint training)")
        print("=" * 60)
        from models.encoders import PoneGNNEncoder
        graph_model = PoneGNNEncoder(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            embedding_dim=args.embedding_dim,
            num_layers=args.ponegnn_layers,
            reg=args.reg,
            contrastive_weight=args.contrastive_weight,
        )
        graph_model = graph_model.to(device)

        data_p, data_n = train_ponegnn(graph_model, train_df, dataset, args, device)

        # Load pre-trained weights into joint model
        print("\nLoading pre-trained graph encoder weights into CREATE++ model...")
        graph_encoder_state = graph_model.state_dict()
        model.graph_encoder.load_state_dict(graph_encoder_state)

        # Stage 2: Joint training
        train_joint(model, train_loader, val_loader, dataset, args, device, data_p, data_n)

    elif args.mode == 'sequential_only':
        train_sequential_only(model, train_loader, val_loader, args, device)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Outputs saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    import torch.nn.functional as F
    main()


# ============================================================================
# KAGGLE USAGE INSTRUCTIONS:
# ============================================================================
#
# For Kaggle notebook usage, import this script and call main() with args:
#
# Option 1: Two-stage training (PoneGNN pretrain + joint training)
# ---------------------------------------------------------------
# !python train.py --dataset books --model create_plus_plus --mode joint \
#     --embedding_dim 64 --ponegnn_layers 2 --sasrec_layers 2 \
#     --pretrain_epochs 50 --num_epochs 100 --batch_size 256 \
#     --lr 0.001 --fusion_type concat --gpu 0
#
# Option 2: Just SASRec (sequential only)
# ---------------------------------------
# !python train.py --dataset beauty --model sasrec --mode sequential_only \
#     --embedding_dim 64 --sasrec_layers 2 --num_epochs 100 \
#     --batch_size 256 --lr 0.001 --gpu 0
#
# Option 3: Just PoneGNN (graph only)
# ------------------------------------
# !python train.py --dataset books --model ponegnn --mode pretrain \
#     --embedding_dim 64 --ponegnn_layers 2 --pretrain_epochs 100 \
#     --batch_size 256 --lr 0.001 --gpu 0
#
# The script will automatically:
# - Check for preprocessed data in common Kaggle paths
# - Skip download/processing if already present
# - Use leave-last-out splitting for train/val/test
#
# ============================================================================
