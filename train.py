"""
CREATE++ / CREATE-Pone Training Script with command-line argument parsing.

Usage:
    # CREATE++ with LightGCN
    python train.py --dataset beauty --sequential sasrec --graph lightgcn

    # CREATE++ with PoneGNN (CREATE-Pone variant)
    python train.py --dataset beauty --sequential sasrec --graph ponegnn

    # CREATE-Pone with custom weights
    python train.py --dataset books --sequential sasrec --graph ponegnn --w_contrast 0.1 --w_ortho 0.1
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from models import CreatePlusPlus, CreatePone
from dataset_loaders import BeautyDataset, BooksDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train CREATE++ / CREATE-Pone model")
    parser.add_argument('--dataset', type=str, default='beauty',
                       choices=['beauty', 'books'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Root directory for data')
    parser.add_argument('--sequential', type=str, default='sasrec',
                       choices=['sasrec', 'bert4rec'],
                       help='Sequential encoder model')
    parser.add_argument('--graph', type=str, default='lightgcn',
                       choices=['lightgcn', 'ultragcn', 'ponegnn'],
                       help='Graph encoder model (use ponegnn for CREATE-Pone variant)')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--max_seq_len', type=int, default=50,
                       help='Maximum sequence length')
    parser.add_argument('--n_heads', type=int, default=2,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of layers (for SASRec/BERT4Rec)')
    parser.add_argument('--gnn_layers', type=int, default=4,
                       help='Number of GNN layers (for graph encoder)')
    parser.add_argument('--n_warmup_epochs', type=int, default=5,
                       help='Number of warmup epochs for graph encoder')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    # Loss weights
    parser.add_argument('--w_local', type=float, default=1.0,
                       help='Weight for local (sequential) loss')
    parser.add_argument('--w_global', type=float, default=1.0,
                       help='Weight for global (graph) loss')
    parser.add_argument('--w_barlow', type=float, default=0.1,
                       help='Weight for Barlow Twins alignment loss')
    parser.add_argument('--w_info', type=float, default=0.0,
                       help='Weight for InfoNCE alignment loss')
    parser.add_argument('--w_ortho', type=float, default=0.1,
                       help='Weight for orthogonality loss (CREATE-Pone only, Eq. 15)')
    parser.add_argument('--w_contrast', type=float, default=0.1,
                       help='Weight for contrastive loss (CREATE-Pone only)')

    # Fusion
    parser.add_argument('--use_mlp_fusion', action='store_true', default=True,
                       help='Use MLP fusion (CREATE++), otherwise linear (CREATE)')
    parser.add_argument('--no_mlp_fusion', action='store_true',
                       help='Use linear fusion instead of MLP')

    # PoneGNN specific
    parser.add_argument('--epsilon_p', type=float, default=0.1,
                       help='PoneGNN: learnable parameter for positive graph self-embedding')
    parser.add_argument('--epsilon_n', type=float, default=0.1,
                       help='PoneGNN: learnable parameter for negative graph self-embedding')
    parser.add_argument('--lambda_cl', type=float, default=0.1,
                       help='PoneGNN: contrastive learning coefficient')
    parser.add_argument('--lambda_reg', type=float, default=5e-5,
                       help='PoneGNN: L2 regularization coefficient')
    parser.add_argument('--use_negative_edges', action='store_true', default=False,
                       help='Use negative edges in training (for PoneGNN)')

    # Device and misc
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_every', type=int, default=10,
                       help='Log every n batches')
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate every n epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_dataset(name, data_root):
    """Load dataset by name."""
    if name == 'beauty':
        dataset = BeautyDataset(root=f"{data_root}/Beauty_and_Personal_Care")
    elif name == 'books':
        dataset = BooksDataset(root=f"{data_root}/Books")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    print(f"Loading {name} dataset...")
    data = dataset.load()
    stats = dataset.get_stats()
    print(f"Dataset loaded: {stats['n_users']} users, {stats['n_items']} items, {stats['n_interactions']} interactions")
    return data, stats


def create_sequences(train_user, train_item, max_seq_len, n_items):
    """Create padded sequences from interaction data."""
    user_sequences = {}
    for u, it in zip(train_user.tolist(), train_item.tolist()):
        if u not in user_sequences:
            user_sequences[u] = []
        user_sequences[u].append(it)

    sequences = []
    for u in sorted(user_sequences.keys()):
        seq = user_sequences[u]
        if len(seq) >= 2:
            for i in range(1, len(seq)):
                padded = [0] * (max_seq_len - min(i, max_seq_len))
                clipped = seq[max(0, i - max_seq_len):i]
                sequences.append(padded + clipped)

    if len(sequences) == 0:
        return torch.zeros(1, max_seq_len, dtype=torch.long)

    return torch.tensor(sequences, dtype=torch.long)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return torch.stack(batch)


def evaluate(model, test_user, test_item, train_user, train_item, seqs, edge_index, edge_weight, negative_edge_index=None, k=10):
    """Evaluate model with Recall@K and NDCG@K."""
    model.eval()
    device = next(model.parameters()).device

    test_user = test_user.to(device)
    test_item = test_item.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    n_users = test_user.max().item() + 1
    n_items = model.n_items

    user_hists = {}
    for u, it in zip(train_user.tolist(), train_item.tolist()):
        if u not in user_hists:
            user_hists[u] = []
        user_hists[u].append(it)

    recall_sum = 0.0
    ndcg_sum = 0.0
    n_test = 0

    seqs = seqs.to(device)
    with torch.no_grad():
        for i in range(len(test_user)):
            u = test_user[i].item()
            true_item = test_item[i].item()

            if u >= seqs.size(0):
                continue

            seq = seqs[u:u+1]

            # Handle CREATE-Pone vs CREATE++
            if isinstance(model, CreatePone):
                fused_emb, item_emb_g = model(seq, edge_index, edge_weight, negative_edge_index)
            else:
                fused_emb, item_emb_g = model(seq, edge_index, edge_weight)

            scores = torch.mm(fused_emb, item_emb_g.t())[0]
            scores[list(set(user_hists.get(u, [])))] = float('-inf')

            _, topk = torch.topk(scores, k)
            topk = topk.cpu().tolist()

            if true_item in topk:
                recall_sum += 1
                rank = topk.index(true_item) + 1
                ndcg_sum += 1.0 / np.log2(rank + 1)

    n_test = len(test_user)
    recall = recall_sum / n_test if n_test > 0 else 0
    ndcg = ndcg_sum / n_test if n_test > 0 else 0

    return {'recall@10': recall, 'ndcg@10': ndcg}


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    data, stats = load_dataset(args.dataset, args.data_root)

    n_users = stats['n_users']
    n_items = stats['n_items']
    edge_index = data['edge_index']
    edge_weight = data.get('edge_weight', torch.ones(edge_index.shape[1]))
    train_user = data['train_user']
    train_item = data['train_item']
    test_user = data['test_user']
    test_item = data['test_item']

    print(f"Creating sequences (max_len={args.max_seq_len})...")
    seqs = create_sequences(train_user, train_item, args.max_seq_len, n_items)

    if len(seqs) > 10000:
        indices = np.random.choice(len(seqs), 10000, replace=False)
        seqs = seqs[indices]

    # Initialize model based on graph encoder choice
    if args.graph == 'ponegnn':
        # Use CREATE-Pone variant
        model = CreatePone(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            max_seq_len=args.max_seq_len,
            sequential_model=args.sequential,
            use_mlp_fusion=not args.no_mlp_fusion,
            n_warmup_epochs=args.n_warmup_epochs,
            w_local=args.w_local,
            w_global=args.w_global,
            w_barlow=args.w_barlow,
            w_info=args.w_info,
            w_ortho=args.w_ortho,
            w_contrast=args.w_contrast,
            ponegnn_kwargs={
                'n_layers': args.gnn_layers,
                'epsilon_p': args.epsilon_p,
                'epsilon_n': args.epsilon_n,
                'lambda_cl': args.lambda_cl,
                'lambda_reg': args.lambda_reg,
            },
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(device)
        print(f"Model: CREATE-Pone with {args.sequential} + PoneGNN")
    else:
        # Use standard CREATE++ with LightGCN or UltraGCN
        model = CreatePlusPlus(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=args.embedding_dim,
            max_seq_len=args.max_seq_len,
            sequential_model=args.sequential,
            graph_model=args.graph,
            use_mlp_fusion=not args.no_mlp_fusion,
            n_warmup_epochs=args.n_warmup_epochs,
            w_local=args.w_local,
            w_global=args.w_global,
            w_barlow=args.w_barlow,
            w_info=args.w_info,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            n_layers_gnn=args.gnn_layers,
        ).to(device)
        print(f"Model: CREATE++ with {args.sequential} + {args.graph}")

    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    seqs = seqs.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nStarting training...")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Fusion: {'MLP + GELU (CREATE++)' if not args.no_mlp_fusion else 'Linear (CREATE)'}")
    print(f"Loss weights: w_local={args.w_local}, w_global={args.w_global}, w_barlow={args.w_barlow}, w_info={args.w_info}", end="")
    if args.graph == 'ponegnn':
        print(f", w_ortho={args.w_ortho}, w_contrast={args.w_contrast}")
    else:
        print()
    print("-" * 50)

    # Prepare negative edges for PoneGNN (ratings <= 3 as negative feedback)
    negative_edge_index = None
    if args.graph == 'ponegnn' and args.use_negative_edges:
        # Create negative edge index from low-rated interactions
        # This would require dataset support for negative edges
        print("Note: Negative edges would be constructed from low-rated items")
        print("      Currently using only positive graph for message passing")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_local = 0.0
        total_global = 0.0
        total_align = 0.0
        total_ortho = 0.0
        total_contrast = 0.0
        total_reg = 0.0

        perm = torch.randperm(len(seqs))
        for i in range(0, len(seqs), args.batch_size):
            batch_idx = perm[i:i+args.batch_size]
            batch_seqs = seqs[batch_idx]

            optimizer.zero_grad()

            if args.graph == 'ponegnn':
                loss, loss_dict = model.compute_loss(
                    batch_seqs, edge_index, edge_weight, negative_edge_index, epoch=epoch
                )
            else:
                loss, loss_dict = model.compute_loss(
                    batch_seqs, edge_index, edge_weight
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_local += loss_dict['local_loss']
            total_global += loss_dict['global_loss']
            total_align += loss_dict.get('alignment_loss', 0.0) + loss_dict.get('info_loss', 0.0)
            total_ortho += loss_dict.get('ortho_loss', 0.0)
            total_contrast += loss_dict.get('contrastive_loss', 0.0)
            total_reg += loss_dict.get('reg_loss', 0.0)

        n_batches = len(seqs) // args.batch_size + 1
        avg_loss = total_loss / n_batches
        avg_local = total_local / n_batches
        avg_global = total_global / n_batches
        avg_align = total_align / n_batches
        avg_ortho = total_ortho / n_batches
        avg_contrast = total_contrast / n_batches
        avg_reg = total_reg / n_batches

        if (epoch + 1) % args.log_every == 0:
            log_msg = (f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                      f"Local: {avg_local:.4f} | Global: {avg_global:.4f} | "
                      f"Align: {avg_align:.4f} | Reg: {avg_reg:.6f}")
            if args.graph == 'ponegnn':
                log_msg += f" | Ortho: {avg_ortho:.4f} | Contrast: {avg_contrast:.4f}"
            print(log_msg)

        if (epoch + 1) % args.eval_every == 0:
            if args.graph == 'ponegnn':
                metrics = evaluate(model, test_user, test_item, train_user, train_item,
                                  seqs, edge_index, edge_weight, negative_edge_index, k=10)
            else:
                metrics = evaluate(model, test_user, test_item, train_user, train_item,
                                  seqs, edge_index, edge_weight, k=10)
            print(f"  Evaluation @10 | Recall: {metrics['recall@10']:.4f} | NDCG: {metrics['ndcg@10']:.4f}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save with appropriate model name
    model_name = f"create_pone_{args.sequential}_{args.dataset}" if args.graph == 'ponegnn' else f"create_plus_plus_{args.sequential}_{args.graph}_{args.dataset}"
    torch.save(model.state_dict(), save_dir / f"{model_name}.pt")
    print(f"\nModel saved to {save_dir}/{model_name}.pt")


if __name__ == '__main__':
    args = parse_args()
    train(args)
