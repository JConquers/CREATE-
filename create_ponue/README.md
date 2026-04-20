# CREATE-Pone

**CREATE-Pone** is a PyTorch implementation of the CREATE-Pone model from the CREATE++ paper. It combines ideas from:

- **CREATE** (2026): Cross-Representation Knowledge Transfer for Improved Sequential Recommendations
- **Pone-GNN** (2025): Integrating Positive and Negative Feedback in Graph Neural Networks

## Architecture

CREATE-Pone extends the CREATE framework by replacing the unsigned bipartite graph encoder with a **signed graph encoder** inspired by Pone-GNN. This enables the model to:

1. **Learn dual embeddings**: Interest embeddings (from positive interactions) and disinterest embeddings (from negative interactions)
2. **Signed alignment**: Align sequential representations with graph interest embeddings while maintaining orthogonality to disinterest embeddings
3. **Multi-objective training**: Joint optimization of local (sequential), global (graph), alignment, and orthogonality losses

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CREATE-Pone Model                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │  Signed Graph    │         │  Sequential      │         │
│  │  Encoder         │         │  Encoder         │         │
│  │  (Pone-GNN)      │         │  (SASRec)        │         │
│  │                  │         │                  │         │
│  │  - Interest emb  │         │  - Item emb      │         │
│  │  - Disinterest   │         │  - Transformer   │         │
│  │    emb           │         │  - Causal attn   │         │
│  └─────────────────┘         │                  │         │
│           │                   └────────┬─────────┘         │
│           │                            │                   │
│           └──────────┬─────────────────┘                   │
│                      │                                     │
│           ┌──────────▼──────────┐                         │
│           │  Alignment Module   │                         │
│           │  (Barlow Twins)     │                         │
│           │                     │                         │
│           │  - Align seq ↔      │                         │
│           │    graph interest   │                         │
│           │  - Orthogonal to    │                         │
│           │    disinterest      │                         │
│           └─────────────────────┘                         │
│                                                            │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
create_ponue/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── signed_encoder.py    # Pone-GNN style signed graph encoder
│   ├── sequential_encoder.py # SASRec transformer + alignment
│   └── create_pone.py       # Main CREATE-Pone model
├── datasets/
│   ├── __init__.py
│   ├── beauty_dataset.py    # Amazon Beauty dataset loader
│   └── books_dataset.py     # Amazon Books dataset loader
├── trainer.py               # Training loop utilities
├── train.py                 # Main training script
└── requirements.txt
```

## Installation

```bash
pip install -r create_ponue/requirements.txt
```

## Usage

### Training on Amazon Beauty

```bash
python create_ponue/train.py --dataset beauty --epochs 100 --lr 0.001 --batch_size 256
```

### Training on Amazon Books

```bash
python create_ponue/train.py --dataset books --epochs 100 --lr 0.001 --embedding_dim 128
```

### Full Example with Custom Hyperparameters

```bash
python create_ponue/train.py \
    --dataset beauty \
    --epochs 150 \
    --lr 0.0005 \
    --embedding_dim 128 \
    --num_graph_layers 3 \
    --num_transformer_layers 2 \
    --num_heads 4 \
    --batch_size 512 \
    --warmup_epochs 20 \
    --local_weight 1.0 \
    --global_weight 0.1 \
    --align_weight 0.1 \
    --ortho_weight 0.1 \
    --save_dir checkpoints/beauty
```

## Command-Line Arguments

### Dataset Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | beauty | Dataset to use (`beauty` or `books`) |
| `--data_dir` | data | Directory to store datasets |
| `--rating_threshold` | 4 | Rating threshold for positive/negative split (≥4 = positive, ≤3 = negative) |

### Model Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--embedding_dim` | 64 | Embedding dimension |
| `--num_graph_layers` | 2 | Number of graph encoder layers |
| `--num_transformer_layers` | 2 | Number of transformer layers |
| `--num_heads` | 2 | Number of attention heads |
| `--dropout` | 0.1 | Dropout rate |
| `--max_seq_len` | 50 | Maximum sequence length |

### Training Options
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 100 | Number of training epochs |
| `--batch_size` | 256 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--weight_decay` | 1e-5 | Weight decay |
| `--warmup_epochs` | 10 | Number of warmup epochs (graph-only training) |
| `--grad_clip` | 1.0 | Gradient clipping value |

### Loss Weights
| Argument | Default | Description |
|----------|---------|-------------|
| `--local_weight` | 1.0 | Weight for sequential (local) loss |
| `--global_weight` | 0.1 | Weight for graph (global) loss |
| `--align_weight` | 0.1 | Weight for alignment loss |
| `--ortho_weight` | 0.1 | Weight for orthogonality loss |

## Training Phases

CREATE-Pone uses a **two-phase training** strategy:

1. **Warm-up Phase** (first `warmup_epochs` epochs):
   - Train only the graph encoder with global BPR loss
   - Learns structural collaborative signals

2. **Joint Optimization Phase**:
   - End-to-end training of all components
   - Multi-objective loss: local + global + alignment + orthogonality

## Loss Functions

### Local Loss (Sequential BPR)
```
L_local = -log(σ(score(pos_item) - score(neg_item)))
```

### Global Loss (Graph BPR)
```
L_global = -log(σ(score_graph(pos_item) - score_graph(neg_item)))
```

### Alignment Loss (Barlow Twins)
```
L_align = Σ(1 - C_ii)² + λ Σ C_ij²
```
where C is the cross-correlation matrix between sequential and graph embeddings.

### Orthogonality Loss
```
L_ortho = Σ(C_neg_ii)²
```
Pushes sequential embeddings away from disinterest embeddings.

### Total Loss
```
L = w_local * L_local + w_global * L_global + w_align * L_align + w_ortho * L_ortho
```

## Evaluation Metrics

- **NDCG@10**: Normalized Discounted Cumulative Gain at 10
- **Recall@10**: Hit rate at 10 recommendations

## Datasets

### Amazon Beauty
- ~730K users, ~207K items, ~6.6M interactions
- 5-core subset (each user/item has ≥5 interactions)

### Amazon Books
- ~777K users, ~495K items, ~9.5M interactions
- 5-core subset

## References

1. Gimranov et al. "Cross-Representation Knowledge Transfer for Improved Sequential Recommendations." KDD 2026.
2. Liu et al. "Pone-GNN: Integrating Positive and Negative Feedback in Graph Neural Networks for Recommender Systems." ACM RecSys 2025.

## License

MIT License
