# CREATE-Uni: Unified Graph and Sequence Model for Sequential Recommendation

CREATE-Uni is a unified graph and sequence model for sequential recommendation, combining ideas from **UniGNN (2021)** and **CREATE (2026)** papers as described in the **CREATE++** paper.

## Overview

CREATE-Uni combines three key components:

1. **UniGNN-style graph convolutions** - Captures collaborative filtering signals from user-item interactions using hypergraph neural networks
2. **Transformer-based sequence encoding** - Models sequential patterns using SASRec or BERT4Rec architectures
3. **Fusion mechanisms** - Combines graph and sequence representations through concatenation, summation, or gating

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        CREATE-Uni                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────┐           ┌──────────────────────┐    │
│  │   Graph Encoder     │           │  Sequence Encoder    │    │
│  │   (UniGNN)          │           │  (SASRec/BERT4Rec)   │    │
│  │                     │           │                      │    │
│  │  - UniGCN           │           │  - Transformer       │    │
│  │  - UniGIN           │           │  - Position Emb      │    │
│  │  - UniSAGE          │           │  - Causal Mask       │    │
│  │  - UniGAT           │           │                      │    │
│  └──────────┬──────────┘           └───────────┬──────────┘    │
│             │                                  │                │
│             │       ┌──────────────┐           │                │
│             └──────►│  Fusion      │◄──────────┘                │
│                     │  Layer       │                            │
│                     │  (concat/    │                            │
│                     │   sum/gate)  │                            │
│                     └──────┬───────┘                            │
│                            │                                    │
│                     ┌──────▼───────┐                            │
│                     │  Prediction  │                            │
│                     │  Head        │                            │
│                     └──────┬───────┘                            │
│                            │                                    │
│                     ┌──────▼───────┐                            │
│                     │  Item Scores │                            │
│                     └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- torch-scatter

### Install Dependencies

```bash
pip install torch torch-geometric torch-scatter
pip install pandas numpy matplotlib tqdm seaborn scipy
```

## Project Structure

```
CREATE_Uni/
├── __init__.py                 # Package initialization
├── models/
│   ├── __init__.py             # Model exports
│   ├── unignn_conv.py          # UniGNN convolution layers (UniGCN, UniGIN, UniSAGE, UniGAT)
│   ├── graph_encoder.py        # Graph encoder using UniGNN
│   ├── sequence_encoder.py     # SASRec/BERT4Rec sequence encoders
│   └── create_uni.py           # Main CREATE-Uni model
├── data.py                     # Data loaders and collators
├── loss.py                     # Loss functions (Local, Global, Contrastive, Barlow Twins)
├── metrics.py                  # Evaluation metrics (HR@K, NDCG@K, Precision@K, Recall@K, MAP@K)
├── utils.py                    # Training utilities (logger, seed fixing, train loop)
├── train.py                    # Main training script with CLI
└── README.md                   # This file
```

## Usage

### Basic Training

```bash
# Train on Amazon Beauty dataset with default settings
python -m CREATE_Uni.train \
    --dataset beauty \
    --data_dir ./data \
    --output_dir ./outputs
```

### Dataset Selection

```bash
# Amazon Beauty
python -m CREATE_Uni.train --dataset beauty

# Amazon Books
python -m CREATE_Uni.train --dataset books
```

### Model Configuration

```bash
# Use BERT4Rec sequence encoder instead of SASRec
python -m CREATE_Uni.train \
    --dataset beauty \
    --seq_encoder bert4rec

# Use UniGAT graph convolution with 8 attention heads
python -m CREATE_Uni.train \
    --dataset beauty \
    --graph_conv_type UniGAT \
    --graph_heads 8

# Change fusion mechanism
python -m CREATE_Uni.train \
    --dataset beauty \
    --fusion_type gate  # Options: concat, sum, gate
```

### Hyperparameters

```bash
python -m CREATE_Uni.train \
    --dataset beauty \
    --embedding_dim 128 \
    --graph_n_layers 3 \
    --seq_n_layers 4 \
    --seq_heads 8 \
    --max_sequence_length 100 \
    --batch_size 512 \
    --lr 0.0005 \
    --num_epochs 200 \
    --early_stopping_rounds 15
```

### Loss Weights

```bash
python -m CREATE_Uni.train \
    --dataset beauty \
    --local_coef 1.0 \
    --global_coef 0.1 \
    --contrastive_coef 0.01 \
    --contrastive_temperature 0.5
```

### Ablation Studies

```bash
# Graph-only model (no sequence)
python -m CREATE_Uni.train \
    --dataset beauty \
    --no_sequence

# Sequence-only model (no graph)
python -m CREATE_Uni.train \
    --dataset beauty \
    --no_graph

# Compare different graph convolutions
python -m CREATE_Uni.train --dataset beauty --graph_conv_type UniGCN
python -m CREATE_Uni.train --dataset beauty --graph_conv_type UniGIN
python -m CREATE_Uni.train --dataset beauty --graph_conv_type UniSAGE
python -m CREATE_Uni.train --dataset beauty --graph_conv_type UniGAT
```

## Command Line Arguments

### Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | beauty | Dataset to use: `beauty`, `books` |
| `--data_dir` | str | ./data | Directory containing dataset CSV files |
| `--output_dir` | str | ./outputs | Directory for outputs (logs, metrics, plots) |

### Model Architecture Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seq_encoder` | str | sasrec | Sequence encoder: `sasrec` or `bert4rec` |
| `--embedding_dim` | int | 64 | Embedding dimension |
| `--graph_conv_type` | str | UniSAGE | Graph convolution: `UniGCN`, `UniGIN`, `UniSAGE`, `UniGAT` |
| `--graph_n_layers` | int | 2 | Number of graph convolution layers |
| `--graph_heads` | int | 8 | Number of attention heads for UniGAT |
| `--seq_n_layers` | int | 2 | Number of transformer layers |
| `--seq_heads` | int | 4 | Number of transformer attention heads |
| `--max_sequence_length` | int | 50 | Maximum sequence length |
| `--fusion_type` | str | concat | Fusion mechanism: `concat`, `sum`, `gate` |
| `--use_graph` / `--no_graph` | flag | True | Use graph encoder |
| `--use_sequence` / `--no_sequence` | flag | True | Use sequence encoder |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 256 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--weight_decay` | float | 1e-5 | Weight decay |
| `--num_epochs` | int | 100 | Maximum number of epochs |
| `--early_stopping_rounds` | int | 10 | Early stopping patience |
| `--seed` | int | 42 | Random seed |
| `--device` | str | cuda | Device to run on |
| `--num_workers` | int | 4 | Number of data loading workers |

### Loss Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--local_coef` | float | 1.0 | Local objective weight (cross-entropy) |
| `--global_coef` | float | 0.1 | Global objective weight (BPR ranking) |
| `--contrastive_coef` | float | 0.01 | Contrastive objective weight (InfoNCE) |
| `--contrastive_temperature` | float | 1.0 | Contrastive loss temperature |

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval_k` | int+ | [5, 10, 20] | K values for evaluation metrics |
| `--log_interval` | int | 1 | Log every N epochs |

## Output

After training, the output directory contains:

- `config.json` - Training configuration and best metrics
- `metrics.json` - Per-epoch metrics for all splits
- `summary.json` - Training summary (total epochs, best epoch)
- `training.log` - Detailed training log
- `*.png` - Learning curve plots for each metric (hr_at_5.png, ndcg_at_10.png, etc.)

## Model Architecture Details

### UniGNN Graph Encoder

The graph encoder builds a bipartite hypergraph between users and items:

- **Vertices**: Users (0 to num_users-1) and Items (num_users to num_users+num_items-1)
- **Hyperedges**: Each user-item interaction forms a hyperedge connecting exactly 2 nodes

Four convolution variants are implemented:

1. **UniGCN**: GCN-style propagation with degree normalization
2. **UniGIN**: GIN-style with learnable epsilon parameter
3. **UniSAGE**: GraphSAGE-style with skip connections
4. **UniGAT**: GAT-style with attention mechanism

### Sequence Encoder

Two sequence encoding options:

1. **SASRec**: Causal transformer for autoregressive next-item prediction
2. **BERT4Rec**: Bidirectional transformer with MLM objective

### Fusion Mechanisms

Three fusion strategies:

1. **Concat**: Concatenate graph and sequence embeddings, then project
2. **Sum**: Element-wise addition of embeddings
3. **Gate**: Learnable gating mechanism to weight contributions

### Loss Functions

CREATE-Uni uses a multi-task learning objective:

1. **Local Objective**: Cross-entropy loss for next-item prediction
2. **Global Objective**: BPR-style pairwise ranking loss
3. **Contrastive Objective**: InfoNCE loss for graph-sequence alignment
4. **Barlow Twins** (optional): Redundancy reduction loss

Total loss: `L = λ_local * L_local + λ_global * L_global + λ_contrastive * L_contrastive`

## Dataset Format

The dataset loaders expect Amazon Reviews format with columns:
- `user_id`: User identifier
- `item_id` or `parent_asin`: Item identifier  
- `rating`: Interaction rating
- `timestamp`: Interaction timestamp

The loaders automatically:
1. Download from Amazon Reviews 2023 (5-core subsets)
2. Build user/item vocabularies
3. Create train/validation/test splits (leave-last-out)
4. Save processed data as `.pt` files

## Evaluation Metrics

Standard sequential recommendation metrics:

- **Hit Rate@K (HR@K)**: Fraction of users with correct item in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Precision@K**: Precision at K
- **Recall@K**: Recall at K
- **MAP@K**: Mean Average Precision at K

## Implementation Fixes

The following fixes have been applied to the initial implementation:

1. **Shape Mismatch Fix**: Fixed `unsqueeze(0).expand` issue in `models/create_uni.py` while using BERT4Rec by implementing proper sequence alignment with `repeat_interleave` utilizing MLM mask limits.
2. **TQDM Imports and Progress Descriptions**: Corrected the module import to `from tqdm import tqdm, trange`. Added specific descriptions to differentiate between "Warmup" and "Joint" training epochs.
3. **Model Checkpointing**: Added model checkpointing `latest_checkpoint.pt` for all epochs as per requirement, instead of just tracking the best performing epoch conditionally. 
4. **Warmup Epochs Command Line Interface**: Added missing `--warmup_epochs` argument inside `train.py` argument parser for fully customizable Warmup phases via CLI.

## References

### Papers

- **CREATE++**: [PDF in repository] - Proposes unified graph+sequence model
- **UniGNN (2021)**: "UniGNN: A Unified Framework for Graph and Hypergraph Neural Networks" - Provides hypergraph convolution operations
- **CREATE (2026)**: "CREATE: Contrastative and Reconstruction-based Embeddings for sequential recommendation" - Introduces contrastive learning for sequential recommendation
- **SASRec**: "Self-Attentive Sequential Recommendation" - Transformer-based sequential recommendation
- **BERT4Rec**: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations" - Bidirectional transformer with MLM

### Codebases

- Official UniGNN: https://github.com/malllabiisc/UniGNN
- Official CREATE: [Repository in project]

## License

This implementation is for educational and research purposes.
