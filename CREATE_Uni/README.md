# CREATE-Uni: Unified Graph and Sequence Model for Sequential Recommendation

CREATE-Uni is the hypergraph variant of the **CREATE++** framework, combining ideas from **UniGNN (2021)** and **CREATE (2026)**. It replaces the standard bipartite GNN encoder with a UniGNN-based hypergraph encoder to capture higher-order multi-way collaborative signals with 1-GWL expressive power, strictly exceeding the topological limits of pairwise GNNs like LightGCN.

## Overview

CREATE-Uni combines two branches with multi-task alignment (no explicit fusion):

1. **UniGNN-style hypergraph encoder** — Captures higher-order collaborative filtering signals from session-based hyperedges using hypergraph neural networks (default: UniGCN). Produces graph user embeddings `g_u` and graph item embeddings `g_i`.
2. **Transformer-based sequence encoder** — Models sequential patterns using SASRec (default) or BERT4Rec. Receives graph-enriched item embeddings as input per Eq. 20: `x_k = g_{i_k} + p_k`, and outputs user representation `h_u`.

**Key design (per CREATE++ paper):**
- **Prediction (Eq. 21):** `scores = h_u @ g_i.T` — dot product of transformer user representation with graph item embeddings (no fusion layer)
- **Alignment (Eq. 22):** Barlow Twins loss between `h_u` (sequence view) and `g_u` (graph view) directly
- **Inference:** `ŷ_{u,i} = h_u^T · g_i`

## Implementation Audit and Fixes

The implementation was audited against `CREATE++.pdf`, `2021-UniGNN.pdf`, and the official `CREATE` code. The following issues were fixed:

- Item-token indexing is now consistent: raw item ids remain `0..N-1` for graph scoring and labels, while sequence tokens are shifted to `1..N` so `0` is reserved for PAD and `N+1` for BERT4Rec MASK.
- SASRec training now follows CREATE's full-softmax supervision pattern: every valid position predicts its next item, instead of supervising only the last event in each sequence.
- BERT4Rec training now uses masked `input_ids` correctly, with raw item ids as prediction labels and a dedicated evaluation-time last-position mask.
- The graph BPR branch now consumes explicit flattened positive `(user, item)` pairs from the batch instead of reusing the sequential labels in a shape-inconsistent way.
- Session hyperedges are now built from fixed-width user-local time bins with unique items per session, matching the set-valued hyperedge definition in CREATE-Uni.
- The UniGNN encoder now layer-averages graph outputs, matching Eq. (18), and the UniGAT path now preserves the requested embedding dimension across heads.

## Exact Current Formulation

After the fixes, the code is doing the following.

### 1. Hypergraph construction

For each user `u`, interactions are sorted by timestamp and partitioned into fixed-width bins of length `session_length`, anchored at that user's first training interaction. Each non-empty session bin forms one hyperedge:

`e_(u,r) = {u} ∪ H_(u,r)`

where `H_(u,r)` is the set of unique items clicked/purchased in that bin.

### 2. Graph encoder

Let `X^(0)` be the concatenated trainable user/item embeddings. A UniGNN layer computes

- hyperedge states: `h_e = phi_1({x_j : j in e})`
- node updates: `x_v^(l+1) = phi_2(x_v^(l), {h_e : e in E_v})`

using the selected UniGNN operator (`UniGCN`, `UniGIN`, `UniSAGE`, or `UniGAT`). The final graph embedding is the average over graph layers:

`g_v = (1 / K) * sum_(l=0)^(K-1) x_v^(l+1)`

### 3. Sequence encoder

- SASRec mode: the input sequence uses graph-enriched item tokens `x_k = g_(i_k) + p_k`, causal attention, reverse positional ids, and full-position supervision. During training, every valid position predicts the next raw item id. During evaluation, only the final valid state is used.
- BERT4Rec mode: the input sequence uses graph-enriched item tokens with MLM masking. During training, only masked positions are scored against the raw item vocabulary. During evaluation, a final MASK token is appended and only that position is scored.

### 4. Prediction and losses

- Local loss:
  - SASRec train: `L_local = CE(H_flat G^T, y_next_flat)`
  - SASRec eval: `scores = h_last G^T`
  - BERT4Rec train: `L_local = CE(H_masked G^T, y_masked)`
  - BERT4Rec eval: `scores = h_mask_last G^T`
- Global loss:
  - For every positive pair `(u, i)` emitted by the collator, sample one random negative `j != i`
  - Optimize `L_global = - mean log sigma(g_u^T g_i - g_u^T g_j)`
- Alignment loss:
  - Project graph-user and sequence-user embeddings and apply Barlow Twins
  - SASRec aligns the last valid sequential state with `g_u`
  - BERT4Rec aligns the per-user mean masked-position state with `g_u`

### 5. Gradient flow by objective

- Warm-up phase: only `L_global` is active, so only the graph encoder learns.
- Joint phase:
  - `L_local` updates the sequence encoder and graph item embeddings.
  - `L_global` updates the graph encoder only.
  - `L_align` updates both the graph-user branch and the sequence branch, plus the projectors.

This is now much closer to the CREATE-Uni theory, with one practical approximation left intentionally in place: the global BPR sampler excludes the current positive item, but does not explicitly filter every historical positive for that user.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CREATE-Uni                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐    g_i (Eq.20)   ┌──────────────────────┐  │
│  │   Graph Encoder     │ ────────────────► │  Sequence Encoder    │  │
│  │   (UniGNN)          │  enriched item    │  (SASRec/BERT4Rec)   │  │
│  │                     │  embeddings       │                      │  │
│  │  - UniGCN (default) │  x_k = g_ik + p_k│  - Transformer       │  │
│  │  - UniGIN           │                   │  - Position Emb      │  │
│  │  - UniSAGE          │                   │  - Causal Mask       │  │
│  │  - UniGAT           │                   │                      │  │
│  └──────┬───────┬──────┘                   └──────────┬───────────┘  │
│         │       │                                     │              │
│      g_u│    g_i│                                  h_u│              │
│         │       │                                     │              │
│         │       │    ┌──────────────────────┐          │              │
│         │       └───►│  Prediction (Eq.21)  │◄─────────┘              │
│         │            │  scores = h_u @ g_i.T│                        │
│         │            └──────────────────────┘                        │
│         │                                     h_u                    │
│         │            ┌──────────────────────┐  │                     │
│         └───────────►│  Alignment (Eq.22)   │◄─┘                     │
│                      │  Barlow Twins(g_u,h_u)│                       │
│                      └──────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
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
│   ├── graph_encoder.py        # Hypergraph encoder using UniGNN
│   ├── sequence_encoder.py     # SASRec/BERT4Rec sequence encoders
│   └── create_uni.py           # Main CREATE-Uni model
├── data.py                     # Data loaders and collators
├── loss.py                     # Loss functions (Local, Global, Barlow Twins)
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

### Full Example with Session Hypergraph

```bash
python -m CREATE_Uni.train \
    --dataset beauty \
    --num_epochs 30 \
    --warmup_epochs 10 \
    --session_length 86400
```

### Dataset Selection

```bash
# Amazon Beauty
python -m CREATE_Uni.train --dataset beauty

# Amazon Office Products
python -m CREATE_Uni.train --dataset office_products
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

# Disable graph encoder (sequence-only ablation)
python -m CREATE_Uni.train \
    --dataset beauty \
    --no_graph
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
    --barlow_twins_coef 0.01 \
    --barlow_lambda 0.1
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
| `--dataset` | str | beauty | Dataset to use: `beauty`, `office_products` |
| `--data_dir` | str | ./data | Directory containing dataset CSV files |
| `--output_dir` | str | ./outputs | Directory for outputs (logs, metrics, plots) |

### Model Architecture Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--seq_encoder` | str | sasrec | Sequence encoder: `sasrec` or `bert4rec` |
| `--embedding_dim` | int | 64 | Embedding dimension |
| `--graph_conv_type` | str | UniGCN | Graph convolution: `UniGCN`, `UniGIN`, `UniSAGE`, `UniGAT` |
| `--graph_n_layers` | int | 2 | Number of graph convolution layers |
| `--graph_heads` | int | 8 | Number of attention heads for UniGAT |
| `--seq_n_layers` | int | 2 | Number of transformer layers |
| `--seq_heads` | int | 4 | Number of transformer attention heads |
| `--max_sequence_length` | int | 50 | Maximum sequence length |

| `--use_graph` / `--no_graph` | flag | True | Use graph encoder |
| `--use_sequence` / `--no_sequence` | flag | True | Use sequence encoder |
| `--session_length` | int | 86400 | Session length in seconds for hypergraph construction |

### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | int | 256 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--weight_decay` | float | 1e-5 | Weight decay |
| `--num_epochs` | int | 100 | Maximum number of epochs |
| `--warmup_epochs` | int | 5 | Number of warmup epochs (global loss only) |
| `--early_stopping_rounds` | int | 10 | Early stopping patience |
| `--seed` | int | 42 | Random seed |
| `--device` | str | cuda | Device to run on |
| `--num_workers` | int | 4 | Number of data loading workers |

### Loss Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--local_coef` | float | 1.0 | Local objective weight (cross-entropy for next-item prediction) |
| `--global_coef` | float | 0.1 | Global objective weight (BPR pairwise ranking) |
| `--barlow_twins_coef` | float | 0.01 | Barlow Twins alignment objective weight |
| `--barlow_lambda` | float | 0.1 | Barlow Twins redundancy reduction lambda |

### Evaluation Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--eval_k` | int+ | [5, 10, 20] | K values for evaluation metrics |
| `--log_interval` | int | 1 | Log every N epochs |

## Training Schedule

CREATE-Uni uses a **two-phase training schedule** following the CREATE framework:

### Phase 1: Warmup (epochs < `warmup_epochs`)

- **Active losses:** Global (BPR) only
- **Purpose:** Allow the UniGNN hypergraph encoder to organize structural node embeddings into a sound neighbourhood topology before the sequential encoder begins optimizing

### Phase 2: Joint Training (epochs ≥ `warmup_epochs`)

- **Active losses:** Global (BPR) + Local (Cross-Entropy) + Barlow Twins (Alignment)
- **Purpose:** The sequence encoder fully optimizes for next-item prediction while anchored by the pre-configured spatial constraints of the hypergraph, with the Barlow Twins objective bridging the two views

### Loss Functions

The three objectives used in CREATE-Uni (no fusion involved):

1. **Local Objective (Eq. 21, L_local):** Cross-entropy loss using `h_u^T · g_{i_{t+1}}` — the dot product of the transformer user representation with graph item embeddings for next-item prediction
2. **Global Objective (Eq. 19, L_global):** BPR-style pairwise ranking loss applied to the hypergraph encoder embeddings: `g_u^T · g_i`
3. **Barlow Twins / Alignment Objective (Eq. 22, L_align):** Redundancy reduction loss that forces the cross-correlation matrix between projected `h_u` (sequence view) and `g_u` (graph view) toward the identity matrix, transferring structural knowledge to the sequential encoder without fusing them

Total loss (joint phase): `L = λ_local × L_local + λ_global × L_global + λ_align × L_align`

### Model Checkpointing

- **Best model:** Saved as `best_model.pt` whenever validation NDCG@10 improves
- **Latest model:** Saved as `latest_model.pt` after every epoch
- **Final model:** Saved as `final_model.pt` at training completion

## Output

After training, the output directory contains:

- `config.json` — Training configuration and best metrics
- `metrics.json` — Per-epoch metrics for all splits
- `summary.json` — Training summary (total epochs, best epoch)
- `training.log` — Detailed training log
- `*.png` — Learning curve plots for each metric (hr_at_5.png, ndcg_at_10.png, etc.)

## Model Architecture Details

### UniGNN Hypergraph Encoder

The graph encoder builds a **session-based hypergraph** from user-item interactions:

- **Vertices:** Users (0 to num_users-1) and Items (num_users to num_users+num_items-1)
- **Hyperedges:** Each session window (configured via `--session_length`, default 24 hours) groups a user with all items they interacted with during that window, forming a multi-node hyperedge `e = {u} ∪ H_{u,session}`

This session-based construction is critical for achieving **1-GWL expressiveness**, which strictly exceeds the topological limits of pairwise bipartite GNNs like LightGCN.

Four convolution variants are implemented:

1. **UniGCN** (default): GCN-style propagation with degree normalization
2. **UniGIN**: GIN-style with learnable epsilon parameter
3. **UniSAGE**: GraphSAGE-style with skip connections
4. **UniGAT**: GAT-style with attention mechanism

### Sequence Encoder

Two sequence encoding options:

1. **SASRec** (default): Causal transformer for autoregressive next-item prediction
2. **BERT4Rec**: Bidirectional transformer with MLM objective

### No Fusion — Paper-Faithful Design

Unlike models that explicitly fuse graph and sequence representations, CREATE-Uni keeps the two branches separate per the paper. They interact only through:

1. **Input enrichment (Eq. 20):** Graph-learned item embeddings `g_i` are fed as input to the transformer (replacing raw embedding lookups), combined with positional embeddings: `x_k = g_{i_k} + p_k`
2. **Alignment loss (Eq. 22):** Barlow Twins aligns the user representations from both views (`h_u` ↔ `g_u`) without merging them
3. **Prediction (Eq. 21):** Scores are computed as `h_u @ g_i.T` — the transformer user representation dotted with graph item embeddings

## Dataset Format

The dataset loaders automatically download Amazon Reviews 2023 5-core JSON Lines data. Expected keys within JSON line:
- `reviewerID`: User identifier
- `asin`: Item identifier  
- `overall`: Interaction rating
- `unixReviewTime`: Interaction timestamp

The loaders automatically:
1. Download from Amazon Reviews 2023 (5-core subsets)
2. Build user/item vocabularies
3. Create train/validation/test splits (leave-last-out)
4. Preserve timestamps for session-based hypergraph construction
5. Save processed data as `.pt` files

## Evaluation Metrics

Standard sequential recommendation metrics:

- **Hit Rate@K (HR@K)**: Fraction of users with correct item in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K
- **Precision@K**: Precision at K
- **Recall@K**: Recall at K
- **MAP@K**: Mean Average Precision at K

Early stopping is based on **NDCG@10** on the validation set.


## References

### Papers

- **CREATE++**: [PDF in repository] — Proposes unified graph+sequence model with signed and higher-order extensions
- **UniGNN (2021)**: "UniGNN: A Unified Framework for Graph and Hypergraph Neural Networks" — Provides hypergraph convolution operations
- **CREATE (2026)**: "CREATE: Cross-Representation Alignment for Training Enhancement" — Introduces cross-representation alignment for sequential recommendation
- **SASRec**: "Self-Attentive Sequential Recommendation" — Transformer-based sequential recommendation
- **BERT4Rec**: "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations" — Bidirectional transformer with MLM

### Codebases

- Official UniGNN: https://github.com/malllabiisc/UniGNN
- Official CREATE: [Repository in project]

## License

This implementation is for educational and research purposes.
