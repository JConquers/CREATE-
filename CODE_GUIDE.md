# CREATE++ Code Guide

## Overview

CREATE++ is a hybrid sequential recommendation model combining:
- **SASRec**: Self-attentive sequential pattern learning
- **PoneGNN**: Signed graph neural network for positive/negative feedback

## Architecture

```
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
```

## Directory Structure

```
CREATE-/
├── train_kaggle.py          # Main training script (Kaggle-ready)
├── models/
│   ├── encoders/
│   │   ├── sequential_encoder.py   # SASRec encoder
│   │   └── graph_encoder.py        # PoneGNN encoder
│   └── fusion/
│       └── joint_fusion.py         # Fusion module + CREATE++ model
├── dataset_loaders/
│   ├── base_dataset.py      # Base dataset class
│   ├── books_dataset.py     # Amazon Books loader
│   └── beauty_dataset.py    # Amazon Beauty loader
└── CODE_GUIDE.md            # This file
```

## Loss Components

CREATE++ optimizes a joint loss with 5 components:

| Loss | Purpose | Formula |
|------|---------|---------|
| **Sequential Loss** | Learn temporal patterns | Cross-Entropy on next-item prediction |
| **Dual-Feedback Loss** | Learn from ratings | BPR for pos/neg items separately |
| **Barlow Twins Loss** | Align encoders | Cross-correlation → Identity |
| **Orthogonal Loss** | Decorrelate pos/neg | Penalize cosine similarity |
| **Contrastive Loss** | Cross-space alignment | InfoNCE (every 10 epochs) |

**Total Loss:**
```
L_total = L_seq + L_fused + L_dual + λ_barlow·L_barlow + λ_ortho·L_ortho + L_contrastive
```

## Training Strategy (Two-Stage)

### Stage 1: Pre-train PoneGNN
- Train only the graph encoder on signed graph edges
- Positive edges: rating > 3.5
- Negative edges: rating < 3.5
- Contrastive loss aligns pos/neg embedding spaces

### Stage 2: Joint Training
- SASRec learns sequential patterns
- PoneGNN refines graph representations
- Fusion module combines both signals
- All 5 loss components active

## Key Files

### `train_kaggle.py`
Main training orchestration:
- `NegativeSampler`: BPR negative sampling (popularity-based)
- `RatingBasedSampler`: Rating-based negative sampling for PoneGNN
- `build_dataloaders()`: Creates train/val DataLoaders
- `evaluate()`: Computes HR@K and NDCG@K
- `train_ponegnn()`: Stage 1 pre-training loop
- `train_joint()`: Stage 2 joint training loop
- `train_create_plus_plus()`: Main entry point

### `models/encoders/sequential_encoder.py`
SASRec implementation:
- Item + position embeddings
- Transformer encoder layers with causal mask
- Output: sequence embeddings + item scores

### `models/encoders/graph_encoder.py`
PoneGNN implementation:
- LightGCN-style convolution with signed edges
- Dual embeddings (positive/negative spaces)
- Contrastive loss for space alignment

### `models/fusion/joint_fusion.py`
Fusion strategies:
- `concat`: Concatenate + project (default)
- `sum`: Element-wise sum
- `gate`: Learned gating mechanism
- `mlp`: MLP fusion

## Usage

### Quick Test Run
```bash
%run train_kaggle.py --dataset beauty \
    --pretrain_epochs 10 --num_epochs 20 \
    --embedding_dim 32 --batch_size 128 --gpu 0
```

### Full Training
```bash
%run train_kaggle.py --dataset books \
    --pretrain_epochs 50 --num_epochs 100 \
    --embedding_dim 64 --sasrec_layers 2 --ponegnn_layers 2 \
    --batch_size 256 --lr 1e-3 --gpu 0
```

### Hyperparameter Guide

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `embedding_dim` | 32-128 | Larger = more capacity, slower |
| `sasrec_layers` | 1-4 | More layers = deeper sequential modeling |
| `sasrec_heads` | 2-8 | More heads = richer attention |
| `ponegnn_layers` | 1-3 | More layers = more graph propagation |
| `fusion_type` | concat/sum/gate/mlp | concat works best generally |
| `alpha` | 0.3-0.7 | Balance: sequential vs graph loss |
| `contrastive_weight` | 0.01-0.5 | Weight for contrastive alignment |
| `lr` | 1e-4 to 1e-3 | Higher = faster but less stable |

## Output Files

After training, the `output_dir` contains:
```
outputs/
└── create++_beauty_20260421_143022/
    ├── ponegnn_pretrain.pt      # Best Stage 1 checkpoint
    ├── joint_training_best.pt   # Best Stage 2 checkpoint
    └── create_plus_plus_final.pt # Final model with config
```

## Metrics

- **HR@K (Hit Rate)**: Fraction of users where ground truth is in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain (accounts for rank)

Higher is better for both metrics.

## Common Issues

### "item_id not in index"
Fixed: Dataset loaders now handle `parent_asin` → `item_id` mapping for Amazon 2023 format.

### Slow pre-training
Fixed: Batched negative sampling instead of per-sample tensor creation.

### OOM on GPU
Reduce: `batch_size`, `embedding_dim`, or `max_sequence_length`.

## References

- SASRec: https://arxiv.org/abs/1808.09781
- PoneGNN: Signed Graph Neural Network for recommendation
- Barlow Twins: https://arxiv.org/abs/2103.03230
