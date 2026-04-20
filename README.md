# CREATE-PONE: Joint Training Recommender with Signed Graph Learning

CREATE-PONE is the CREATE++ paper's joint training variant that unifies **temporal sequence modeling** (from CREATE) with **explicit positive/negative feedback learning** (from Pone-GNN).

## Overview

The key innovation of CREATE-PONE is learning complementary representations from:
1. **Sequential patterns** - temporal dynamics in user behavior
2. **Signed feedback** - explicit positive (like) and negative (dislike) ratings

While ensuring the two encoders learn **complementary, not redundant** features via Barlow Twins redundancy reduction.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREATE-PONE Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User History: [Item₁, Item₂, Item₃, ..., Itemₜ]               │
│               │                                                 │
│      ┌────────┴────────┐                                        │
│      ▼                 ▼                                        │
│ ┌───────────┐    ┌──────────────┐                              │
│ │  SASRec   │    │   PoneGNN    │                              │
│ │  Encoder  │    │   Encoder    │                              │
│ │           │    │              │                              │
│ │ - Item    │    │ - Dual Emb:  │                              │
│ │   Embed   │    │   pos_embed  │                              │
│ │ - Pos     │    │   neg_embed  │                              │
│ │   Embed   │    │ - Signed     │                              │
│ │ - Transformer│ │   Edges      │                              │
│ │ - Causal  │    │              │                              │
│ │   Attn    │    │              │                              │
│ └─────┬─────┘    └──────┬───────┘                              │
│       │ u_seq           │ u_pos, u_neg                         │
│       └────────┬────────┘                                       │
│                │                                                │
│        ┌───────┴───────┐                                        │
│        │ Fusion Module │ (Concat + Project)                     │
│        └───────┬───────┘                                        │
│                │                                                │
│                ▼                                                │
│        ┌───────────────┐                                        │
│        │ Output Layer  │ → Item Scores                          │
│        └───────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
```

## Model Architecture

### 1. Sequential Encoder (SASRec)

Captures temporal patterns in user behavior sequences using causal self-attention.

| Component | Description |
|-----------|-------------|
| **Item Embedding** | Maps item IDs to dense vectors (dim=64) |
| **Position Embedding** | Encodes sequence position (recent items get different positions) |
| **Transformer Encoder** | Self-attention with causal mask (can only attend to past items) |
| **Layer Norm + Dropout** | Stabilization and regularization |

**Design Choices:**
- Uses **causal self-attention** (not bidirectional) - future items shouldn't influence past representations
- Last valid embedding extracted as user representation
- Feedforward dimension = 256 (4× embedding_dim for transformer expressiveness)

### 2. Graph Encoder (PoneGNN)

Learns from explicit user-item interactions with signed feedback (positive/negative ratings).

#### Dual Embeddings

| Embedding | Purpose | Learned From |
|-----------|---------|--------------|
| `user_pos`, `item_pos` | "What users like" | Positive interactions (rating > 3.5) |
| `user_neg`, `item_neg` | "What users dislike" | Negative interactions (rating < 3.5) |

**Why dual embeddings?** Standard GNNs treat all edges the same. PoneGNN recognizes that *disliking* something is fundamentally different from *not liking* it. Negative feedback provides a distinct signal: "avoid these items."

#### Graph Convolution (LightGINConv)

Layer-wise propagation with separate positive/negative edge handling:

```
α = 1 / (num_layers + 1)  # Aggregation weight

# For each layer l:
pos_emb^(l+1) = α × initial_emb + Σ(α × aggregated_from_pos_neighbors)
neg_emb^(l+1) = α × initial_emb + Σ(α × aggregated_from_neg_neighbors)
```

**Edge Construction:**
- **Positive edges:** rating > 3.5 (bidirectional: user ↔ item)
- **Negative edges:** rating < 3.5 (bidirectional: user ↔ item)

### 3. Fusion Module

**Default Strategy: Concatenation Fusion**

```python
fused = Concat(u_seq, u_pos)      # [batch, 2*dim]
fused = Linear(fused)              # [batch, dim]
scores = fused @ item_embeddings.T
```

Alternative strategies: `sum`, `gate`, `mlp`

---

## Loss Functions

CREATE-PONE optimizes a **multi-task loss with 5 components**:

| # | Loss | Formula | Purpose | When Applied |
|---|------|---------|---------|--------------|
| 1 | Sequential Loss | `CE(seq_scores, labels)` | Learn next-item prediction | Every batch |
| 2 | Dual-Feedback Loss | `L_pos + L_neg` | Learn from ratings | Every batch |
| 3 | Barlow Twins Loss | `Σ(1-C_ii)² + λΣC_ij²` | Reduce redundancy | Every batch |
| 4 | Orthogonal Loss | `mean(cos²(u_pos, u_neg))` | Decorrelate pos/neg | Every batch |
| 5 | Contrastive Loss | `InfoNCE` | Cross-space alignment | Every 10 epochs |

### Loss 1: Sequential Loss (SASRec Cross-Entropy)

```
L_seq = CrossEntropy(sequential_scores, next_item_labels)
```

**Purpose:** Learn temporal patterns - predict what item comes next in the sequence.

**Why Cross-Entropy?** Standard next-item prediction objective - maximizes probability of the correct next item.

---

### Loss 2: Dual-Feedback Loss (Positive + Negative BPR)

```
# Positive BPR: rank positive items higher than negatives
L_pos = -log(σ(u_pos · i_pos - u_pos · n_pos))

# Negative BPR: actively push away negative items
L_neg = -log(σ(u_neg · n_neg - u_neg · i_neg))

L_dual = L_pos + L_neg
```

**Purpose:** Learn separately from positive and negative ratings.

**Key Insight:**
- **Positive BPR:** "Make positive items score higher than random negatives"
- **Negative BPR:** "Make negative items score LOWER than random negatives" (inverted!)

This is fundamentally different from standard BPR - negative feedback *actively pushes* unwanted items away.

---

### Loss 3: Barlow Twins Alignment Loss

```
# Normalize embeddings to zero mean, unit variance
z_seq_norm  = (z_seq - mean(z_seq)) / std(z_seq)
z_graph_norm = (z_graph - mean(z_graph)) / std(z_graph)

# Cross-correlation matrix
C = z_seq_norm.T @ z_graph_norm / batch_size

# Loss: diagonal → 1, off-diagonal → 0
L_barlow = Σ_i (1 - C_ii)² + λ_barlow × Σ_i Σ_{j≠i} C_ij²
```

**Purpose:** Redundancy reduction between encoders.

**Why needed?** Without this loss, SASRec and PoneGNN might learn the same features. Barlow Twins forces:
- Corresponding features align (`C_ii → 1`)
- Different features decorrelate (`C_ij → 0`)

**Result:** Encoders learn *complementary* representations.

---

### Loss 4: Orthogonal Loss

```
similarity = cosine(u_pos, u_neg) = (u_pos · u_neg) / (||u_pos|| × ||u_neg||)

L_ortho = mean(similarity²)
```

**Purpose:** Force positive and negative embedding spaces to be independent.

**Why?** If `u_pos` and `u_neg` are correlated, they're not learning distinct signals. Orthogonality ensures "what users like" and "what users dislike" are independent dimensions.

---

### Loss 5: Contrastive Loss (InfoNCE-style)

```
# Normalize embeddings
u_pos_norm = normalize(u_pos)
i_pos_norm = normalize(i_pos)
u_neg_norm = normalize(u_neg)
i_neg_norm = normalize(i_neg)

# Similarities
pos_sim = u_pos_norm · i_pos_norm    # Should be HIGH
neg_sim = u_neg_norm · i_neg_norm    # Should be HIGH (in negative space)

# InfoNCE loss
L_contrast = -log( exp(pos_sim/τ) / (exp(pos_sim/τ) + exp(neg_sim/τ)) )
```

**Purpose:** Cross-space alignment between positive and negative embedding spaces.

**Applied every 10 epochs** to prevent over-regularization.

**Why periodic?** Continuous contrastive loss can overwhelm other objectives. Periodic application provides alignment "checkpoints" without dominating training.

---

## Training Design (Two-Stage)

### Stage 1: PoneGNN Pre-training (100-150 epochs)

```
L_stage1 = L_dual_feedback + L_ortho + L_contrast (every 10 epochs)
```

**What trains:** Only PoneGNN graph encoder (dual embeddings)

**Why pre-train?**
- Graph encoder needs to learn good dual embeddings before fusion
- Joint training from scratch can be unstable
- Pre-training provides a warm start for the graph encoder

**Training details:**
- Uses rating-based negative sampling (k=40 negatives per positive item)
- Builds positive/negative edge indices from ratings
- Optimizer: AdamW with warmup + cosine annealing

---

### Stage 2: Joint Training (200-300 epochs)

```
L_stage2 = L_seq + L_fused
         + λ_dual × L_dual_feedback
         + λ_barlow × L_barlow
         + λ_ortho × L_ortho
         + L_contrast (every 10 epochs)
```

**What trains:** Full CREATE-PONE model (SASRec + PoneGNN + Fusion)

**Key steps:**
1. Load pre-trained PoneGNN weights
2. Train SASRec + PoneGNN jointly with fusion
3. Gradient clipping (`max_norm=1.0`) for stability
4. CosineAnnealingLR scheduler with warmup

**Default Loss Weights:**

| Weight | Default | Role |
|--------|---------|------|
| λ_dual | 1.0 | Dual-feedback importance |
| λ_barlow | 0.01 | Redundancy reduction strength |
| λ_ortho | 0.01 | Orthogonal regularization |
| λ_contrast | 0.1 | Cross-space alignment |

---

## Key Design Insights

### 1. Signed Graph Learning

Unlike standard GNNs (all edges treated equally), PoneGNN maintains separate embeddings for positive/negative feedback:

| Aspect | Standard GNN | PoneGNN |
|--------|--------------|---------|
| Edges | All same | Signed (+/-) |
| Embeddings | Single | Dual (pos/neg) |
| Negative signal | Absent | Active push-away |

### 2. Redundancy Reduction

Barlow Twins loss ensures SASRec and PoneGNN learn *complementary* features:

| Without Barlow | With Barlow |
|----------------|-------------|
| Both encoders may learn same preferences | Sequential captures temporal patterns |
| Redundant representations | Graph captures rating-based preferences |

### 3. Contrastive Alignment (Periodic)

InfoNCE loss aligns positive/negative spaces only every 10 epochs:
- Prevents over-regularization
- Allows encoders to develop distinct representations before alignment

### 4. Orthogonal Constraint

Forces `u_pos ⊥ u_neg`:
- Ensures the two embedding spaces capture independent signals
- "What users like" and "what users dislike" are not just opposites

---

## Comparison with Parent Papers

| Aspect | CREATE | Pone-GNN | CREATE-PONE (Joint) |
|--------|--------|----------|---------------------|
| Sequential | SASRec encoder | ❌ | SASRec encoder |
| Graph | ❌ | LightGCN signed | PoneGNN dual embeddings |
| Dual Embeddings | ❌ | ✅ | ✅ |
| Fusion | ❌ | ❌ | Concat + Project |
| Barlow Twins | ❌ | ❌ | Redundancy reduction |
| Training | End-to-end | Two-stage | Two-stage (pretrain + joint) |

**CREATE-PONE = CREATE's sequential modeling + Pone-GNN's signed graph learning + Barlow Twins for complementarity**

---

## Usage

### Pre-train PoneGNN only (Stage 1)

```bash
python train_kaggle.py --dataset books --mode pretrain \
    --pretrain_epochs 150 --warmup_epochs 20
```

### Full CREATE-PONE (Stage 1 + Stage 2)

```bash
python train_kaggle.py --dataset books --model create_plus_plus --mode joint \
    --pretrain_epochs 150 --num_epochs 300 --warmup_epochs 20 \
    --embedding_dim 64 --sasrec_layers 2 --ponegnn_layers 2 \
    --barlow_weight 0.01 --orthogonal_weight 0.01 --contrastive_weight 0.1
```

### SASRec baseline (sequential only)

```bash
python train_kaggle.py --dataset books --model sasrec --mode sequential_only
```

---

## References

- **CREATE Paper:** Contrastive Representation learning with Adaptive Temporal Encoding
- **Pone-GNN Paper:** Positive and Negative feedback GNN for signed graph learning
- **Barlow Twins:** "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"

## Directory Structure

```
CREATE-/
├── README.md                 # This file - architecture & loss overview
├── IMPLEMENTATION.md         # Implementation guide
├── train_kaggle.py           # Kaggle-friendly training script
├── models/
│   ├── encoders/
│   │   ├── graph_encoder.py      # PoneGNN encoder
│   │   └── sequential_encoder.py # SASRec encoder
│   └── fusion/
│       └── joint_fusion.py       # CREATE-PONE joint model
├── dataset_loaders/
│   ├── base_dataset.py
│   ├── books_dataset.py
│   ├── beauty_dataset.py
│   └── collators.py
├── Pone-GNN/               # Reference Pone-GNN implementation
└── CREATE/                 # Reference CREATE implementation
```
