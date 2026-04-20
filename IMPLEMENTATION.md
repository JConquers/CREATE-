# CREATE-PONE Implementation Guide

This guide explains how to implement the CREATE-PONE variant based on the CREATE++ paper. The implementation builds upon two base papers: **CREATE** (sequential modeling) and **Pone-GNN** (signed graph learning).

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Implementation](#component-implementation)
3. [Loss Functions - Mathematical Formulation](#loss-functions---mathematical-formulation)
4. [Training Strategy](#training-strategy)
5. [Kaggle Setup and Training Commands](#kaggle-setup-and-training-commands)

---

## Architecture Overview

CREATE-PONE consists of three main components:

```
┌─────────────────────────────────────────────────────────────┐
│                    CREATE-PONE Model                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: User sequence [i₁, i₂, ..., iₜ]                    │
│                                                             │
│         ┌─────────────┬─────────────┐                      │
│         │             │             │                      │
│         ▼             ▼             ▼                      │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐               │
│   │  SASRec   │ │  PoneGNN  │ │  Fusion   │               │
│   │  Encoder  │ │  Encoder  │ │  Module   │               │
│   │           │ │           │ │           │               │
│   │ u_seq     │ │ u_pos     │ │ fused     │               │
│   │           │ │ u_neg     │ │           │               │
│   └───────────┘ └───────────┘ └───────────┘               │
│         │             │             │                      │
│         └─────────────┴─────────────┘                      │
│                       │                                    │
│                       ▼                                    │
│              ┌─────────────────┐                          │
│              │  Output Layer   │ → Item Scores            │
│              └─────────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Implementation

### 1. Sequential Encoder (SASRec)

The SASRec encoder captures temporal patterns using causal self-attention.

#### Architecture

```python
class SASRecEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_sequence_length: int = 50,
    ):
        super().__init__()
        
        # Item and position embeddings
        self.item_embedding = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length + 1, embedding_dim)
        
        # Transformer encoder layers with causal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
```

#### Forward Pass

```python
def forward(self, item_sequences, mask):
    """
    Args:
        item_sequences: (batch, seq_len) - item IDs
        mask: (batch, seq_len) - boolean mask for valid positions
    
    Returns:
        dict with user_embedding and item_scores
    """
    batch_size, seq_len = item_sequences.shape
    
    # Get embeddings with scaling
    item_emb = self.item_embedding(item_sequences) * (self.embedding_dim ** 0.5)
    
    # Position embeddings (reverse order - recent items get lower indices)
    positions = torch.arange(seq_len - 1, -1, step=-1, device=self.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    pos_emb = self.position_embedding(positions)
    
    # Combine and normalize
    hidden = self.layer_norm(item_emb + pos_emb)
    hidden = self.dropout(hidden)
    
    # Create causal attention mask (lower triangular)
    attn_mask = torch.tril(torch.ones(seq_length, seq_length)).bool()
    attn_mask = ~attn_mask  # Invert: True means mask out
    
    # Transformer encoding
    transformer_output = self.transformer_encoder(
        hidden,
        mask=attn_mask,
        src_key_padding_mask=~mask if mask is not None else None,
    )
    
    # Extract last valid embedding as user representation
    lengths = mask.sum(dim=-1) - 1
    user_indices = lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
    user_embeddings = torch.gather(transformer_output, 1, user_indices).squeeze(1)
    
    # Compute item scores via dot product with all item embeddings
    item_scores = user_embeddings @ self.item_embedding.weight.T
    
    return {
        'user_embedding': user_embeddings,
        'item_scores': item_scores,
    }
```

#### Key Implementation Notes

1. **Causal Attention:** The transformer uses causal masking (lower triangular) so each position can only attend to past positions.

2. **Reverse Position Embeddings:** Recent items get lower position indices (following SASRec convention).

3. **Last Valid Embedding:** The user representation is extracted from the last valid (non-padded) position in the sequence.

---

### 2. Graph Encoder (PoneGNN)

The PoneGNN encoder learns dual embeddings for positive and negative feedback.

#### Architecture

```python
class PoneGNNEncoder(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        reg: float = 1e-4,
        temperature: float = 1.0,
        contrastive_weight: float = 0.1,
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Dual embeddings: positive and negative spaces
        self.user_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))
        self.user_neg_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))
        
        # Initialize with Xavier
        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(LightGINConv(embedding_dim, embedding_dim, first_aggr=True))
            else:
                self.conv_layers.append(LightGINConv(embedding_dim, embedding_dim, first_aggr=False))
```

#### LightGINConv Layer

```python
class LightGINConv(MessagePassing):
    """LightGCN-style convolution with normalized aggregation."""
    
    def __init__(self, in_channels, out_channels, first_aggr=True):
        super().__init__(aggr='add')
        self.first_aggr = first_aggr
        self.eps = nn.Parameter(torch.empty(1))
        self.eps.data.fill_(0.0)
    
    def forward(self, x, pos_edge_index, neg_edge_index):
        """
        Args:
            x: Tuple of (pos_embeddings, neg_embeddings)
            pos_edge_index: Positive edge indices (2, num_edges)
            neg_edge_index: Negative edge indices (2, num_edges)
        """
        def get_norm(node, edge_index):
            row, col = edge_index
            deg = degree(col, node.size(0), dtype=node.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm, deg_inv_sqrt
        
        def gin_norm(out, input_x, deg_inv_sqrt):
            norm_self = deg_inv_sqrt * deg_inv_sqrt
            norm_self = norm_self.unsqueeze(1).repeat(1, input_x.size(1))
            return out + (1 + self.eps) * norm_self * input_x
        
        norm_pos, deg_inv_sqrt_pos = get_norm(x[0], pos_edge_index)
        norm_neg, deg_inv_sqrt_neg = get_norm(x[1], neg_edge_index)
        
        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x[0], norm=norm_pos)
            out_neg = self.propagate(neg_edge_index, x=x[0], norm=norm_neg)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[0], deg_inv_sqrt_neg)
            return out_pos, out_neg
        else:
            out_pos = self.propagate(pos_edge_index, x=x[0], norm=norm_pos)
            out_neg = self.propagate(pos_edge_index, x=x[1], norm=norm_pos)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[1], deg_inv_sqrt_pos)
            return out_pos, out_neg
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
```

#### Forward Pass

```python
def forward(self, pos_edge_index, neg_edge_index):
    """
    Args:
        pos_edge_index: Positive edge indices (2, num_pos_edges)
        neg_edge_index: Negative edge indices (2, num_neg_edges)
    
    Returns:
        Tuple of (positive_embeddings, negative_embeddings)
    """
    alpha = 1.0 / (self.num_layers + 1)
    
    # Concatenate user and item embeddings
    ego_pos = torch.cat([self.user_embedding, self.item_embedding], dim=0)  # (U+I, dim)
    ego_neg = torch.cat([self.user_neg_embedding, self.item_neg_embedding], dim=0)
    
    # Initialize layer outputs
    pos_emb = ego_pos * alpha
    neg_emb = ego_neg * alpha
    
    ego_embeddings = (ego_pos, ego_neg)
    
    # Apply graph convolutions
    for i in range(self.num_layers):
        ego_embeddings = self.conv_layers[i](ego_embeddings, pos_edge_index, neg_edge_index)
        pos_emb = pos_emb + ego_embeddings[0] * alpha
        neg_emb = neg_emb + ego_embeddings[1] * alpha
    
    # Cache embeddings for loss computation
    self.pos_emb = pos_emb
    self.neg_emb = neg_emb
    
    return pos_emb, neg_emb
```

#### Key Implementation Notes

1. **Layer-wise Aggregation:** Each layer contributes `α = 1/(L+1)` to the final embedding, including the initial embedding.

2. **Separate Propagation:** Positive edges propagate through positive embeddings; negative edges propagate through negative embeddings.

3. **First Layer Special Case:** The first layer uses the initial embeddings for both spaces; subsequent layers use their respective spaces.

---

### 3. Fusion Module

The fusion module combines sequential and graph representations.

```python
class JointFusionModule(nn.Module):
    def __init__(self, embedding_dim, fusion_type='concat', hidden_dim=None):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            self.output_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        elif fusion_type == 'sum':
            self.output_proj = nn.Identity()
        elif fusion_type == 'gate':
            self.gate = nn.Linear(embedding_dim * 2, embedding_dim)
            self.output_proj = nn.Identity()
        elif fusion_type == 'mlp':
            hidden_dim = hidden_dim or embedding_dim * 2
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.output_proj = nn.Identity()
    
    def forward(self, sequential_emb, graph_emb):
        if self.fusion_type == 'concat':
            combined = torch.cat([sequential_emb, graph_emb], dim=-1)
            fused = self.output_proj(combined)
        elif self.fusion_type == 'sum':
            fused = sequential_emb + graph_emb
        elif self.fusion_type == 'gate':
            combined = torch.cat([sequential_emb, graph_emb], dim=-1)
            gate = torch.sigmoid(self.gate(combined))
            fused = gate * sequential_emb + (1 - gate) * graph_emb
        elif self.fusion_type == 'mlp':
            combined = torch.cat([sequential_emb, graph_emb], dim=-1)
            fused = self.mlp(combined)
        return fused
```

---

## Loss Functions - Mathematical Formulation

### Total Loss

```
L_total = L_seq + L_fused 
        + λ_dual × L_dual_feedback
        + λ_barlow × L_barlow
        + λ_ortho × L_ortho
        + L_contrast (every 10 epochs)
```

### 1. Sequential Loss (Cross-Entropy)

**Formula:**
```
L_seq = -Σ_i log(P(item_i | sequence))
      = CrossEntropy(seq_scores, labels)
```

**Implementation:**
```python
def sequential_loss(seq_scores, labels):
    """
    Args:
        seq_scores: (batch, num_items) - predicted scores
        labels: (batch,) - target item indices
    """
    return F.cross_entropy(seq_scores, labels)
```

---

### 2. Dual-Feedback Loss (Positive + Negative BPR)

**Mathematical Formulation:**

```
For positive feedback (rating > 3.5):
  L_pos = -E[log(σ(u_pos · i_pos - u_pos · n_pos))]

For negative feedback (rating < 3.5):
  L_neg = -E[log(σ(u_neg · n_neg - u_neg · i_neg))]

L_dual = L_pos + L_neg
```

**Key Insight:** The negative BPR is *inverted* - it learns to push negative items away, not just rank them lower.

**Implementation:**
```python
def dual_feedback_loss(users, pos_items, neg_items, pos_emb, neg_emb, num_users, num_items):
    """
    Args:
        users: (batch,) - user indices
        pos_items: (batch,) - positive item indices
        neg_items: (batch,) - negative sample indices
        pos_emb: (num_users+num_items, dim) - positive embeddings
        neg_emb: (num_users+num_items, dim) - negative embeddings
    """
    # Get embeddings
    u_pos = pos_emb[users]                              # (batch, dim)
    u_neg = neg_emb[users]
    i_pos = pos_emb[num_users + pos_items]              # (batch, dim)
    n_pos = pos_emb[num_users + neg_items]
    i_neg = neg_emb[num_users + pos_items]
    n_neg = neg_emb[num_users + neg_items]
    
    # Positive BPR: maximize u_pos · i_pos - u_pos · n_pos
    pos_scores = (u_pos * i_pos).sum(dim=1)             # (batch,)
    neg_scores = (u_pos * n_pos).sum(dim=1)
    pos_bpr = -F.logsigmoid(pos_scores - neg_scores).mean()
    
    # Negative BPR: maximize u_neg · n_neg - u_neg · i_neg
    neg_scores_neg = (u_neg * n_neg).sum(dim=1)
    pos_scores_neg = (u_neg * i_neg).sum(dim=1)
    neg_bpr = -F.logsigmoid(neg_scores_neg - pos_scores_neg).mean()
    
    return pos_bpr, neg_bpr, pos_bpr + neg_bpr
```

---

### 3. Barlow Twins Alignment Loss

**Mathematical Formulation:**

```
Given embeddings z_seq and z_graph:

1. Normalize to zero mean, unit variance:
   z_norm = (z - μ) / σ
   where μ = mean(z, dim=0), σ = std(z, dim=0)

2. Cross-correlation matrix:
   C = (z_seq_norm)^T @ z_graph_norm / batch_size
   C_ij = correlation between feature i (seq) and feature j (graph)

3. Loss:
   L_barlow = Σ_i (1 - C_ii)² + λ × Σ_i Σ_{j≠i} C_ij²
   
   - On-diagonal: Forces C_ii → 1 (alignment)
   - Off-diagonal: Forces C_ij → 0 for i≠j (redundancy reduction)
```

**Implementation:**
```python
def barlow_twins_loss(z_seq, z_graph, barlow_weight=0.01):
    """
    Args:
        z_seq: (batch, dim) - sequential embeddings
        z_graph: (batch, dim) - graph embeddings
    """
    batch_size = z_seq.size(0)
    
    # Normalize (zero mean, unit variance)
    z_seq_norm = (z_seq - z_seq.mean(dim=0)) / (z_seq.std(dim=0) + 1e-8)
    z_graph_norm = (z_graph - z_graph.mean(dim=0)) / (z_graph.std(dim=0) + 1e-8)
    
    # Cross-correlation matrix
    correlation = torch.mm(z_seq_norm.t(), z_graph_norm) / batch_size
    
    # On-diagonal and off-diagonal terms
    diagonal = torch.diag(correlation)
    off_diagonal = correlation - torch.diag_embed(diagonal)
    
    on_diag_loss = ((1 - diagonal) ** 2).sum()
    off_diag_loss = (off_diagonal ** 2).sum()
    
    return on_diag_loss + barlow_weight * off_diag_loss
```

---

### 4. Orthogonal Loss

**Mathematical Formulation:**

```
L_ortho = E[cos²(u_pos, u_neg)]
        = E[(u_pos · u_neg / (||u_pos|| × ||u_neg||))²]

Goal: Minimize cosine similarity → embeddings become orthogonal
```

**Implementation:**
```python
def orthogonal_loss(pos_emb, neg_emb):
    """
    Args:
        pos_emb: (batch, dim) - positive embeddings
        neg_emb: (batch, dim) - negative embeddings
    """
    # L2 normalize
    pos_norm = F.normalize(pos_emb, dim=1)
    neg_norm = F.normalize(neg_emb, dim=1)
    
    # Cosine similarity
    similarity = (pos_norm * neg_norm).sum(dim=1)  # (batch,)
    
    # Mean squared cosine similarity
    return (similarity ** 2).mean()
```

---

### 5. Contrastive Loss (InfoNCE)

**Mathematical Formulation:**

```
Given normalized embeddings:

1. Compute similarities:
   pos_sim = u_pos_norm · i_pos_norm    (positive space alignment)
   neg_sim = u_neg_norm · i_neg_norm    (negative space alignment)

2. InfoNCE loss:
   L_contrast = -log( exp(pos_sim/τ) / (exp(pos_sim/τ) + exp(neg_sim/τ)) )

where τ is the temperature parameter.
```

**Implementation:**
```python
def contrastive_loss(users, pos_items, neg_items, pos_emb, neg_emb, 
                     num_users, num_items, temperature=1.0):
    """
    Args:
        users: (batch,) - user indices
        pos_items: (batch,) - positive item indices
        neg_items: (batch,) - negative sample indices
        pos_emb: (num_users+num_items, dim) - positive embeddings
        neg_emb: (num_users+num_items, dim) - negative embeddings
    """
    # Get embeddings
    u_pos = pos_emb[num_users:][users]
    u_neg = neg_emb[num_users:][users]
    i_pos = pos_emb[num_users:][pos_items]
    i_neg = neg_emb[num_users:][pos_items]
    
    # Normalize
    u_pos_norm = F.normalize(u_pos, dim=1)
    u_neg_norm = F.normalize(u_neg, dim=1)
    i_pos_norm = F.normalize(i_pos, dim=1)
    i_neg_norm = F.normalize(i_neg, dim=1)
    
    # Similarities
    pos_similarity = (u_pos_norm * i_pos_norm).sum(dim=1)
    neg_similarity = (u_neg_norm * i_neg_norm).sum(dim=1)
    
    # InfoNCE
    pos_exp = torch.exp(pos_similarity / temperature)
    neg_exp = torch.exp(neg_similarity / temperature)
    
    loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()
    
    return loss
```

---

## Training Strategy

### Two-Stage Training

CREATE-PONE uses a two-stage training approach for stability and better convergence.

### Stage 1: PoneGNN Pre-training

**Objective:** Train the graph encoder to learn good dual embeddings before fusion.

```
L_stage1 = L_dual_feedback + L_ortho + L_contrast (every 10 epochs)
```

**Training Loop:**
```python
def train_ponegnn(model, train_df, args):
    # Build graph edges from ratings
    data_p, data_n = build_graph_edges(train_df, num_users, num_items, device)
    
    # Create sampler
    sampler = RatingBasedSampler(train_df, k=40)
    
    # Optimizer with warmup
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(1, pretrain_epochs + 1):
        negatives = sampler.generate_negatives(epoch)
        
        for batch in dataloader:
            users, pos_items, neg_items = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            pos_emb, neg_emb = model(pos_edge_index, neg_edge_index)
            
            # Compute losses
            _, _, dual_loss = model.compute_dual_feedback_loss(users, pos_items, neg_items)
            ortho_loss = model.compute_orthogonal_loss(users)
            
            # Contrastive every 10 epochs
            if epoch % 10 == 1 or epoch == 1:
                contrast_loss = model.compute_contrastive_loss(users, pos_items, neg_items)
            else:
                contrast_loss = torch.tensor(0.0)
            
            # Total loss
            loss = dual_loss + ortho_loss + contrast_loss
            loss.backward()
            optimizer.step()
        
        scheduler.step()
```

### Stage 2: Joint Training

**Objective:** Train the full CREATE-PONE model with all components.

```
L_stage2 = L_seq + L_fused 
         + λ_dual × L_dual_feedback
         + λ_barlow × L_barlow
         + λ_ortho × L_ortho
         + L_contrast (every 10 epochs)
```

**Training Loop:**
```python
def train_joint(model, train_loader, val_loader, args, data_p, data_n):
    # Load pre-trained graph encoder weights
    model.graph_encoder.load_state_dict(pretrained_weights)
    
    # Negative sampler
    neg_sampler = NegativeSampler(dataset.num_items, dataset.user2items)
    
    # Optimizer with warmup
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model.train()
    best_hr = 0
    best_ndcg = 0
    
    for epoch in range(1, num_epochs + 1):
        for batch in train_loader:
            sequences, mask, labels, users = batch
            
            # Sample negatives
            neg_items = neg_sampler.sample(users, n_samples=1)
            
            optimizer.zero_grad()
            
            # Compute joint loss with all 5 components
            loss_dict = model.compute_joint_loss(
                item_sequences=sequences,
                mask=mask,
                labels=labels,
                pos_edge_index=data_p.edge_index,
                neg_edge_index=data_n.edge_index,
                negative_samples=neg_items,
                epoch=epoch,
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            # Gradient clipping for stability
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        if epoch % eval_every == 0:
            metrics = evaluate(model, val_loader)
```

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 64 | Embedding dimension |
| `sasrec_heads` | 4 | Attention heads in SASRec |
| `sasrec_layers` | 2 | Transformer layers |
| `ponegnn_layers` | 2 | Graph convolution layers |
| `batch_size` | 256 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `pretrain_epochs` | 50-150 | Stage 1 epochs |
| `num_epochs` | 100-300 | Stage 2 epochs |
| `dropout` | 0.1 | Dropout rate |
| `reg` | 1e-4 | L2 regularization |

### Loss Weights

| Weight | Default | Description |
|--------|---------|-------------|
| `dual_feedback_weight` | 1.0 | Dual-feedback importance |
| `barlow_weight` | 0.01 | Barlow Twins strength |
| `orthogonal_weight` | 0.01 | Orthogonal regularization |
| `contrastive_weight` | 0.1 | Contrastive loss weight |
| `temperature` | 1.0 | Contrastive temperature |

### Rating Threshold

The threshold for separating positive and negative feedback:
- **Positive:** rating > 3.5
- **Negative:** rating < 3.5

---

## Kaggle Setup and Training Commands

### Kaggle Notebook Setup

1. **Create a new Kaggle Notebook** and add the following datasets:
   - Amazon Books dataset (or Beauty dataset)
   - Upload the CREATE- codebase

2. **Install required dependencies** in the notebook:
```python
!pip install torch torch-geometric tqdm
```

3. **Add data files** to the `./data/` directory:
```
/data/
  books/
    train.csv
    val.csv
    test.csv
  beauty/
    train.csv
    val.csv
    test.csv
```

### Training Commands

#### Option 1: Full CREATE-PONE Training (Two-Stage)

```bash
# Complete two-stage training
python train_kaggle.py \
    --dataset books \
    --pretrain_epochs 150 \
    --num_epochs 300 \
    --embedding_dim 64 \
    --sasrec_heads 4 \
    --sasrec_layers 2 \
    --ponegnn_layers 2 \
    --fusion_type concat \
    --batch_size 256 \
    --lr 1e-3 \
    --barlow_weight 0.01 \
    --orthogonal_weight 0.01 \
    --contrastive_weight 0.1 \
    --top_k 10 \
    --eval_every 10 \
    --gpu 0 \
    --seed 42
```

#### Option 2: PoneGNN Pre-training Only (Stage 1)

```bash
# Pre-train graph encoder
python train_kaggle.py \
    --dataset books \
    --pretrain_epochs 150 \
    --embedding_dim 64 \
    --ponegnn_layers 2 \
    --batch_size 256 \
    --lr 1e-3 \
    --gpu 0
```

#### Option 3: Beauty Dataset

```bash
# Train on Beauty dataset
python train_kaggle.py \
    --dataset beauty \
    --pretrain_epochs 150 \
    --num_epochs 300 \
    --embedding_dim 64 \
    --batch_size 256 \
    --lr 1e-3 \
    --gpu 0
```

### Kaggle Notebook Example

```python
# In a Kaggle notebook cell:
%run train_kaggle.py --dataset books --num_epochs 100 --eval_every 5

# Or import and run programmatically:
from train_kaggle import train_create_plus_plus

model, dataset, metrics = train_create_plus_plus(
    dataset_name='books',
    data_dir='/kaggle/input/amazon-books/data',
    embedding_dim=64,
    pretrain_epochs=150,
    num_epochs=300,
    batch_size=256,
    lr=1e-3,
    gpu=0,
    seed=42,
)

print(f"Best HR@10: {metrics['hit_rate']:.4f}")
print(f"Best NDCG@10: {metrics['ndcg']:.4f}")
```

### Expected Output

```
============================================================
CREATE++ Pone Variant Training
============================================================

Loading dataset: books
Dataset statistics:
  num_users: 5000
  num_items: 10000
  ...

============================================================
Stage 1: Pre-training PoneGNN Graph Encoder
============================================================
Epoch   1/150 | Loss: 0.8234
Epoch  10/150 | Loss: 0.5123
...
Pre-training complete. Best loss: 0.4521

Creating CREATE++ model with pre-trained graph encoder...
Loaded pre-trained PoneGNN weights.

============================================================
Stage 2: Joint Training of SASRec + PoneGNN
============================================================
Epoch   1/300 | Loss: 1.2345 | HR@10: 0.1234 | NDCG@10: 0.0876
Epoch  10/300 | Loss: 0.8765 | HR@10: 0.2345 | NDCG@10: 0.1543
...
Joint training complete.
Best HR@10: 0.3456
Best NDCG@10: 0.2345

============================================================
CREATE++ Training Complete!
============================================================
```

---

## File Structure

```
CREATE-/
├── README.md                      # This file - architecture & loss overview
├── IMPLEMENTATION.md              # Implementation guide
├── train_kaggle.py                # Kaggle-friendly training script
├── models/
│   ├── encoders/
│   │   ├── sequential_encoder.py  # SASRec implementation
│   │   └── graph_encoder.py       # PoneGNN implementation
│   └── fusion/
│       └── joint_fusion.py        # CREATE-PONE joint model
├── dataset_loaders/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── books_dataset.py
│   ├── beauty_dataset.py
│   └── collators.py
├── Pone-GNN/                      # Reference Pone-GNN code
└── CREATE/                        # Reference CREATE code
```

---

## Troubleshooting

### Common Issues

1. **NaN Loss:** Reduce learning rate, check for division by zero in Barlow Twins normalization.

2. **Poor Convergence:** Increase warmup epochs, verify graph edges are correctly built.

3. **Memory Issues:** Reduce batch size, gradient accumulation can help.

4. **Overfitting:** Increase dropout, add L2 regularization, reduce model capacity.

### Debugging Tips

1. **Monitor loss components:** Print individual loss values to identify which component is problematic.

2. **Check embedding norms:** Ensure embeddings don't explode or vanish.

3. **Verify edge construction:** Positive edges should only contain ratings > 3.5.

4. **Gradient flow:** Use gradient hooks to verify all components receive gradients.
