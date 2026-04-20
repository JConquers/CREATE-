# CREATE-Pone Implementation Plan and Technical Documentation

## 1. Scope

This implementation covers the CREATE-Pone signed-graph variant described in CREATE++.

Implemented:
- Signed dual-branch graph encoder with separate positive and negative propagation
- Sequential encoder for next-item prediction (SASRec-style causal Transformer)
- CREATE-Pone losses: local, global (dual feedback + contrastive), and alignment
- Two-stage optimization schedule: warm-up then joint training
- Dataset integration for Amazon Beauty and Amazon Books via existing dataset_loaders package
- Command-line training entrypoint with configurable hyperparameters

Not implemented in this module:
- CREATE-Uni hypergraph variant

## 2. Repository Layout

- run_create_pone.py
  - CLI entrypoint
- create_pone/trainer.py
  - argument parser, data setup, model construction, training loop, checkpointing
- create_pone/losses.py
  - CREATE-Pone objective terms and combined loss
- create_pone/models/
  - signed_gnn.py: signed dual-branch GNN
  - sequence_encoder.py: causal Transformer sequential encoder
  - create_pone.py: integrated model wiring sequence + signed branches
- create_pone/dataset/
  - loader.py: wraps dataset_loaders factory for beauty/books
  - sequence_dataset.py: user sequence dataset and batch collator
  - signed_graph.py: positive/negative graph construction and mini-batch triplet sampler

## 3. End-to-End Implementation Plan

### Step A: Load and normalize dataset artifacts

File: create_pone/dataset/loader.py

1. Use dataset_loaders.get_dataset for beauty/books.
2. Load train/validation/test DataFrames.
3. Build cleaned user sequences with length >= 2.
4. Return a DatasetBundle containing:
   - train_df, val_df, test_df
   - user_sequences
   - num_users, num_items

### Step B: Build sequential training batches

File: create_pone/dataset/sequence_dataset.py

1. UserSequenceDataset stores per-user interaction sequences.
2. NextItemCollator creates autoregressive next-item samples:
   - context = sequence[:-1]
   - target = sequence[1:]
3. Produces padded tensors:
   - input_ids
   - target_ids (with -100 on padded positions)
   - attention_mask
   - user_ids

### Step C: Build signed graph and triplets

File: create_pone/dataset/signed_graph.py

1. Split training interactions by thresholds:
   - positive: rating >= pos_threshold
   - negative: rating <= neg_threshold
2. Build normalized bipartite sparse adjacency for both graphs.
3. Store:
   - pos_adj, neg_adj
   - pos_deg_inv, neg_deg_inv
4. SignedTripleSampler samples mini-batch triplets:
   - positive branch triplets from positive interactions
   - negative branch triplets from negative interactions

### Step D: Build model branches

Files: create_pone/models/signed_gnn.py, create_pone/models/sequence_encoder.py, create_pone/models/create_pone.py

1. SignedDualGNN
   - Separate interest and disinterest embeddings for users/items.
   - Separate propagation on positive and negative graphs.
   - Uses the Eq. (7) style residual form:
     h_new = A_norm h + (1 + epsilon) D_inv h
   - Uses K-state averaging aligned with Eq. (8):
     - num_layers = K states
     - K-1 propagation steps

2. SequenceEncoder
   - Positional embedding + causal TransformerEncoder.
   - Causal mask prevents future-token leakage.
   - Returns per-position hidden states and last valid hidden state.

3. CreatePoneModel
   - Runs signed GNN first.
   - During joint stage, builds sequence input from interest item embeddings.
   - Computes token-level sequence logits over all items.

### Step E: Optimize with two training phases

File: create_pone/trainer.py

1. Warm-up phase (epoch < warmup_epochs)
   - Run signed GNN branch only.
   - Optimize global objective only.

2. Joint phase (epoch >= warmup_epochs)
   - Run signed GNN + sequential branch.
   - Optimize full objective:
     L = L_local + w_global L_global + w_align L_align

3. Save outputs:
   - checkpoint: create_pone_<dataset>_<timestamp>.pt
   - history: create_pone_<dataset>_<timestamp>_history.json

## 4. Full Mathematical Formulation (LaTeX)

This section writes the implementation math explicitly for sequence updates, signed-GNN updates, and all losses.

### 4.1 Sequential Encoder Updates

For a user sequence

$$
S_u = (i_1, i_2, \dots, i_T),
$$

define interest-item embedding lookup $E(\cdot)$ and positional embedding $p_t$. The sequence token at position $t$ is

$$
x_t = E(i_t) + p_t.
$$

Stacking tokens gives

$$
\mathbf{X}_u = [x_1; x_2; \dots; x_T] \in \mathbb{R}^{T \times d}.
$$

Using a causal Transformer encoder,

$$
\mathbf{H}_u = \mathrm{Transformer}_{\mathrm{causal}}(\mathbf{X}_u), \qquad \mathbf{H}_u \in \mathbb{R}^{T \times d}.
$$

The next-item logits at each valid timestep are

$$
\mathbf{L}_u = \mathbf{H}_u \mathbf{E}_{\text{all}}^\top,
$$

where $\mathbf{E}_{\text{all}} \in \mathbb{R}^{|\mathcal{I}| \times d}$ is the matrix of interest item embeddings. The user representation used for ranking is the last valid hidden state

$$
h_u = \mathbf{H}_u[t_u^{\text{last}}].
$$

### 4.2 Signed GNN Updates

There are two branches on disjoint signed graphs: positive graph $G^+$ and negative graph $G^-$. Let

$$
\mathbf{A}^+, \mathbf{A}^- \in \mathbb{R}^{N \times N}
$$

be normalized sparse adjacency matrices and

$$
\mathbf{D}_+^{-1}, \mathbf{D}_-^{-1} \in \mathbb{R}^{N \times N}
$$

the inverse-degree diagonal matrices.

Initial node states are separate for interest and disinterest:

$$
\mathbf{H}^{+,0} = [\mathbf{U}^+; \mathbf{I}^+], \qquad \mathbf{H}^{-,0} = [\mathbf{U}^-; \mathbf{I}^-].
$$

For $k = 0, \dots, K-2$, propagation is

$$
\mathbf{H}^{+,k+1} = \mathbf{A}^+ \mathbf{H}^{+,k} + (1 + \epsilon_k^+)\,\mathbf{D}_+^{-1}\mathbf{H}^{+,k},
$$

$$
\mathbf{H}^{-,k+1} = \mathbf{A}^- \mathbf{H}^{-,k} + (1 + \epsilon_k^-)\,\mathbf{D}_-^{-1}\mathbf{H}^{-,k}.
$$

The implementation keeps exactly $K$ states and performs $K-1$ propagation steps, then layer-averages:

$$
\mathbf{Z} = \frac{1}{K}\sum_{k=0}^{K-1}\mathbf{H}^{+,k},
\qquad
\mathbf{V} = \frac{1}{K}\sum_{k=0}^{K-1}\mathbf{H}^{-,k}.
$$

After splitting rows by node type:

$$
\mathbf{Z} = [\mathbf{Z}_u; \mathbf{Z}_i], \qquad \mathbf{V} = [\mathbf{V}_u; \mathbf{V}_i].
$$

### 4.3 Local Sequence Loss $L_{\text{local}}$

Let $\mathcal{M}$ be valid (non-padding) training positions from the batch, with target item $y_{u,t}$. Then full-softmax cross-entropy is

$$
L_{\text{local}} = -\frac{1}{|\mathcal{M}|}\sum_{(u,t)\in\mathcal{M}}
\log\frac{\exp\big(\ell_{u,t,y_{u,t}}\big)}{\sum_{j\in\mathcal{I}}\exp\big(\ell_{u,t,j}\big)}.
$$

### 4.4 Dual Feedback-Aware Loss $L_{\text{DF}}$

With positive triplets $B_p$ and negative triplets $B_n$:

Positive branch:

$$
L_{\text{DF}}^{+} = -\frac{1}{|B_p|}\sum_{(u,i,j)\in B_p}
\log\sigma\big(z_u^\top z_i - z_u^\top z_j\big).
$$

Negative branch (with scaling coefficient $b$ = neg_branch_scale):

$$
L_{\text{DF}}^{-} = -\frac{1}{|B_n|}\sum_{(u,i,j)\in B_n}
\log\sigma\big(v_u^\top v_j - b\,v_u^\top v_i\big).
$$

Combined dual-feedback term:

$$
L_{\text{DF}} = L_{\text{DF}}^{+} + L_{\text{DF}}^{-}.
$$

### 4.5 Contrastive Loss $L_{\text{CL}}$

The implementation matches users present in both branches in the same mini-batch. For each matched user $u$ with a positive pair $(z_u, z_{i_u})$ and negative pair $(v_u, v_{i'_u})$:

$$
s_u^{+} = \frac{z_u^\top z_{i_u}}{\tau},
\qquad
s_u^{-} = \frac{v_u^\top v_{i'_u}}{\tau}.
$$

Then

$$
L_{\text{CL}} = -\frac{1}{|\mathcal{U}_{\cap}|}\sum_{u\in\mathcal{U}_{\cap}}
\log\frac{\exp(s_u^{+})}{\exp(s_u^{+}) + \exp(s_u^{-})}.
$$

Global loss is

$$
L_{\text{global}} = L_{\text{DF}} + L_{\text{CL}}.
$$

### 4.6 Alignment Loss $L_{\text{align}}$

For a batch of size $B$, let

$$
\mathbf{H} \in \mathbb{R}^{B\times d}, \quad
\mathbf{Z}_b \in \mathbb{R}^{B\times d}, \quad
\mathbf{V}_b \in \mathbb{R}^{B\times d}
$$

be sequence user embeddings, interest user embeddings, and disinterest user embeddings for batch users. After per-dimension standardization,

$$
\widetilde{\mathbf{H}},\ \widetilde{\mathbf{Z}}_b,\ \widetilde{\mathbf{V}}_b,
$$

cross-correlation matrices are

$$
\mathbf{C}_{hz} = \frac{1}{B}\tilde{\mathbf{H}}^\top\tilde{\mathbf{Z}}_b,
\qquad
\mathbf{C}_{hv} = \frac{1}{B}\tilde{\mathbf{H}}^\top\tilde{\mathbf{V}}_b.
$$

Alignment objective:

$$
L_{\text{align}} =
\sum_{m}(1 - C_{hz,mm})^2
+ \lambda\sum_{m\neq n}C_{hz,mn}^2
+ \mu\sum_{m,n}C_{hv,mn}^2.
$$

### 4.7 Total Objective and Two-Phase Schedule

Warm-up phase ($e < E_{\text{warmup}}$):

$$
L_{\text{total}} = L_{\text{global}}.
$$

Joint phase ($e \ge E_{\text{warmup}}$):

$$
L_{\text{total}} = L_{\text{local}} + w_{\text{global}}L_{\text{global}} + w_{\text{align}}L_{\text{align}}.
$$

## 5. Sequential Encoder Details

File: create_pone/models/sequence_encoder.py

- Input:
  - item embeddings from interest item table
  - attention mask
- Processing:
  - add positional embeddings
  - layer norm + dropout
  - causal Transformer encoding
- Outputs:
  - sequence hidden states $\mathbf{H}_u$
  - last hidden state $h_u$

## 6. Signed GNN Details

File: create_pone/models/signed_gnn.py

- Two disjoint branches:
  - interest branch on positive graph
  - disinterest branch on negative graph
- Separate embedding tables:
  - user_interest_embedding, item_interest_embedding
  - user_disinterest_embedding, item_disinterest_embedding
- Layer behavior:
  - sparse normalized message passing
  - residual self-term with learnable $\epsilon_k^{+}, \epsilon_k^{-}$
  - optional dropout
- Output matrices:
  - $\mathbf{Z}_u, \mathbf{V}_u, \mathbf{Z}_i, \mathbf{V}_i$

## 7. How to Train on Beauty or Books

### Install dependencies

Run from repository root:

python -m pip install -r CREATE/reqs.txt

### Train on Amazon Beauty

python run_create_pone.py --dataset beauty --data-dir ./data --epochs 30 --warmup-epochs 5

### Train on Amazon Books

python run_create_pone.py --dataset books --data-dir ./data --epochs 30 --warmup-epochs 5

### Useful optional overrides

- --device cuda
- --embedding-dim 64
- --gnn-layers 2
- --transformer-layers 2
- --w-global 0.6 --w-align 0.1
- --pos-threshold 4.0 --neg-threshold 3.0

## 8. Verification Status

Verified in current workspace:

1. Syntax/compile check passed for all CREATE-Pone Python modules via py_compile.
2. Editor diagnostics reported no static errors for run_create_pone.py and create_pone package.

Runtime note:

- Running CLI help currently fails until torch is installed in the active environment:
  ModuleNotFoundError: No module named torch

After dependency installation, the training commands above are ready to run.
