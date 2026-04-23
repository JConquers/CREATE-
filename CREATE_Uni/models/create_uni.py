"""
CREATE-Uni: Unified Graph and Sequence Model for Sequential Recommendation.

This model combines:
1. UniGNN-style graph convolutions for capturing collaborative filtering signals
2. Transformer-based sequence encoding for modeling sequential patterns
3. Multi-task learning with local, global, and Barlow Twins alignment objectives

Per the CREATE++ paper:
- Eq. 20: Sequence input is enriched with graph item embeddings: x_k = g_{i_k} + p_k
- Eq. 21: Local loss uses h_u (from transformer) dotted with g_i (from graph)
- Eq. 22: Alignment loss (Barlow Twins) between h_u and g_u directly (no fusion)
- Inference: ŷ_{u,i} = h_u^T * g_i

Based on CREATE++ paper ideas, combining UniGNN (2021) and CREATE (2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .graph_encoder import UniGNNEncoder, LightGCNStyleEncoder
from .sequence_encoder import SequentialEncoder, Bert4RecEncoder


class Projector(nn.Module):
    """
    Projection head for alignment learning.
    Maps embeddings to a common projection space.
    """

    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 64,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.proj_dim = proj_dim

        layers = []
        for i in range(num_layers):
            in_size = input_dim if i == 0 else proj_dim
            out_size = proj_dim
            layers.append(nn.Linear(in_size, out_size, bias=False))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(out_size, affine=False, eps=1e-8))
                layers.append(nn.ReLU(inplace=True))

        self.projector = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if hasattr(module, 'weight') and module.weight is not None:
                    torch.nn.init.xavier_uniform_(module.weight.data)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                if module.affine:
                    module.bias.data.zero_()
                    module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class CREATEUni(nn.Module):
    """
    CREATE-Uni: Unified Graph and Sequence Model.

    Combines graph-based collaborative filtering with sequential pattern learning
    via multi-task learning objectives (no explicit fusion of representations).

    Architecture:
    - Graph branch: UniGNN encodes user-item hypergraph → g_u (user), g_i (item)
    - Sequence branch: Transformer encodes item sequences using graph-enriched
      item embeddings (Eq. 20) → h_u (user representation)
    - Prediction: scores = h_u @ g_i.T (Eq. 21, no fusion)
    - Alignment: Barlow Twins between h_u and g_u (Eq. 22, no fusion)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        # Graph encoder params
        graph_n_layers: int = 2,
        graph_conv_type: str = "UniGCN",
        graph_heads: int = 8,
        graph_dropout: float = 0.1,
        graph_use_norm: bool = True,
        graph_first_agg: str = "mean", # edge_embedding from node embeddings 
        graph_second_agg: str = "sum", # node_embedding from edge embeddings
        # Sequence encoder params
        seq_n_layers: int = 2,
        seq_heads: int = 4,
        seq_dim_feedforward: int = 128,
        seq_dropout: float = 0.1,
        max_sequence_length: int = 50,
        seq_encoder_type: str = "sasrec",  # "sasrec" or "bert4rec"
        # Projection params
        proj_dim: int = 64,
        # Other
        use_graph: bool = True,
        use_sequence: bool = True,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_graph = use_graph
        self.use_sequence = use_sequence
        self.seq_encoder_type = seq_encoder_type


        # Graph Encoder
        if use_graph:
            self.graph_encoder = UniGNNEncoder(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=embedding_dim,
                n_layers=graph_n_layers,
                conv_type=graph_conv_type,
                heads=graph_heads,
                dropout=graph_dropout,
                use_norm=graph_use_norm,
                first_aggregate=graph_first_agg,
                second_aggregate=graph_second_agg,
            )
            # Store graph structure for message passing
            self.register_buffer("graph_vertex", torch.zeros(0, dtype=torch.long))
            self.register_buffer("graph_edges", torch.zeros(0, dtype=torch.long))

        # Sequence Encoder
        if use_sequence:
            if seq_encoder_type == "bert4rec":
                self.seq_encoder = Bert4RecEncoder(
                    num_items=num_items,
                    embedding_dim=embedding_dim,
                    num_heads=seq_heads,
                    num_layers=seq_n_layers,
                    dim_feedforward=seq_dim_feedforward,
                    dropout=seq_dropout,
                    max_sequence_length=max_sequence_length,
                )
            else:  # sasrec
                self.seq_encoder = SequentialEncoder(
                    num_items=num_items,
                    embedding_dim=embedding_dim,
                    num_heads=seq_heads,
                    num_layers=seq_n_layers,
                    dim_feedforward=seq_dim_feedforward,
                    dropout=seq_dropout,
                    max_sequence_length=max_sequence_length,
                )

        # No fusion mechanism: per the paper, prediction is h_u @ g_i.T
        # and alignment is Barlow Twins between h_u and g_u directly.

        # Projectors for Barlow Twins alignment
        if use_graph and use_sequence:
            self.graph_projector = Projector(embedding_dim, proj_dim)
            self.seq_projector = Projector(embedding_dim, proj_dim)

        # Item embeddings for prediction output
        if use_graph:
            # Use graph encoder's item embeddings for scoring
            self.output_item_embeddings = self.graph_encoder.item_embeddings
        elif use_sequence:
            self.output_item_embeddings = self.seq_encoder.item_embeddings
        else:
            raise ValueError("At least one of use_graph or use_sequence must be True")

    def set_graph_structure(
        self,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degV: torch.Tensor = None,
        degE: torch.Tensor = None,
    ):
        """
        Set the graph structure for message passing.

        Args:
            vertex: Row indices of incidence matrix (2*E,)
            edges: Column indices of incidence matrix (2*E,)
            degV: Vertex degree normalization (N,)
            degE: Edge degree normalization (E,)
        """
        self.register_buffer("graph_vertex", vertex)
        self.register_buffer("graph_edges", edges)
        if degV is not None:
            self.register_buffer("graph_degV", degV)
        if degE is not None:
            self.register_buffer("graph_degE", degE)

    def encode_graph(
        self,
        user_ids: torch.Tensor = None,
        item_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode using graph encoder.

        Returns:
            user_emb: User embeddings (num_users, D) or (batch_size, D) if user_ids provided
            item_emb: Item embeddings (num_items, D)
        """
        vertex = getattr(self, "graph_vertex", None)
        edges = getattr(self, "graph_edges", None)
        degV = getattr(self, "graph_degV", None)  # Note: stored as graph_degV buffer
        degE = getattr(self, "graph_degE", None)  # Note: stored as graph_degE buffer

        user_emb, item_emb = self.graph_encoder(vertex, edges, degE, degV)

        if user_ids is not None:
            user_emb = user_emb[user_ids]
        if item_ids is not None:
            item_emb = item_emb[item_ids]

        return user_emb, item_emb

    def encode_sequence(
        self,
        item_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        mlm_mask: torch.Tensor = None,
        graph_item_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode using sequence encoder.

        Per Eq. 20 of CREATE++: x_k = g_{i_k} + p_k
        When graph_item_emb is provided, the transformer receives graph-learned
        item embeddings instead of raw embedding table lookups.

        Args:
            item_sequence: Item IDs (batch_size, seq_len)
            attention_mask: Boolean mask (batch_size, seq_len)
            mlm_mask: MLM positions (optional, for BERT4Rec)
            graph_item_emb: Graph-learned item embeddings (num_items, D), optional.
                            If provided, used as transformer input per Eq. 20.

        Returns:
            seq_emb: Sequence embeddings (batch_size, D) or (num_masked, D) for BERT4Rec
        """
        # Build precomputed embeddings from graph if available
        precomputed = None
        if graph_item_emb is not None:
            # graph_item_emb: (num_items, D), indexed 0..num_items-1
            # item_sequence uses: 0=PAD, 1..num_items=items, num_items+1=MASK
            # Prepend zero row for PAD, append zero row for MASK
            pad_row = torch.zeros(1, graph_item_emb.shape[1], device=graph_item_emb.device)
            padded = torch.cat([pad_row, graph_item_emb, pad_row], dim=0)  # (num_items+2, D)
            precomputed = padded[item_sequence]  # (B, L, D)

        if self.seq_encoder_type == "bert4rec":
            masked_emb, all_emb = self.seq_encoder(
                item_sequence, attention_mask, mlm_mask,
                precomputed_emb=precomputed,
            )
            return masked_emb, all_emb
        else:
            seq_emb = self.seq_encoder(
                item_sequence, attention_mask, return_last=True,
                precomputed_emb=precomputed,
            )
            return seq_emb



    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_alignment: bool = False,  # True only during training Phase 2 to compute Barlow Twins (Eq. 22) projections
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CREATE-Uni.

        Args:
            batch: Dictionary containing:
                - user.ids: User IDs (batch_size,)
                - item.ids: Item sequences (batch_size, seq_len) or flattened
                - item.length: Sequence lengths (batch_size,)
                - mask: Attention mask (batch_size, seq_len)
                - labels.ids: Target items (batch_size,) or (num_masked,)
                - mlm_mask: MLM positions for BERT4Rec (optional)
            return_alignment: Whether to return alignment embeddings

        Returns:
            outputs: Dictionary with:
                - local_prediction: Item scores (batch_size, num_items) or (num_masked, num_items)
                - alignment_fst_embeddings: Graph projections (for alignment loss)
                - alignment_snd_embeddings: Sequence projections (for alignment loss)
        """
        outputs = {}

        user_ids = batch.get("user.ids")
        item_sequence = batch.get("item.ids")
        attention_mask = batch.get("mask")  # Boolean mask: 1 for valid items, 0 for padded positions (prevents attending to padding)
        labels = batch.get("labels.ids")
        mlm_mask = batch.get("mlm_mask")  # Boolean mask: 1 for items masked out for Masked Language Modeling (BERT4Rec target)

        batch_size = user_ids.shape[0] if user_ids is not None else item_sequence.shape[0]
        device = item_sequence.device if item_sequence is not None else user_ids.device

        # === 1. Graph Branch Encoding ===
        # Process the global hypergraph through the UniGNN encoder to get embeddings 
        # for ALL users and items that capture collaborative filtering signals.
        graph_user_emb = None
        all_item_emb = None
        if self.use_graph:
            all_user_emb, all_item_emb = self.encode_graph()
            if user_ids is not None:
                # Extract only the graph embeddings for the specific users in this current batch
                graph_user_emb = all_user_emb[user_ids]

        # === 2. Sequence Branch Encoding ===
        # Process the chronological sequence of items using Transformer-based models.
        seq_emb = None
        all_seq_emb = None
        if self.use_sequence and item_sequence is not None:
            if self.seq_encoder_type == "bert4rec":
                # For BERT4Rec (Masked Language Modeling): we extract embeddings only 
                # at the specifically masked positions to calculate the targeted MLM loss later.
                mlm_emb, all_seq_emb = self.encode_sequence(
                    item_sequence, attention_mask, mlm_mask,
                    # Pass the graph-learned item embeddings (all_item_emb) to the sequence encoder
                    # instead of using raw table lookups (Implementation of Eq. 20: x_k = g_{i_k} + p_k)
                    graph_item_emb=all_item_emb,  
                )
                seq_emb = mlm_emb  # Shape: (total_num_masked_items_in_batch, D)
            else:
                # For SASRec (Next-item prediction): we just need the final aggregated 
                # sequence embedding to predict the next user action.
                seq_emb = self.encode_sequence(
                    item_sequence, attention_mask,
                    graph_item_emb=all_item_emb,  # Eq. 20: x_k = g_{i_k} + p_k
                )  # Shape: (batch_size, D)

        # === 3. Prediction (No Fusion) ===
        # Per the paper, there is NO fusion of graph and sequence embeddings.
        # Prediction: scores = h_u @ g_i.T (Eq. 21)
        # Alignment:  Barlow Twins between h_u and g_u (Eq. 22)
        if self.use_graph and self.use_sequence:
            # Eq. 21: L_local uses h_u (seq_emb) dotted with g_i (all_item_emb)
            scores = seq_emb @ all_item_emb.T
            outputs["local_prediction"] = scores

            # Global objective (BPR): positive/negative scores from graph embeddings
            # Used during warmup epochs to pre-condition the graph encoder
            if user_ids is not None and self.use_graph:
                # Sample negative items for BPR loss
                neg_item_ids = torch.randint(
                    1, self.num_items + 1, (batch_size,), device=user_ids.device
                )
                pos_scores = (graph_user_emb * all_item_emb[user_ids]).sum(dim=1, keepdim=True)
                neg_scores = (graph_user_emb * all_item_emb[neg_item_ids]).sum(dim=1, keepdim=True)
                outputs["global_positive"] = pos_scores
                outputs["global_negative"] = neg_scores

            # Eq. 22: Barlow Twins alignment between h_u and g_u (no fusion)
            if return_alignment:
                if self.seq_encoder_type == "bert4rec" and mlm_mask is not None:
                    # For BERT4Rec, average the per-position sequence embeddings
                    # back to one embedding per user for alignment with g_u
                    # all_seq_emb: (batch_size, seq_len, D)
                    seq_emb_per_user = torch.zeros_like(graph_user_emb)
                    for i in range(batch_size):
                        user_mlm_mask = mlm_mask[i]
                        if user_mlm_mask.any():
                            # Average embeddings at masked positions for this user
                            seq_emb_per_user[i] = all_seq_emb[i][user_mlm_mask].mean(dim=0)
                        else:
                            # Fallback: use mean of all positions
                            seq_emb_per_user[i] = all_seq_emb[i][attention_mask[i]].mean(dim=0) if attention_mask[i].any() else all_seq_emb[i].mean(dim=0)
                    graph_proj = self.graph_projector(graph_user_emb)   # project g_u
                    seq_proj = self.seq_projector(seq_emb_per_user)     # project h_u
                else:
                    graph_proj = self.graph_projector(graph_user_emb)   # project g_u
                    seq_proj = self.seq_projector(seq_emb)              # project h_u
                outputs["alignment_fst_embeddings"] = graph_proj
                outputs["alignment_snd_embeddings"] = seq_proj

        elif self.use_graph:
            # Graph-only mode: scores = g_u @ g_i.T
            scores = graph_user_emb @ all_item_emb.T
            outputs["local_prediction"] = scores

        elif self.use_sequence:
            # Sequence-only mode: scores = h_u @ item_emb.T
            scores = seq_emb @ self.output_item_embeddings.weight.T
            outputs["local_prediction"] = scores

        return outputs

    def predict(
        self,
        user_ids: torch.Tensor,
        item_sequence: torch.Tensor,
        attention_mask: torch.Tensor,
        topk: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate top-k recommendations.

        Args:
            user_ids: User IDs (batch_size,)
            item_sequence: Item history (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            topk: Number of recommendations to return

        Returns:
            topk_scores: Top-k scores (batch_size, topk)
            topk_indices: Top-k item indices (batch_size, topk)
        """
        self.eval()

        with torch.no_grad():
            # Graph encoding
            if self.use_graph:
                all_user_emb, all_item_emb = self.encode_graph()
                graph_user_emb = all_user_emb[user_ids]
            else:
                graph_user_emb = None
                all_item_emb = None

            # Sequence encoding (with graph-learned item embeddings per Eq. 20)
            if self.use_sequence:
                seq_emb = self.encode_sequence(
                    item_sequence, attention_mask,
                    graph_item_emb=all_item_emb,
                )
            else:
                seq_emb = None

            # No fusion: prediction is h_u @ g_i.T per the paper
            if self.use_graph and self.use_sequence:
                # Eq. 21 / Inference: ŷ_{u,i} = h_u^T * g_i
                scores = seq_emb @ all_item_emb.T
            elif self.use_graph:
                scores = graph_user_emb @ all_item_emb.T
            else:
                scores = seq_emb @ self.output_item_embeddings.weight.T

            # Mask padding and special tokens
            scores[:, 0] = -torch.inf  # PAD
            if self.use_graph:
                scores[:, self.num_items + 1:] = -torch.inf  # MASK and beyond
            else:
                scores[:, self.num_items + 1:] = -torch.inf

            # Get top-k
            topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)

            return topk_scores, topk_indices
