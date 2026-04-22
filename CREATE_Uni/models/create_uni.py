"""
CREATE-Uni: Unified Graph and Sequence Model for Sequential Recommendation.

This model combines:
1. UniGNN-style graph convolutions for capturing collaborative filtering signals
2. Transformer-based sequence encoding for modeling sequential patterns
3. Fusion mechanism to combine graph and sequence representations
4. Multi-task learning with local, global, and contrastive objectives

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
    Projection head for contrastive learning.
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
                torch.nn.init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.BatchNorm1d):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class CREATEUni(nn.Module):
    """
    CREATE-Uni: Unified Graph and Sequence Model.

    Combines graph-based collaborative filtering with sequential pattern learning
    using a fusion mechanism and multi-task learning objectives.

    Architecture:
    - Graph branch: UniGNN encodes user-item bipartite graph for CF signals
    - Sequence branch: Transformer (SASRec/BERT4Rec) encodes item sequences
    - Fusion: Combines both representations via concat/sum/gating
    - Prediction: Scores items using fused user representation
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        # Graph encoder params
        graph_n_layers: int = 2,
        graph_conv_type: str = "UniSAGE",
        graph_heads: int = 8,
        graph_dropout: float = 0.1,
        graph_use_norm: bool = True,
        graph_first_agg: str = "mean",
        graph_second_agg: str = "sum",
        # Sequence encoder params
        seq_n_layers: int = 2,
        seq_heads: int = 4,
        seq_dim_feedforward: int = 128,
        seq_dropout: float = 0.1,
        max_sequence_length: int = 50,
        seq_encoder_type: str = "sasrec",  # "sasrec" or "bert4rec"
        # Fusion params
        fusion_type: str = "concat",  # "concat", "sum", "gate"
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
        self.fusion_type = fusion_type

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

        # Fusion mechanism
        fusion_input_dim = embedding_dim
        if use_graph and use_sequence:
            if fusion_type == "concat":
                fusion_input_dim = embedding_dim * 2
            elif fusion_type == "gate":
                self.gate = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.Sigmoid(),
                )

        # Prediction head - maps fused representation to embedding space
        self.prediction_head = nn.Linear(fusion_input_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Projectors for contrastive learning
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
        degV = getattr(self, "graph_degV", None)
        degE = getattr(self, "graph_degE", None)

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
    ) -> torch.Tensor:
        """
        Encode using sequence encoder.

        Args:
            item_sequence: Item IDs (batch_size, seq_len)
            attention_mask: Boolean mask (batch_size, seq_len)
            mlm_mask: MLM positions (optional, for BERT4Rec)

        Returns:
            seq_emb: Sequence embeddings (batch_size, D) or (num_masked, D) for BERT4Rec
        """
        if self.seq_encoder_type == "bert4rec":
            masked_emb, all_emb = self.seq_encoder(
                item_sequence, attention_mask, mlm_mask
            )
            return masked_emb, all_emb
        else:
            seq_emb = self.seq_encoder(item_sequence, attention_mask, return_last=True)
            return seq_emb

    def fuse_representations(
        self,
        graph_emb: torch.Tensor,
        seq_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse graph and sequence representations.

        Args:
            graph_emb: Graph-based embeddings (batch_size, D)
            seq_emb: Sequence-based embeddings (batch_size, D)

        Returns:
            fused_emb: Fused embeddings (batch_size, embedding_dim)
        """
        if self.fusion_type == "concat":
            fused = torch.cat([graph_emb, seq_emb], dim=-1)
        elif self.fusion_type == "sum":
            fused = graph_emb + seq_emb
        elif self.fusion_type == "gate":
            combined = torch.cat([graph_emb, seq_emb], dim=-1)
            gate = self.gate(combined)
            fused = gate * graph_emb + (1 - gate) * seq_emb
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Apply prediction head and normalization
        fused = self.prediction_head(fused)
        fused = self.layer_norm(fused)
        fused = F.gelu(fused)

        return fused

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_contrastive: bool = False,
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
            return_contrastive: Whether to return contrastive embeddings

        Returns:
            outputs: Dictionary with:
                - local_prediction: Item scores (batch_size, num_items) or (num_masked, num_items)
                - contrastive_fst_embeddings: Graph projections (for contrastive loss)
                - contrastive_snd_embeddings: Sequence projections (for contrastive loss)
        """
        outputs = {}

        user_ids = batch.get("user.ids")
        item_sequence = batch.get("item.ids")
        attention_mask = batch.get("mask")
        labels = batch.get("labels.ids")
        mlm_mask = batch.get("mlm_mask")

        batch_size = user_ids.shape[0] if user_ids is not None else item_sequence.shape[0]
        device = item_sequence.device if item_sequence is not None else user_ids.device

        # Graph encoding
        graph_user_emb = None
        all_item_emb = None
        if self.use_graph:
            all_user_emb, all_item_emb = self.encode_graph()
            if user_ids is not None:
                graph_user_emb = all_user_emb[user_ids]

        # Sequence encoding
        seq_emb = None
        all_seq_emb = None
        if self.use_sequence and item_sequence is not None:
            if self.seq_encoder_type == "bert4rec":
                mlm_emb, all_seq_emb = self.encode_sequence(
                    item_sequence, attention_mask, mlm_mask
                )
                seq_emb = mlm_emb  # (num_masked, D)
            else:
                seq_emb = self.encode_sequence(item_sequence, attention_mask)  # (batch_size, D)

        # Fusion and prediction
        if self.use_graph and self.use_sequence:
            if self.seq_encoder_type == "bert4rec":
                # BERT4Rec: seq_emb is (num_masked, D)
                # Need to expand graph_user_emb to match
                if mlm_mask is not None:
                    # Expand user embedding for each masked position
                    counts = mlm_mask.sum(dim=1).to(torch.int64)
                    graph_user_expanded = graph_user_emb.repeat_interleave(counts, dim=0)
                    fused = self.fuse_representations(graph_user_expanded, seq_emb)
                else:
                    fused = seq_emb
            else:
                # SASRec: seq_emb is (batch_size, D), graph_user_emb is (batch_size, D)
                fused = self.fuse_representations(graph_user_emb, seq_emb)

            # Compute scores against item embeddings
            scores = fused @ all_item_emb.T
            outputs["local_prediction"] = scores

            # Contrastive learning embeddings
            if return_contrastive:
                if self.seq_encoder_type == "bert4rec" and mlm_mask is not None:
                    # For BERT4Rec, average sequence embeddings per user
                    seq_emb_per_user = torch.zeros_like(graph_user_emb)
                    counts = mlm_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    for i in range(batch_size):
                        if mlm_mask[i].any():
                            seq_emb_per_user[i] = all_seq_emb[i][mlm_mask[i]].mean(dim=0)
                    graph_proj = self.graph_projector(graph_user_emb)
                    seq_proj = self.seq_projector(seq_emb_per_user)
                else:
                    graph_proj = self.graph_projector(graph_user_emb)
                    seq_proj = self.seq_projector(seq_emb)
                outputs["contrastive_fst_embeddings"] = graph_proj
                outputs["contrastive_snd_embeddings"] = seq_proj

        elif self.use_graph:
            # Graph-only mode
            scores = graph_user_emb @ all_item_emb.T
            outputs["local_prediction"] = scores

        elif self.use_sequence:
            # Sequence-only mode
            if self.seq_encoder_type == "bert4rec":
                scores = seq_emb @ self.output_item_embeddings.weight.T
            else:
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

            # Sequence encoding
            if self.use_sequence:
                seq_emb = self.encode_sequence(item_sequence, attention_mask)
            else:
                seq_emb = None

            # Fusion
            if self.use_graph and self.use_sequence:
                fused = self.fuse_representations(graph_user_emb, seq_emb)
            elif self.use_graph:
                fused = graph_user_emb
            else:
                fused = seq_emb

            # Compute scores
            if self.use_graph:
                scores = fused @ all_item_emb.T
            else:
                scores = fused @ self.output_item_embeddings.weight.T

            # Mask padding and special tokens
            scores[:, 0] = -torch.inf  # PAD
            if self.use_graph:
                scores[:, self.num_items + 1:] = -torch.inf  # MASK and beyond
            else:
                scores[:, self.num_items + 1:] = -torch.inf

            # Get top-k
            topk_scores, topk_indices = torch.topk(scores, k=topk, dim=-1)

            return topk_scores, topk_indices
