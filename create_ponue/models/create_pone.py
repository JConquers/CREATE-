"""
CREATE-Pone: Cross-Representation Alignment with Signed Graph Encoder.

Combines ideas from:
- CREATE: Cross-representation knowledge transfer with alignment
- Pone-GNN: Signed graph learning with dual-branch message passing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .signed_encoder import SignedGraphEncoder
from .sequential_encoder import SequentialEncoder, AlignmentModule


class CREATEPone(nn.Module):
    """
    CREATE-Pone model for sequential recommendation.

    Architecture:
    1. Signed Graph Encoder: Learns interest/disinterest embeddings from positive/negative graphs
    2. Sequential Encoder: SASRec-style transformer for sequential patterns
    3. Alignment Module: Aligns sequential embeddings with graph interest embeddings,
       while maintaining orthogonality to disinterest embeddings
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_graph_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        # Loss weights
        local_weight: float = 1.0,
        global_weight: float = 0.1,
        align_weight: float = 0.1,
        ortho_weight: float = 0.1,
        # Hyperparameters
        reg_weight: float = 1e-4,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.align_weight = align_weight
        self.ortho_weight = ortho_weight
        self.reg_weight = reg_weight
        self.temperature = temperature

        # Graph encoder (signed)
        self.graph_encoder = SignedGraphEncoder(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=num_graph_layers,
            reg_weight=reg_weight,
        )

        # Sequential encoder
        self.sequential_encoder = SequentialEncoder(
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        # Alignment module
        self.alignment_module = AlignmentModule(
            embedding_dim=embedding_dim,
            lambda_param=0.1,
        )

        # Prediction head
        self.prediction_head = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        item_sequence,
        pos_edge_index,
        neg_edge_index,
        attention_mask=None,
        compute_alignment=True,
    ):
        """
        Forward pass.

        Args:
            item_sequence: (batch_size, seq_len) item IDs
            pos_edge_index: (2, E_pos) positive graph edges
            neg_edge_index: (2, E_neg) negative graph edges
            attention_mask: (batch_size, seq_len) boolean mask
            compute_alignment: Whether to compute alignment loss

        Returns:
            outputs: Dictionary with predictions and losses
        """
        # Graph encoder
        user_pos_emb, item_pos_emb, user_neg_emb, item_neg_emb = self.graph_encoder(
            pos_edge_index, neg_edge_index
        )

        # Sequential encoder
        seq_output, user_seq_emb = self.sequential_encoder(
            item_sequence, attention_mask
        )

        # Get graph embeddings for batch users/items
        batch_users = item_sequence[:, 0]  # Assume first item indicates user
        # For actual batch, we need user indices passed separately

        outputs = {
            "user_seq_emb": user_seq_emb,
            "item_pos_emb": item_pos_emb,
            "item_neg_emb": item_neg_emb,
            "seq_output": seq_output,
        }

        return outputs

    def compute_losses(
        self,
        user_seq_emb,
        user_pos_emb,
        user_neg_emb,
        item_pos_emb,
        item_neg_emb,
        pos_pairs,  # (user_idx, pos_item_idx)
        neg_pairs,  # (user_idx, neg_item_idx)
    ):
        """
        Compute multi-objective loss.

        Args:
            user_seq_emb: (B, D) sequential user embeddings
            user_pos_emb: (N_users, D) graph interest embeddings
            user_neg_emb: (N_users, D) graph disinterest embeddings
            item_pos_emb: (N_items, D) graph interest item embeddings
            item_neg_emb: (N_items, D) graph disinterest item embeddings
            pos_pairs: (B, 2) positive (user, item) indices
            neg_pairs: (B, 2) negative (user, item) indices

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}

        # 1. Local loss: Sequential next-item prediction (BPR-style)
        local_loss = self._compute_local_loss(
            user_seq_emb, item_pos_emb, pos_pairs, neg_pairs
        )
        loss_dict["local_loss"] = local_loss

        # 2. Global loss: Graph-based BPR
        global_loss = self._compute_global_loss(
            user_pos_emb, item_pos_emb, pos_pairs, neg_pairs
        )
        loss_dict["global_loss"] = global_loss

        # 3. Alignment loss: Align seq with graph interest
        # 4. Orthogonality loss: Push seq away from disinterest
        align_loss, ortho_loss = self._compute_alignment_loss(
            user_seq_emb, user_pos_emb, user_neg_emb, pos_pairs[:, 0]
        )
        loss_dict["align_loss"] = align_loss
        loss_dict["ortho_loss"] = ortho_loss

        # Total loss
        total_loss = (
            self.local_weight * local_loss +
            self.global_weight * global_loss +
            self.align_weight * align_loss +
            self.ortho_weight * ortho_loss
        )

        return total_loss, loss_dict

    def _compute_local_loss(self, user_seq_emb, item_emb, pos_pairs, neg_pairs):
        """Compute sequential BPR loss."""
        user_emb = user_seq_emb  # (B, D)

        # Get item embeddings for positive and negative items
        pos_items = pos_pairs[:, 1]  # (B,)
        neg_items = neg_pairs[:, 1]  # (B,)

        pos_item_emb = item_emb[pos_items]  # (B, D)
        neg_item_emb = item_emb[neg_items]  # (B, D)

        # BPR loss
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # (B,)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)  # (B,)

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return bpr_loss

    def _compute_global_loss(self, user_emb, item_emb, pos_pairs, neg_pairs):
        """Compute graph-based BPR loss."""
        pos_users = pos_pairs[:, 0]  # (B,)
        pos_items = pos_pairs[:, 1]  # (B,)
        neg_items = neg_pairs[:, 1]  # (B,)

        pos_user_emb = user_emb[pos_users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]

        pos_scores = (pos_user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (pos_user_emb * neg_item_emb).sum(dim=1)

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        return bpr_loss

    def _compute_alignment_loss(self, seq_emb, pos_emb, neg_emb, user_indices):
        """Compute alignment and orthogonality losses."""
        # Get graph embeddings for batch users
        batch_pos_emb = pos_emb[user_indices]  # (B, D)
        batch_neg_emb = neg_emb[user_indices]  # (B, D)

        align_loss, ortho_loss = self.alignment_module(
            seq_emb, batch_pos_emb, batch_neg_emb
        )
        return align_loss, ortho_loss

    @torch.no_grad()
    def predict(self, item_sequence, item_pos_emb, attention_mask=None):
        """
        Predict next item scores.

        Args:
            item_sequence: (batch_size, seq_len) item IDs
            item_pos_emb: (N_items, D) item interest embeddings
            attention_mask: (batch_size, seq_len) boolean mask

        Returns:
            scores: (batch_size, N_items) prediction scores
        """
        _, user_seq_emb = self.sequential_encoder(item_sequence, attention_mask)

        # Compute scores with all items
        scores = user_seq_emb @ item_pos_emb.T  # (B, N_items)

        return scores

    @torch.no_grad()
    def recommend(self, item_sequence, item_pos_emb, k=10, attention_mask=None):
        """
        Get top-k recommendations.

        Args:
            item_sequence: (batch_size, seq_len) item IDs
            item_pos_emb: (N_items, D) item interest embeddings
            k: Number of recommendations
            attention_mask: (batch_size, seq_len) boolean mask

        Returns:
            topk_items: (batch_size, k) top-k item indices
            topk_scores: (batch_size, k) top-k scores
        """
        scores = self.predict(item_sequence, item_pos_emb, attention_mask)

        # Mask already seen items
        for i, seq in enumerate(item_sequence):
            scores[i, seq] = float("-inf")

        topk_scores, topk_items = torch.topk(scores, k=k, dim=1)

        return topk_items, topk_scores
