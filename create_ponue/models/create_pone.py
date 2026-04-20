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
        # Loss weights (CREATE++ Eq. 16)
        local_weight: float = 1.0,
        global_weight: float = 0.1,
        align_weight: float = 0.1,
        ortho_weight: float = 0.1,
        contrastive_weight: float = 0.1,
        bt_lambda: float = 0.1,  # Barlow Twins off-diagonal regularization
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
        self.contrastive_weight = contrastive_weight
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

        # Alignment module (Barlow Twins + Orthogonality)
        # Implements CREATE++ Eq. 15: L_align = L_barlow_twins + mu * L_orthogonality
        self.alignment_module = AlignmentModule(
            embedding_dim=embedding_dim,
            lambda_param=bt_lambda,     # lambda for off-diagonal (redundancy reduction)
            mu_param=ortho_weight,      # mu for orthogonality to disinterest
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
        Compute multi-objective loss for CREATE-Pone.

        According to CREATE++ paper Eq. 16:
        L = L_local + w_global * L_global + w_align * L_align

        Where:
        - L_global = L_dual_feedback + L_contrastive  (Eq. 12)
        - L_dual_feedback = L_pos_bpr + L_neg_bpr  (dual-feedback BPR loss)
        - L_align = L_barlow_twins + L_orthogonality  (Eq. 15)

        This gives 5 individual losses:
        1. L_local (sequential/transformer loss)
        2. L_pos_bpr (positive BPR - part of dual-feedback)
        3. L_neg_bpr (negative BPR - part of dual-feedback)
        4. L_contrastive (contrastive loss between interest/disinterest)
        5. L_barlow_twins (alignment loss)
        6. L_orthogonality (orthogonality to disinterest)

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

        # ========== LOCAL LOSS ==========
        # 1. Local loss: Sequential next-item prediction (BPR-style)
        local_loss = self._compute_local_loss(
            user_seq_emb, item_pos_emb, pos_pairs, neg_pairs
        )
        loss_dict["local_loss"] = local_loss

        # ========== GLOBAL LOSS (Eq. 12: L_global = L_DF + L_CL) ==========
        # 2. Dual-feedback BPR loss (positive + negative BPR)
        pos_bpr_loss, neg_bpr_loss = self._compute_dual_feedback_loss(
            user_pos_emb, user_neg_emb,
            item_pos_emb, item_neg_emb,
            pos_pairs, neg_pairs
        )
        dual_feedback_loss = pos_bpr_loss + neg_bpr_loss
        loss_dict["pos_bpr_loss"] = pos_bpr_loss
        loss_dict["neg_bpr_loss"] = neg_bpr_loss
        loss_dict["dual_feedback_loss"] = dual_feedback_loss

        # 3. Contrastive loss: InfoNCE between interest and disinterest embeddings
        contrastive_loss = self._compute_contrastive_loss(
            user_pos_emb, user_neg_emb, pos_pairs[:, 0]
        )
        loss_dict["contrastive_loss"] = contrastive_loss

        # Global loss = dual-feedback + contrastive (Eq. 12)
        global_loss = dual_feedback_loss + self.contrastive_weight * contrastive_loss
        loss_dict["global_loss"] = global_loss

        # ========== ALIGNMENT LOSS (Eq. 15: L_align = L_BT + L_ortho) ==========
        # 4. Barlow Twins loss: Align seq with graph interest
        # 5. Orthogonality loss: Push seq away from disinterest
        barlow_twins_loss, orthogonality_loss = self._compute_alignment_loss(
            user_seq_emb, user_pos_emb, user_neg_emb, pos_pairs[:, 0]
        )
        loss_dict["barlow_twins_loss"] = barlow_twins_loss
        loss_dict["orthogonality_loss"] = orthogonality_loss

        # Alignment loss combines Barlow Twins + orthogonality
        align_loss = barlow_twins_loss + self.ortho_weight * orthogonality_loss
        loss_dict["align_loss"] = align_loss

        # ========== TOTAL LOSS (Eq. 16) ==========
        total_loss = (
            self.local_weight * local_loss +
            self.global_weight * global_loss +
            self.align_weight * align_loss
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

    def _compute_dual_feedback_loss(
        self,
        user_pos_emb, user_neg_emb,
        item_pos_emb, item_neg_emb,
        pos_pairs, neg_pairs
    ):
        """
        Compute dual-feedback BPR loss from Pone-GNN.

        Positive BPR: Push user-item positive closer than negative samples
        Negative BPR: Push user-item negative farther than negative samples

        Returns:
            pos_bpr_loss, neg_bpr_loss
        """
        pos_users = pos_pairs[:, 0]  # (B,)
        pos_items = pos_pairs[:, 1]  # (B,)
        neg_items = neg_pairs[:, 1]  # (B,)

        # Positive branch BPR (interest embeddings)
        pos_user_emb = user_pos_emb[pos_users]  # (B, D)
        pos_item_emb = item_pos_emb[pos_items]  # (B, D)
        neg_item_emb_pos = item_pos_emb[neg_items]  # (B, D)

        pos_scores_pos = (pos_user_emb * pos_item_emb).sum(dim=1, keepdim=True)  # (B, 1)
        neg_scores_pos = (pos_user_emb.unsqueeze(1) * neg_item_emb_pos).sum(dim=2)  # (B, 1)

        pos_bpr_loss = -F.logsigmoid(pos_scores_pos - neg_scores_pos).mean()

        # Negative branch BPR (disinterest embeddings)
        # For negative interactions, we want to push user and item apart
        neg_user_emb = user_neg_emb[pos_users]  # (B, D) - same users
        neg_item_emb = item_neg_emb[pos_items]  # (B, D)
        neg_item_emb_neg = item_neg_emb[neg_items]  # (B, D)

        # In negative branch: negative score should be higher than positive score
        pos_scores_neg = (neg_user_emb * neg_item_emb).sum(dim=1, keepdim=True)  # (B, 1)
        neg_scores_neg = (neg_user_emb.unsqueeze(1) * neg_item_emb_neg).sum(dim=2)  # (B, 1)

        neg_bpr_loss = -F.logsigmoid(neg_scores_neg - pos_scores_neg).mean()

        return pos_bpr_loss, neg_bpr_loss

    def _compute_contrastive_loss(self, user_pos_emb, user_neg_emb, user_indices):
        """
        Compute InfoNCE-style contrastive loss between interest and disinterest embeddings.

        Treats interest embedding of a user as anchor and its disinterest as negative,
        while other users' disinterest embeddings serve as additional negatives.
        """
        # Get batch embeddings
        batch_pos_emb = user_pos_emb[user_indices]  # (B, D)
        batch_neg_emb = user_neg_emb[user_indices]  # (B, D)

        # Normalize embeddings
        batch_pos_norm = F.normalize(batch_pos_emb, dim=1, p=2)
        batch_neg_norm = F.normalize(batch_neg_emb, dim=1, p=2)

        # All negative embeddings in batch (for contrastive negatives)
        all_neg_norm = F.normalize(user_neg_emb, dim=1, p=2)

        batch_size = batch_pos_emb.size(0)

        # Positive similarity: interest vs disinterest for same user
        pos_similarity = (batch_pos_norm * batch_neg_norm).sum(dim=1) / self.temperature  # (B,)

        # Negative similarity: interest vs all disinterest embeddings
        neg_similarity = batch_pos_norm @ all_neg_norm.T / self.temperature  # (B, N_users)

        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # Numerator: exp(pos_similarity)
        # Denominator: exp(pos_similarity) + sum(exp(neg_similarity))
        pos_exp = torch.exp(pos_similarity)  # (B,)
        neg_exp = torch.exp(neg_similarity).sum(dim=1)  # (B,) - sum over all negatives

        # Subtract max for numerical stability
        logits = pos_similarity - torch.logsumexp(neg_similarity, dim=1, keepdim=True).squeeze(1)
        contrastive_loss = -logits.mean()

        return contrastive_loss

    def _compute_alignment_loss(self, seq_emb, pos_emb, neg_emb, user_indices):
        """
        Compute Barlow Twins alignment loss and orthogonality loss.

        Args:
            seq_emb: (B, D) sequential user embeddings
            pos_emb: (N_users, D) graph interest embeddings
            neg_emb: (N_users, D) graph disinterest embeddings
            user_indices: (B,) indices of users in batch

        Returns:
            barlow_twins_loss: Alignment + redundancy reduction loss
            orthogonality_loss: Orthogonality to disinterest loss
        """
        # Get graph embeddings for batch users
        batch_pos_emb = pos_emb[user_indices]  # (B, D)
        batch_neg_emb = neg_emb[user_indices]  # (B, D)

        barlow_twins_loss, orthogonality_loss = self.alignment_module(
            seq_emb, batch_pos_emb, batch_neg_emb
        )
        return barlow_twins_loss, orthogonality_loss

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
