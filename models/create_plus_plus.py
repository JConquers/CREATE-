"""
CREATE++: Cross-Representation Aligned Transfer Encoders with non-linear fusion.

Combines sequential encoders (SASRec/BERT4Rec) with graph encoders (LightGCN/UltraGCN/PoneGNN)
using MLPFusion and InfoNCE alignment loss.

CREATE-Pone variant uses PoneGNN as the graph encoder, leveraging dual embeddings
for positive and negative feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import SASRec, BERT4Rec, LightGCN, UltraGCN, PoneGNN
from .fusion import MLPFusion


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss for aligning local and global representations."""

    def __init__(self, embedding_dim, lambda_offdiag=1e-3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lambda_offdiag = lambda_offdiag
        self.bn = nn.BatchNorm1d(embedding_dim, affine=False)

    def forward(self, z_local, z_global):
        """
        Args:
            z_local: (n_items, embedding_dim) from sequential encoder
            z_global: (n_items, embedding_dim) from graph encoder
        Returns:
            loss: scalar
        """
        z_local = self.bn(z_local)
        z_global = self.bn(z_global)

        c = torch.mm(z_local, z_global.t()) / z_local.size(0)

        on_diag = torch.sum((c.diag() - 1) ** 2)
        off_diag = torch.sum(c ** 2) - torch.sum(c.diag() ** 2)

        return on_diag + self.lambda_offdiag * off_diag


class InfoNCELoss(nn.Module):
    """InfoNCE loss with stricter negatives for alignment (CREATE++)."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_local, z_global, negatives=None):
        """
        Args:
            z_local: (n_items, embedding_dim) from sequential encoder
            z_global: (n_items, embedding_dim) from graph encoder
            negatives: Optional (n_items, n_negatives, embedding_dim) stricter negatives
        Returns:
            loss: scalar
        """
        n_items = z_local.size(0)
        device = z_local.device

        z_local = F.normalize(z_local, p=2, dim=1)
        z_global = F.normalize(z_global, p=2, dim=1)

        pos_sim = torch.sum(z_local * z_global, dim=-1) / self.temperature

        if negatives is not None:
            neg_sim = torch.einsum('ik,jnk->in', z_local, z_global[negatives]) / self.temperature
        else:
            neg_sim = torch.mm(z_local, z_global.t()) / self.temperature
            mask = torch.eye(n_items, device=device).bool()
            neg_sim = neg_sim[~mask].reshape(n_items, -1)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(n_items, dtype=torch.long, device=device)

        return F.cross_entropy(logits, labels)


class DualFeedbackLoss(nn.Module):
    """Dual feedback-aware loss for CREATE-Pone (Eq. 9 in CREATE++ paper).

    Uses BPR-style loss on both positive and negative interaction sets.
    """

    def __init__(self):
        super().__init__()

    def forward(self, user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg,
                positive_pairs, negative_pairs):
        """
        Args:
            user_emb_pos: (n_users, embedding_dim) interest embeddings
            item_emb_pos: (n_items, embedding_dim) interest embeddings
            user_emb_neg: (n_users, embedding_dim) disinterest embeddings
            item_emb_neg: (n_items, embedding_dim) disinterest embeddings
            positive_pairs: (n_pairs, 2) user-item positive pairs
            negative_pairs: (n_pairs, 2) user-item negative pairs
        Returns:
            loss: scalar
        """
        # Positive feedback loss
        user_idx_pos = positive_pairs[:, 0]
        item_idx_pos = positive_pairs[:, 1]
        pos_user_emb = user_emb_pos[user_idx_pos]
        pos_item_emb = item_emb_pos[item_idx_pos]
        pos_scores = torch.sum(pos_user_emb * pos_item_emb, dim=1)

        # Negative feedback loss (maximize distance from disinterest)
        user_idx_neg = negative_pairs[:, 0]
        item_idx_neg = negative_pairs[:, 1]
        neg_user_emb = user_emb_neg[user_idx_neg]
        neg_item_emb = item_emb_neg[item_idx_neg]
        neg_scores = torch.sum(neg_user_emb * neg_item_emb, dim=1)

        # BPR-style loss: maximize positive scores, minimize negative scores
        loss = -torch.mean(F.logsigmoid(pos_scores) + F.logsigmoid(-neg_scores))
        return loss


class CreatePlusPlus(nn.Module):
    """CREATE++ model combining sequential and graph encoders."""

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        max_seq_len=50,
        sequential_model='sasrec',
        graph_model='lightgcn',
        use_mlp_fusion=True,
        n_warmup_epochs=5,
        w_local=1.0,
        w_global=1.0,
        w_barlow=0.1,
        w_info=0.1,
        **kwargs
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_mlp_fusion = use_mlp_fusion
        self.n_warmup_epochs = n_warmup_epochs

        if sequential_model.lower() == 'sasrec':
            self.sequential_encoder = SASRec(n_items, embedding_dim, max_seq_len, **kwargs)
        elif sequential_model.lower() == 'bert4rec':
            self.sequential_encoder = BERT4Rec(n_items, embedding_dim, max_seq_len, **kwargs)
        else:
            raise ValueError(f"Unknown sequential model: {sequential_model}")

        if graph_model.lower() == 'lightgcn':
            self.graph_encoder = LightGCN(n_users, n_items, embedding_dim, **kwargs)
        elif graph_model.lower() == 'ultragcn':
            self.graph_encoder = UltraGCN(n_users, n_items, embedding_dim, **kwargs)
        elif graph_model.lower() == 'ponegnn':
            self.graph_encoder = PoneGNN(n_users, n_items, embedding_dim, **kwargs)
        else:
            raise ValueError(f"Unknown graph model: {graph_model}")

        if use_mlp_fusion:
            self.fusion = MLPFusion(embedding_dim)
        else:
            self.fusion = lambda loc, glob: w_local * loc + w_global * glob

        self.barlow_loss = BarlowTwinsLoss(embedding_dim)
        self.info_nce_loss = InfoNCELoss()

        self.w_local = w_local
        self.w_global = w_global
        self.w_barlow = w_barlow
        self.w_info = w_info


class CreatePone(nn.Module):
    """CREATE-Pone: CREATE variant using PoneGNN as the graph encoder.

    Combines sequential encoders (SASRec/BERT4Rec) with PoneGNN graph encoder.
    Uses only positive embeddings from PoneGNN for recommendations while leveraging
    contrastive learning between positive and negative feedback during training.

    Implements Eq. 15-16 from CREATE++ paper with extended alignment loss including
    orthogonality constraint to disinterest embeddings.
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        max_seq_len=50,
        sequential_model='sasrec',
        use_mlp_fusion=True,
        n_warmup_epochs=5,
        w_local=1.0,
        w_global=1.0,
        w_barlow=0.1,
        w_info=0.1,
        w_ortho=0.1,
        w_contrast=0.1,
        ponegnn_kwargs=None,
        **kwargs
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.use_mlp_fusion = use_mlp_fusion
        self.n_warmup_epochs = n_warmup_epochs

        if sequential_model.lower() == 'sasrec':
            self.sequential_encoder = SASRec(n_items, embedding_dim, max_seq_len, **kwargs)
        elif sequential_model.lower() == 'bert4rec':
            self.sequential_encoder = BERT4Rec(n_items, embedding_dim, max_seq_len, **kwargs)
        else:
            raise ValueError(f"Unknown sequential model: {sequential_model}")

        ponegnn_kwargs = ponegnn_kwargs or {}
        self.graph_encoder = PoneGNN(n_users, n_items, embedding_dim, **ponegnn_kwargs)

        if use_mlp_fusion:
            self.fusion = MLPFusion(embedding_dim)
        else:
            self.fusion = lambda loc, glob: w_local * loc + w_global * glob

        self.barlow_loss = BarlowTwinsLoss(embedding_dim)
        self.info_nce_loss = InfoNCELoss()
        self.dual_feedback_loss = DualFeedbackLoss()

        self.w_local = w_local
        self.w_global = w_global
        self.w_barlow = w_barlow
        self.w_info = w_info
        self.w_ortho = w_ortho  # Orthogonality loss weight (Eq. 15 in CREATE++ paper)
        self.w_contrast = w_contrast

    def get_sequential_embeddings(self, seq):
        """Get item embeddings from sequential encoder."""
        return self.sequential_encoder(seq)

    def get_graph_embeddings(self, edge_index, edge_weight=None, negative_edge_index=None):
        """Get user and item embeddings from PoneGNN encoder (positive embeddings only)."""
        user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg = self.graph_encoder(
            edge_index, edge_weight, negative_edge_index
        )
        return user_emb_pos, item_emb_pos

    def forward(self, seq, edge_index, edge_weight=None, negative_edge_index=None):
        """Full forward pass computing fused embeddings using PoneGNN positive embeddings."""
        seq_emb = self.sequential_encoder(seq)
        user_emb_pos, item_emb_pos, _, _ = self.graph_encoder(
            edge_index, edge_weight, negative_edge_index
        )

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, item_emb_pos)

        return fused_emb, item_emb_pos

    def compute_loss(
        self, seq, edge_index, edge_weight=None, negative_edge_index=None, negatives=None, epoch=0
    ):
        """Compute total loss with local, global, alignment, and contrastive terms.

        Implements Eq. 16 from CREATE++ paper:
        L = L_local + w_global * L_global + w_align * L_align + w_ortho * L_ortho
        """
        seq_emb = self.sequential_encoder(seq)
        user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg = self.graph_encoder(
            edge_index, edge_weight, negative_edge_index
        )

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, item_emb_pos)

        # Local loss: sequential encoder next-item prediction (Eq. 14)
        local_loss = F.cross_entropy(
            torch.mm(last_item_emb, item_emb_pos[: self.n_items].t()),
            seq[:, -1],
        )

        # Global loss: fused embedding prediction
        global_loss = F.cross_entropy(
            torch.mm(fused_emb, item_emb_pos[: self.n_items].t()),
            seq[:, -1],
        )

        # Extended alignment loss (Eq. 15 in CREATE++ paper)
        # Includes Barlow Twins style alignment + orthogonality to disinterest
        alignment_loss = torch.tensor(0.0, device=last_item_emb.device)
        ortho_loss = torch.tensor(0.0, device=last_item_emb.device)

        if self.w_barlow > 0:
            # Standard Barlow Twins alignment between local and global (interest)
            alignment_loss = self.barlow_loss(last_item_emb, item_emb_pos)

        if self.w_ortho > 0:
            # Orthogonality constraint: push sequential rep away from disinterest
            # This is the "push" term in Eq. 15 (third term with mu coefficient)
            h_u_norm = F.normalize(last_item_emb, p=2, dim=1)
            v_u_norm = F.normalize(user_emb_neg, p=2, dim=1)
            # Cross-correlation between local rep and disinterest embeddings
            C_hv = torch.mm(h_u_norm.t(), v_u_norm) / h_u_norm.size(0)
            ortho_loss = torch.sum(C_hv ** 2)  # Drive to zero (orthogonality)

        info_loss = (
            self.info_nce_loss(last_item_emb, item_emb_pos, negatives)
            if self.w_info > 0
            else torch.tensor(0.0)
        )

        # PoneGNN contrastive loss between positive and negative embeddings
        contrastive_loss = (
            self.graph_encoder.compute_contrastive_loss(
                user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg
            )
            if self.w_contrast > 0
            else torch.tensor(0.0)
        )

        # Total loss (Eq. 16)
        total_loss = (
            self.w_local * local_loss
            + self.w_global * global_loss
            + self.w_barlow * alignment_loss
            + self.w_info * info_loss
            + self.w_ortho * ortho_loss
            + self.w_contrast * contrastive_loss
        )

        return total_loss, {
            "local_loss": local_loss.item(),
            "global_loss": global_loss.item(),
            "alignment_loss": alignment_loss.item(),
            "ortho_loss": ortho_loss.item(),
            "info_loss": info_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def recommend(self, seq, edge_index, edge_weight=None, negative_edge_index=None, k=10):
        """Generate top-k recommendations for a batch of sequences.

        Uses only positive embeddings for scoring (standard PoneGNN approach).
        Optionally, disinterest-score filter can be applied by subtracting negative scores.
        """
        self.eval()
        with torch.no_grad():
            fused_emb, item_emb_pos = self.forward(seq, edge_index, edge_weight, negative_edge_index)
            scores = torch.mm(fused_emb, item_emb_pos.t())
            _, topk_items = torch.topk(scores, k, dim=1)
        return topk_items

    def recommend_with_disinterest_filter(
        self, seq, edge_index, edge_weight=None, negative_edge_index=None, k=10, alpha=0.5
    ):
        """Generate top-k recommendations with disinterest-score filter.

        Args:
            seq: Input sequence
            edge_index: Positive user-item interaction graph
            edge_weight: Optional edge weights for positive graph
            negative_edge_index: Negative user-item interaction graph
            k: Number of recommendations
            alpha: Weight for disinterest penalty (0 = no penalty, 1 = full penalty)

        Returns:
            topk_items: Top-k recommended items
        """
        self.eval()
        with torch.no_grad():
            seq_emb = self.sequential_encoder(seq)
            user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg = self.graph_encoder(
                edge_index, edge_weight, negative_edge_index
            )

            last_item_emb = seq_emb[:, -1, :]
            fused_emb = self.fusion(last_item_emb, item_emb_pos)

            positive_scores = torch.mm(fused_emb, item_emb_pos.t())
            negative_scores = torch.mm(fused_emb, item_emb_neg.t())

            final_scores = positive_scores - alpha * negative_scores

            _, topk_items = torch.topk(final_scores, k, dim=1)
        return topk_items
