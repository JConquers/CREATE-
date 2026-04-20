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

    def forward(self, z_local, z_global, negatives=None, n_negatives=1000):
        """
        Args:
            z_local: (batch_size, embedding_dim) from sequential encoder
            z_global: (n_items, embedding_dim) from graph encoder
            negatives: Optional (batch_size, n_negatives) of negative item indices
            n_negatives: Number of negatives to sample if negatives not provided and matrix too large
        Returns:
            loss: scalar
        """
        device = z_local.device
        batch_size = z_local.size(0)

        z_local = F.normalize(z_local, p=2, dim=1)
        z_global = F.normalize(z_global, p=2, dim=1)

        # Positives: last-item embedding vs its graph embedding (same item index)
        pos_sim = torch.sum(z_local * z_global, dim=-1) / self.temperature

        if negatives is not None:
            # Use provided negative indices
            neg_emb = z_global[negatives]                        # (batch, n_neg, dim)
            neg_sim = torch.einsum('bd,bnd->bn', z_local, neg_emb) / self.temperature
        else:
            n_items = z_global.size(0)
            if n_items * batch_size > 50_000_000:
                # Matrix too large: sample random negatives instead of full softmax
                neg_indices = torch.randint(0, n_items, (batch_size, n_negatives), device=device)
                neg_emb = z_global[neg_indices]                  # (batch, n_neg, dim)
                neg_sim = torch.einsum('bd,bnd->bn', z_local, neg_emb) / self.temperature
            else:
                # Full matrix: mask out self (diagonal)
                neg_sim = torch.mm(z_local, z_global.t()) / self.temperature
                mask = torch.eye(batch_size, n_items, device=device).bool()
                neg_sim = neg_sim[~mask].reshape(batch_size, -1)

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

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
            self.graph_encoder = LightGCN(n_users, n_items, embedding_dim, n_layers=kwargs.pop('n_layers_gnn', 4))
        elif graph_model.lower() == 'ultragcn':
            self.graph_encoder = UltraGCN(n_users, n_items, embedding_dim, n_layers=kwargs.pop('n_layers_gnn', 4))
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

    def forward(self, seq, edge_index, edge_weight=None):
        """Full forward pass computing fused embeddings."""
        seq_emb = self.sequential_encoder(seq)
        user_emb, item_emb = self.graph_encoder(edge_index, edge_weight)
        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, item_emb)
        return fused_emb, item_emb

    def compute_loss(self, seq, edge_index, edge_weight=None, negatives=None):
        """Compute total loss for CREATE++ model.

        L = w_local * L_local + w_global * L_global + w_barlow * L_barlow + w_info * L_info
        """
        seq_emb = self.sequential_encoder(seq)
        user_emb, item_emb = self.graph_encoder(edge_index, edge_weight)

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, item_emb)

        # Local loss: next-item prediction from sequential encoder
        local_loss = F.cross_entropy(
            torch.mm(last_item_emb, item_emb[: self.n_items].t()),
            seq[:, -1],
        )

        # Global loss: next-item prediction from fused embedding
        global_loss = F.cross_entropy(
            torch.mm(fused_emb, item_emb[: self.n_items].t()),
            seq[:, -1],
        )

        # Alignment loss: Barlow Twins between local and global item embeddings
        alignment_loss = self.barlow_loss(last_item_emb, item_emb)

        info_loss = (
            self.info_nce_loss(last_item_emb, item_emb, negatives)
            if self.w_info > 0
            else torch.tensor(0.0, device=last_item_emb.device)
        )

        total_loss = (
            self.w_local * local_loss
            + self.w_global * global_loss
            + self.w_barlow * alignment_loss
            + self.w_info * info_loss
        )

        return total_loss, {
            "local_loss": local_loss.item(),
            "global_loss": global_loss.item(),
            "alignment_loss": alignment_loss.item(),
            "info_loss": info_loss.item(),
        }


class CreatePone(nn.Module):
    """CREATE-Pone: CREATE variant using PoneGNN as the graph encoder.

    Combines sequential encoders (SASRec/BERT4Rec) with PoneGNN graph encoder.
    Uses dual BPR loss + contrastive loss on positive/negative graphs from PoneGNN,
    plus alignment losses (Barlow Twins) between sequential and graph representations.

    Implements Eq. 14-16 from CREATE++ paper.
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
        w_contrast=1.0,
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

        self.w_local = w_local
        self.w_global = w_global
        self.w_barlow = w_barlow
        self.w_info = w_info
        self.w_ortho = w_ortho
        self.w_contrast = w_contrast

    def forward(self, seq, pos_edge_index, neg_edge_index=None):
        """Full forward pass computing fused embeddings.

        Args:
            seq: (batch, seq_len) item sequence
            pos_edge_index: (2, n_pos_edges) positive interaction graph
            neg_edge_index: (2, n_neg_edges) negative interaction graph
        Returns:
            fused_emb: (batch, embedding_dim)
            item_emb_pos: (n_items, embedding_dim) positive item embeddings
        """
        seq_emb = self.sequential_encoder(seq)
        pos_emb, neg_emb = self.graph_encoder(pos_edge_index, neg_edge_index)
        u_pos, i_pos, u_neg, i_neg = self.graph_encoder.split_embeddings(pos_emb, neg_emb)
        del u_pos, u_neg, i_neg  # not needed for forward

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, i_pos)

        return fused_emb, i_pos

    def compute_loss(
        self,
        seq,
        pos_edge_index,
        neg_edge_index,
        train_user,
        train_item,
        train_weights,
        negative_samples,
        epoch=0,
    ):
        """Compute total CREATE-Pone loss.

        Combines:
        - PoneGNN dual BPR loss + contrastive (graph component)
        - Local loss: next-item prediction from sequential encoder
        - Global loss: next-item prediction from fused embedding
        - Barlow Twins alignment between sequential last-item and graph item embeddings
        - Orthogonality: push sequential rep away from disinterest embeddings

        Args:
            seq: (batch, seq_len) padded sequences
            pos_edge_index: (2, n_pos_edges)
            neg_edge_index: (2, n_neg_edges)
            train_user: (n_train,) user indices for BPR pairs
            train_item: (n_train,) positive item indices for BPR pairs
            train_weights: (n_train,) rating - offset, for weighted BPR
            negative_samples: (n_train, n_neg) negative item indices for BPR
            epoch: current epoch for PoneGNN periodic contrastive loss (every 10 epochs)
        """
        seq_emb = self.sequential_encoder(seq)
        pos_emb, neg_emb = self.graph_encoder(pos_edge_index, neg_edge_index)
        u_pos, i_pos, u_neg, i_neg = self.graph_encoder.split_embeddings(pos_emb, neg_emb)

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, i_pos)

        # ---- PoneGNN loss (dual BPR + contrastive) ----
        ponegnn_loss = self.graph_encoder.compute_loss(
            train_user,
            train_item,
            train_weights,
            negative_samples,
            pos_edge_index,
            neg_edge_index,
            epoch=epoch,
        )

        # ---- Local loss: sequential encoder next-item prediction ----
        local_loss = F.cross_entropy(
            torch.mm(last_item_emb, i_pos[:self.n_items].t()),
            seq[:, -1],
        )

        # ---- Global loss: fused embedding prediction ----
        global_loss = F.cross_entropy(
            torch.mm(fused_emb, i_pos[:self.n_items].t()),
            seq[:, -1],
        )

        # ---- Alignment: Barlow Twins between last-item seq rep and graph item rep ----
        alignment_loss = self.barlow_loss(last_item_emb, i_pos)

        # ---- Orthogonality: push seq rep away from disinterest ----
        ortho_loss = torch.tensor(0.0, device=last_item_emb.device)
        if self.w_ortho > 0:
            h_u_norm = F.normalize(last_item_emb, p=2, dim=1)
            v_u_norm = F.normalize(u_neg, p=2, dim=1)
            C_hv = torch.mm(h_u_norm.t(), v_u_norm) / h_u_norm.size(0)
            ortho_loss = torch.sum(C_hv ** 2)

        total_loss = (
            ponegnn_loss
            + self.w_local * local_loss
            + self.w_global * global_loss
            + self.w_barlow * alignment_loss
            + self.w_ortho * ortho_loss
        )

        return total_loss, {
            "ponegnn_loss": ponegnn_loss.item(),
            "local_loss": local_loss.item(),
            "global_loss": global_loss.item(),
            "alignment_loss": alignment_loss.item(),
            "ortho_loss": ortho_loss.item(),
        }

    def recommend(self, seq, pos_edge_index, neg_edge_index, k=10):
        """Top-k recommendations using disinterest filter (Eq. in paper)."""
        self.eval()
        with torch.no_grad():
            seq_emb = self.sequential_encoder(seq)
            pos_emb, neg_emb = self.graph_encoder(pos_edge_index, neg_edge_index)
            u_pos, i_pos, u_neg, i_neg = self.graph_encoder.split_embeddings(pos_emb, neg_emb)

            last_item_emb = seq_emb[:, -1, :]
            fused_emb = self.fusion(last_item_emb, i_pos)

            # Positive scores minus weighted negative scores (disinterest filter)
            pos_scores = torch.mm(fused_emb, i_pos.t())
            neg_scores = torch.mm(fused_emb, i_neg.t())
            final_scores = pos_scores - 0.5 * neg_scores

            _, topk_items = torch.topk(final_scores, k, dim=1)
        return topk_items
