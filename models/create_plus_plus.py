"""
CREATE++: Cross-Representation Aligned Transfer Encoders with non-linear fusion.

Combines sequential encoders (SASRec/BERT4Rec) with graph encoders (LightGCN/UltraGCN)
using MLPFusion and InfoNCE alignment loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import SASRec, BERT4Rec, LightGCN, UltraGCN
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

    def get_sequential_embeddings(self, seq):
        """Get item embeddings from sequential encoder."""
        return self.sequential_encoder(seq)

    def get_graph_embeddings(self, edge_index, edge_weight=None):
        """Get user and item embeddings from graph encoder."""
        return self.graph_encoder(edge_index, edge_weight)

    def forward(self, seq, edge_index, edge_weight=None):
        """Full forward pass computing fused embeddings."""
        seq_emb = self.sequential_encoder(seq)
        user_emb_g, item_emb_g = self.graph_encoder(edge_index, edge_weight)

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, item_emb_g)

        return fused_emb, item_emb_g

    def compute_loss(self, seq, edge_index, edge_weight=None, negatives=None, epoch=0):
        """Compute total loss with local, global, and alignment terms."""
        seq_emb = self.sequential_encoder(seq)
        user_emb_g, item_emb_g = self.graph_encoder(edge_index, edge_weight)

        last_item_emb = seq_emb[:, -1, :]
        fused_emb = self.fusion(last_item_emb, item_emb_g)

        local_loss = F.cross_entropy(
            torch.mm(last_item_emb, item_emb_g[:self.n_items].t()),
            seq[:, -1]
        )

        global_loss = F.cross_entropy(
            torch.mm(fused_emb, item_emb_g[:self.n_items].t()),
            seq[:, -1]
        )

        alignment_loss = self.barlow_loss(last_item_emb, item_emb_g) if self.w_barlow > 0 else torch.tensor(0.0)
        info_loss = self.info_nce_loss(last_item_emb, item_emb_g, negatives) if self.w_info > 0 else torch.tensor(0.0)

        total_loss = (
            self.w_local * local_loss +
            self.w_global * global_loss +
            self.w_barlow * alignment_loss +
            self.w_info * info_loss
        )

        return total_loss, {
            'local_loss': local_loss.item(),
            'global_loss': global_loss.item(),
            'alignment_loss': alignment_loss.item(),
            'info_loss': info_loss.item()
        }

    def recommend(self, seq, edge_index, edge_weight=None, k=10):
        """Generate top-k recommendations for a batch of sequences."""
        self.eval()
        with torch.no_grad():
            fused_emb, item_emb_g = self.forward(seq, edge_index, edge_weight)
            scores = torch.mm(fused_emb, item_emb_g.t())
            _, topk_items = torch.topk(scores, k, dim=1)
        return topk_items
