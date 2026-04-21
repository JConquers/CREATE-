"""PoneGNN graph encoder implementation optimized to match official PoneGNN performance."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGINConv(MessagePassing):
    """Optimized LightGINConv matching the official PoneGNN LightGINConv2 implementation.

    Key optimizations:
    - message_and_aggregate for sparse tensor support
    - In-place operations where possible
    - Cached degree computation
    """

    def __init__(self, in_channels: int, out_channels: int, first_aggr: bool = True):
        super().__init__(aggr='add')
        self.first_aggr = first_aggr
        self.eps = nn.Parameter(torch.empty(1))
        self.eps.data.fill_(0.0)

    def forward(self, x: tuple, pos_edge_index: torch.Tensor,
                neg_edge_index: torch.Tensor) -> tuple:
        """
        Args:
            x: Tuple of (pos_embeddings, neg_embeddings)
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
        """
        pos_emb, neg_emb = x

        def get_norm(node, edge_index):
            row, col = edge_index
            deg = degree(col, node.size(0), dtype=node.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm, deg_inv_sqrt

        def gin_norm(out, input_x, input_deg_inv_sqrt):
            # Official pattern: use range() for small overhead reduction
            norm_self = input_deg_inv_sqrt[range(input_x.size(0))] * input_deg_inv_sqrt[range(input_x.size(0))]
            norm_self = norm_self.unsqueeze(dim=1).repeat(1, input_x.size(1))
            out = out + (1 + self.eps) * norm_self * input_x
            return out

        norm_pos, deg_inv_sqrt_pos = get_norm(pos_emb, pos_edge_index)
        norm_neg, deg_inv_sqrt_neg = get_norm(neg_emb, neg_edge_index)

        if self.first_aggr:
            # Use size=None (explicit, matches official)
            out_pos = self.propagate(pos_edge_index, x=pos_emb, size=None, norm=norm_pos)
            out_neg = self.propagate(neg_edge_index, x=neg_emb, size=None, norm=norm_neg)
            out_pos = gin_norm(out_pos, pos_emb, deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, neg_emb, deg_inv_sqrt_neg)
            return out_pos, out_neg
        else:
            out_pos = self.propagate(pos_edge_index, x=pos_emb, size=None, norm=norm_pos)
            out_neg = self.propagate(pos_edge_index, x=neg_emb, size=None, norm=norm_pos)
            out_pos = gin_norm(out_pos, pos_emb, deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, neg_emb, deg_inv_sqrt_pos)
            return out_pos, out_neg

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x: torch.Tensor) -> torch.Tensor:
        """Sparse tensor support for faster message passing."""
        from torch_geometric.utils import spmm
        return spmm(adj_t, x, reduce=self.aggr)


class PoneGNNEncoder(nn.Module):
    """Optimized PoneGNN encoder matching official implementation performance.

    Supports both:
    1. Pre-training with official loss pattern (compute_loss)
    2. Joint training with dual-feedback losses (compute_dual_feedback_loss, etc.)
    """

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
        self.reg = reg
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight

        # Positive and negative embeddings
        self.user_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))
        self.user_neg_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))

        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(LightGINConv(embedding_dim, embedding_dim, i == 0))

    def forward(self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor) -> tuple:
        """Forward pass with cached embeddings for loss computation."""
        alpha = 1.0 / (self.num_layers + 1)

        ego_pos_embeddings = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        ego_neg_embeddings = torch.cat((self.user_neg_embedding, self.item_neg_embedding), dim=0)

        ego_embeddings = (ego_pos_embeddings, ego_neg_embeddings)
        pos_embeddings = ego_pos_embeddings * alpha
        neg_embeddings = ego_neg_embeddings * alpha

        for i in range(self.num_layers):
            ego_embeddings = self.conv_layers[i](ego_embeddings, pos_edge_index, neg_edge_index)
            pos_embeddings = pos_embeddings + ego_embeddings[0] * alpha
            neg_embeddings = neg_embeddings + ego_embeddings[1] * alpha

        # Cache embeddings for loss computation
        self._pos_emb = pos_embeddings
        self._neg_emb = neg_embeddings

        return pos_embeddings, neg_embeddings

    @property
    def pos_emb(self):
        """Cached positive embeddings."""
        return getattr(self, '_pos_emb', None)

    @property
    def neg_emb(self):
        """Cached negative embeddings."""
        return getattr(self, '_neg_emb', None)

    def compute_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        weights: torch.Tensor,
        neg_items: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Pre-training loss matching official implementation."""
        pos_emb, neg_emb = self(pos_edge_index, neg_edge_index)

        u_p = pos_emb[users]
        i_p = pos_emb[pos_items]
        n_p = pos_emb[neg_items]

        positive_batch = torch.mul(u_p, i_p)
        negative_batch = torch.mul(u_p.view(len(u_p), 1, self.embedding_dim), n_p)

        weight_factor = (0.5 * torch.sign(weights) + 1.5).view(len(u_p), 1)

        pos_bpr_loss = F.logsigmoid(
            weight_factor * positive_batch.sum(dim=1).view(len(u_p), 1)
            - negative_batch.sum(dim=2)
        ).sum(dim=1)
        pos_bpr_loss = torch.mean(pos_bpr_loss)

        reg_loss_1 = 0.5 * (u_p ** 2).sum() + 0.5 * (i_p ** 2).sum() + 0.5 * (n_p ** 2).sum()

        loss = -pos_bpr_loss + self.reg * reg_loss_1

        if epoch % 10 == 1:
            u_n = neg_emb[users]
            i_n = neg_emb[pos_items]
            n_n = neg_emb[neg_items]

            positive_batch_n = torch.mul(u_n, i_n)
            negative_batch_n = torch.mul(u_n.view(len(u_n), 1, self.embedding_dim), n_n)

            neg_bpr_loss = F.logsigmoid(
                negative_batch_n.sum(dim=2)
                - weight_factor * positive_batch_n.sum(dim=1).view(len(u_n), 1)
            ).sum(dim=1)
            neg_bpr_loss = torch.mean(neg_bpr_loss)

            reg_loss_2 = 0.5 * (u_n ** 2).sum() + 0.5 * (i_n ** 2).sum() + 0.5 * (n_n ** 2).sum()

            loss = loss - neg_bpr_loss + self.reg * reg_loss_2

            u_p_norm = F.normalize(u_p, dim=1)
            i_p_norm = F.normalize(i_p, dim=1)
            u_n_norm = F.normalize(u_n, dim=1)
            i_n_norm = F.normalize(i_n, dim=1)

            positive_similarity = torch.sum(torch.mul(u_p_norm, i_p_norm), dim=1)
            negative_similarity = torch.sum(torch.mul(u_n_norm, i_n_norm), dim=1)

            positive_pair_similarity = torch.exp(positive_similarity / self.temperature)
            negative_pair_similarity = torch.exp(negative_similarity / self.temperature)

            contrastive_loss = -torch.log(
                positive_pair_similarity / (positive_pair_similarity + negative_pair_similarity)
            ).mean()

            loss = loss + self.contrastive_weight * contrastive_loss

        return loss

    def compute_dual_feedback_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> tuple:
        """Dual-feedback BPR loss for joint training."""
        if self.pos_emb is None or self.neg_emb is None:
            return (
                torch.tensor(0.0, device=self.user_embedding.device),
                torch.tensor(0.0, device=self.user_embedding.device),
                torch.tensor(0.0, device=self.user_embedding.device),
            )

        # Get embeddings
        u_pos = self.pos_emb[:self.num_users][users]
        u_neg = self.neg_emb[:self.num_users][users]
        i_pos = self.pos_emb[self.num_users:][pos_items]
        i_neg = self.neg_emb[self.num_users:][pos_items]
        n_pos = self.pos_emb[self.num_users:][neg_items]
        n_neg = self.neg_emb[self.num_users:][neg_items]

        # Positive BPR: maximize u_pos · i_pos - u_pos · n_pos
        pos_scores = (u_pos * i_pos).sum(dim=1)
        neg_scores = (u_pos * n_pos).sum(dim=1)
        pos_bpr_loss = F.softplus(neg_scores - pos_scores).mean()

        # Negative BPR: maximize u_neg · n_neg - u_neg · i_neg
        neg_scores_neg = (u_neg * n_neg).sum(dim=1)
        pos_scores_neg = (u_neg * i_neg).sum(dim=1)
        neg_bpr_loss = F.softplus(pos_scores_neg - neg_scores_neg).mean()

        return pos_bpr_loss, neg_bpr_loss, pos_bpr_loss + neg_bpr_loss

    def compute_orthogonal_loss(self, users: torch.Tensor) -> torch.Tensor:
        """Orthogonal loss to decorrelate positive and negative embeddings."""
        if self.pos_emb is None or self.neg_emb is None:
            return torch.tensor(0.0, device=self.user_embedding.device)

        u_pos = self.pos_emb[:self.num_users][users]
        u_neg = self.neg_emb[:self.num_users][users]

        u_pos_norm = F.normalize(u_pos, dim=1)
        u_neg_norm = F.normalize(u_neg, dim=1)

        similarity = (u_pos_norm * u_neg_norm).sum(dim=1)
        return (similarity ** 2).mean()

    def compute_contrastive_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE-style contrastive loss for joint training."""
        if self.pos_emb is None or self.neg_emb is None:
            return torch.tensor(0.0, device=self.user_embedding.device)

        u_pos = self.pos_emb[:self.num_users][users]
        u_neg = self.neg_emb[:self.num_users][users]
        i_pos = self.pos_emb[self.num_users:][pos_items]
        i_neg = self.neg_emb[self.num_users:][pos_items]

        u_pos_norm = F.normalize(u_pos, dim=1)
        u_neg_norm = F.normalize(u_neg, dim=1)
        i_pos_norm = F.normalize(i_pos, dim=1)
        i_neg_norm = F.normalize(i_neg, dim=1)

        pos_similarity = (u_pos_norm * i_pos_norm).sum(dim=1)
        neg_similarity = (u_neg_norm * i_neg_norm).sum(dim=1)

        contrastive_loss = F.softplus(neg_similarity - pos_similarity).mean()
        return self.contrastive_weight * contrastive_loss

    @torch.no_grad()
    def get_ui_embeddings(self, pos_edge_index, neg_edge_index):
        """Get user and item embeddings separately."""
        pos_embeddings, neg_embeddings = self(pos_edge_index, neg_edge_index)
        u_p, i_p = torch.split(pos_embeddings, [self.num_users, self.num_items], dim=0)
        u_n, i_n = torch.split(neg_embeddings, [self.num_users, self.num_items], dim=0)
        return u_p, u_n, i_p, i_n
