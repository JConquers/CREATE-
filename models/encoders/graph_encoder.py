"""
Graph encoders for CREATE++: LightGCN, UltraGCN, and PoneGNN.

PoneGNN implementation follows the paper:
"Pone-GNN: Integrating Positive and Negative Feedback in Graph Neural Networks
for Recommender Systems" (https://github.com/Young0222/Pone-GNN)

Key architecture:
- LightGINConv2: message passing with (1+eps)*self-emb + neighbor aggregation
- Forward pass: pos_emb propagated on positive edges; neg_emb propagated on negative edges
- Loss: dual BPR (positive + negative) + contrastive loss (every 10 epochs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union


class GraphEncoder(nn.Module):
    """Base class for graph encoders."""

    def __init__(self, n_users, n_items, embedding_dim):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim


class LightGCN(GraphEncoder):
    """LightGCN: Simplified Graph Convolutional Network.

    Removes feature transformation and nonlinear activation from standard GCN.
    Only uses neighborhood aggregation via smoothed adjacency matrix.
    """

    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=3):
        super().__init__(n_users, n_items, embedding_dim)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.n_layers = n_layers
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, edge_index, edge_weight=None):
        """
        Args:
            edge_index: LongTensor of shape (2, n_edges) with user-item interactions
            edge_weight: Optional FloatTensor of shape (n_edges,)
        Returns:
            user_emb: FloatTensor of shape (n_users, embedding_dim)
            item_emb: FloatTensor of shape (n_items, embedding_dim)
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        all_emb = torch.cat([user_emb, item_emb], dim=0)

        for _ in range(self.n_layers):
            all_emb = self._propagate(edge_index, all_emb, edge_weight)

        return all_emb[:self.n_users], all_emb[self.n_users:]

    def _propagate(self, edge_index, embeddings, edge_weight=None):
        """Propagate embeddings through one GCN layer."""
        row, col = edge_index[0], edge_index[1]

        if edge_weight is None:
            edge_weight = torch.ones(row.shape[0], device=row.device)

        deg_inv_sqrt = torch.pow(torch.bincount(row).float().clamp(min=1), -0.5)
        deg_inv_sqrt = deg_inv_sqrt[row]

        deg_inv_sqrt_col = torch.pow(torch.bincount(col).float().clamp(min=1), -0.5)
        deg_inv_sqrt_col = deg_inv_sqrt_col[col]

        weight = deg_inv_sqrt * deg_inv_sqrt_col * edge_weight

        messages = embeddings[col] * weight.unsqueeze(1)
        new_embeddings = torch.zeros_like(embeddings)
        new_embeddings = new_embeddings.scatter_add_(0, row.unsqueeze(1).expand_as(messages), messages)

        return new_embeddings


class UltraGCN(GraphEncoder):
    """UltraGCN: Infinite-Layer Graph Neural Networks.

    Approximates infinite-layer GCN propagation with a constrained optimization objective.
    Uses item-user and user-item relationship modeling.
    """

    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=4, gamma=40, lambda_=1e-4):
        super().__init__(n_users, n_items, embedding_dim)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        self.n_layers = n_layers
        self.gamma = gamma
        self.lambda_ = lambda_
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, edge_index, edge_weight=None):
        """
        Args:
            edge_index: LongTensor of shape (2, n_edges) with user-item interactions
            edge_weight: Optional FloatTensor of shape (n_edges,)
        Returns:
            user_emb: FloatTensor of shape (n_users, embedding_dim)
            item_emb: FloatTensor of shape (n_items, embedding_dim)
        """
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        for _ in range(self.n_layers):
            user_emb = self._propagate_user(edge_index, item_emb, user_emb)
            item_emb = self._propagate_item(edge_index, user_emb, item_emb)

        return user_emb, item_emb

    def _propagate_user(self, edge_index, item_emb, user_emb):
        """Propagate item embeddings to users."""
        row, col = edge_index[0], edge_index[1]

        scores = torch.sum(user_emb[row] * item_emb[col], dim=-1)
        scores = torch.softmax(scores, dim=0)

        aggregated = torch.zeros_like(user_emb)
        aggregated = aggregated.scatter_add_(0, row.unsqueeze(1).expand_as(item_emb[col]), item_emb[col] * scores.unsqueeze(1))

        return user_emb + aggregated / (self.gamma + 1)

    def _propagate_item(self, edge_index, user_emb, item_emb):
        """Propagate user embeddings to items."""
        row, col = edge_index[0], edge_index[1]

        scores = torch.sum(item_emb[col] * user_emb[row], dim=-1)
        scores = torch.softmax(scores, dim=0)

        aggregated = torch.zeros_like(item_emb)
        aggregated = aggregated.scatter_add_(0, col.unsqueeze(1).expand_as(user_emb[row]), user_emb[row] * scores.unsqueeze(1))

        return item_emb + aggregated / (self.gamma + 1)



# ---------------------------------------------------------------------------
# LightGINConv2 — PoneGNN's convolution layer
# ---------------------------------------------------------------------------

class LightGINConv2(nn.Module):
    """Light Graph Isomorphism Network layer used by PoneGNN.

    Forward: out = (1 + eps) * self_emb + aggregate(neighbors)
    Both positive and negative embeddings are updated per layer.

    Args:
        x: Tuple (pos_emb, neg_emb) of shape (n_users + n_items, embedding_dim)
        pos_edge_index: edges for positive graph
        neg_edge_index: edges for negative graph (can be same as pos for non-first layers)
    """

    def __init__(self, embedding_dim: int, first_aggr: bool):
        super().__init__()
        self.first_aggr = first_aggr
        self.eps = nn.Parameter(torch.zeros(1))  # eps in paper, initialized to 0

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor],
                pos_edge_index: torch.Tensor,
                neg_edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: tuple of (pos_emb, neg_emb), each (n_users+n_items, dim)
            pos_edge_index: (2, n_pos_edges)
            neg_edge_index: (2, n_neg_edges)
        Returns:
            (out_pos, out_neg) each (n_users+n_items, dim)
        """
        pos_emb, neg_emb = x

        def get_norm(node_size: int, edge_index: torch.Tensor):
            row, col = edge_index
            deg = torch.bincount(col, minlength=node_size).float().clamp(min=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm, deg_inv_sqrt

        def gin_norm(out: torch.Tensor, input_x: torch.Tensor,
                     deg_inv_sqrt: torch.Tensor) -> torch.Tensor:
            # (1 + eps) * self_emb + neighbor_agg
            norm_self = deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(1) * input_x
            out = out + (1 + self.eps) * norm_self
            return out

        norm_pos, deg_inv_sqrt_pos = get_norm(pos_emb.size(0), pos_edge_index)
        norm_neg, deg_inv_sqrt_neg = get_norm(neg_emb.size(0), neg_edge_index)

        # Aggregate from positive edges
        row_pos, col_pos = pos_edge_index[0], pos_edge_index[1]
        msg_pos = norm_pos.unsqueeze(1) * pos_emb[col_pos]
        out_pos = torch.zeros_like(pos_emb)
        out_pos = out_pos.scatter_add(0, row_pos.unsqueeze(1).expand_as(msg_pos), msg_pos)

        # Aggregate from negative edges
        row_neg, col_neg = neg_edge_index[0], neg_edge_index[1]
        msg_neg = norm_neg.unsqueeze(1) * neg_emb[col_neg]
        out_neg = torch.zeros_like(neg_emb)
        out_neg = out_neg.scatter_add(0, row_neg.unsqueeze(1).expand_as(msg_neg), msg_neg)

        if self.first_aggr:
            # First layer: positive neighbors aggregate from pos_emb, negative from pos_emb
            out_pos = gin_norm(out_pos, pos_emb, deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, neg_emb, deg_inv_sqrt_neg)
        else:
            # Subsequent layers: positive neighbors aggregate from pos_emb on pos edges
            # negative neighbors aggregate from neg_emb on pos edges (using pos norms)
            out_pos = gin_norm(out_pos, pos_emb, deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, neg_emb, deg_inv_sqrt_pos)

        return out_pos, out_neg


# ---------------------------------------------------------------------------
# PoneGNN — positive/negative dual-embedding GNN
# ---------------------------------------------------------------------------

class PoneGNN(GraphEncoder):
    """PoneGNN: Positive/Negative Feedback GNN.

    Follows the official implementation from https://github.com/Young0222/Pone-GNN

    Key features:
    - Dual embeddings: Z (interest) for likes, V (disinterest) for dislikes
    - LightGINConv2 message passing on positive and negative graphs
    - Skip connections with alpha = 1 / (n_layers + 1)
    - BPR loss on both positive and negative branches
    - Contrastive loss between positive and negative user embeddings (every 10 epochs)
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 4,
        epsilon_p: float = 0.0,   # not used directly — eps is learned in LightGINConv2
        epsilon_n: float = 0.0,
        lambda_cl: float = 1.0,   # contrastive loss weight
        lambda_reg: float = 5e-5,
        temperature: float = 1.0,
    ):
        super().__init__(n_users, n_items, embedding_dim)
        self.n_layers = n_layers
        self.lambda_cl = lambda_cl
        self.lambda_reg = lambda_reg
        self.temperature = temperature

        n_total = n_users + n_items

        # Positive embeddings (interest)
        self.user_embedding_pos = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_pos = nn.Embedding(n_items, embedding_dim)

        # Negative embeddings (disinterest)
        self.user_embedding_neg = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_neg = nn.Embedding(n_items, embedding_dim)

        nn.init.xavier_normal_(self.user_embedding_pos.weight)
        nn.init.xavier_normal_(self.item_embedding_pos.weight)
        nn.init.xavier_normal_(self.user_embedding_neg.weight)
        nn.init.xavier_normal_(self.item_embedding_neg.weight)

        # LightGINConv2 layers: first layer uses pos_emb→neg on neg edges,
        # subsequent layers use pos_emb→neg on pos edges
        self.conv = nn.ModuleList()
        for i in range(n_layers):
            self.conv.append(LightGINConv2(embedding_dim, first_aggr=(i == 0)))

    def forward(self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        """Full forward pass computing positive and negative embeddings.

        Args:
            pos_edge_index: (2, n_pos_edges) user-item positive interaction edges
            neg_edge_index: (2, n_neg_edges) user-item negative interaction edges
        Returns:
            pos_emb: (n_users + n_items, embedding_dim) positive/interest embeddings
            neg_emb: (n_users + n_items, embedding_dim) negative/disinterest embeddings
        """
        # Combine user + item embeddings into unified node table
        pos_emb = torch.cat([self.user_embedding_pos.weight, self.item_embedding_pos.weight], dim=0)
        neg_emb = torch.cat([self.user_embedding_neg.weight, self.item_embedding_neg.weight], dim=0)

        alpha = 1.0 / (self.n_layers + 1)

        for i in range(self.n_layers):
            pos_emb, neg_emb = self.conv[i]((pos_emb, neg_emb), pos_edge_index, neg_edge_index)
            # Skip connection: accum = prev + alpha * new
            pos_emb = pos_emb + alpha * torch.cat(
                [self.user_embedding_pos.weight, self.item_embedding_pos.weight], dim=0)
            neg_emb = neg_emb + alpha * torch.cat(
                [self.user_embedding_neg.weight, self.item_embedding_neg.weight], dim=0)

        return pos_emb, neg_emb

    def split_embeddings(self, pos_emb: torch.Tensor, neg_emb: torch.Tensor):
        """Split unified embeddings into (user_pos, item_pos, user_neg, item_neg)."""
        u_pos, i_pos = pos_emb[:self.n_users], pos_emb[self.n_users:]
        u_neg, i_neg = neg_emb[:self.n_users], neg_emb[self.n_users:]
        return u_pos, i_pos, u_neg, i_neg

    def compute_loss(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        weights: torch.Tensor,         # rating - offset (for weighted BPR)
        negative_samples: torch.Tensor, # (batch, n_neg) item indices
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        epoch: int = 0,
    ):
        """Compute full PoneGNN loss (dual BPR + contrastive).

        Args:
            users: (batch,) user indices
            items: (batch,) positive item indices
            weights: (batch,) rating - offset, used for weighted BPR
            negative_samples: (batch, n_neg) negative item indices
            pos_edge_index: (2, n_pos_edges)
            neg_edge_index: (2, n_neg_edges)
            epoch: current epoch — contrastive loss triggered every 10 epochs
        """
        pos_emb, neg_emb = self.forward(pos_edge_index, neg_edge_index)
        u_pos, i_pos, u_neg, i_neg = self.split_embeddings(pos_emb, neg_emb)

        # Shift indices: items are offset by n_users in the unified embedding table
        item_offset = self.n_users

        u_p = u_pos[users]                                    # (batch, dim)
        i_p = i_pos[items]                                    # (batch, dim)
        i_n = i_pos[negative_samples]                         # (batch, n_neg, dim)
        u_n = u_neg[users]
        i_n_emb = i_neg[items]                                # (batch, dim)
        i_n_neg = i_neg[negative_samples]                      # (batch, n_neg, dim)

        # Positive BPR loss on interest embeddings
        pos_score = torch.sum(u_p * i_p, dim=1)                # (batch,)
        neg_score = torch.bmm(i_n, u_p.unsqueeze(2)).squeeze(2)  # (batch, n_neg)
        weight_factor = (-0.5 * torch.sign(weights) + 1.5).unsqueeze(1)  # like→1.0, dislike→2.0
        pos_bpr = F.logsigmoid(weight_factor * pos_score.unsqueeze(1) - neg_score).sum(dim=1)
        pos_bpr_loss = -torch.mean(pos_bpr)

        # Regularization on positive embeddings
        reg_loss_pos = (u_p ** 2).sum() + (i_p ** 2).sum() + (i_n ** 2).sum()

        total_loss = pos_bpr_loss + self.lambda_reg * reg_loss_pos

        # Negative BPR + Contrastive loss: triggered every 10 epochs
        if epoch % 10 == 1:
            u_n_full = u_neg[users]
            i_n_full = i_neg[items]
            i_neg_neg = i_neg[negative_samples]

            neg_score_bpr = torch.bmm(i_neg_neg, u_n_full.unsqueeze(2)).squeeze(2)
            pos_score_neg = torch.sum(u_n_full * i_n_full, dim=1)
            weight_factor_neg = (0.5 * torch.sign(weights) + 1.5).unsqueeze(1)
            neg_bpr = F.logsigmoid(neg_score_bpr - weight_factor_neg * pos_score_neg.unsqueeze(1)).sum(dim=1)
            neg_bpr_loss = -torch.mean(neg_bpr)

            reg_loss_neg = (u_n_full ** 2).sum() + (i_n_full ** 2).sum() + (i_neg_neg ** 2).sum()
            total_loss = total_loss + neg_bpr_loss + self.lambda_reg * reg_loss_neg

            # Contrastive loss: push u_pos away from u_neg
            u_p_norm = F.normalize(u_p, dim=1)
            u_n_norm = F.normalize(u_n, dim=1)
            pos_sim = torch.sum(u_p_norm * F.normalize(i_pos[items], dim=1), dim=1)
            neg_sim = torch.sum(u_n_norm * F.normalize(i_neg[items], dim=1), dim=1)
            pos_pair = torch.exp(pos_sim / self.temperature)
            neg_pair = torch.exp(neg_sim / self.temperature)
            contrastive_loss = -torch.log(pos_pair / (pos_pair + neg_pair)).mean()
            total_loss = total_loss + self.lambda_cl * contrastive_loss

        return total_loss

    def compute_reg_loss(self):
        """L2 regularization on all embedding tables."""
        return self.lambda_reg * (
            torch.norm(self.user_embedding_pos.weight) ** 2 +
            torch.norm(self.item_embedding_pos.weight) ** 2 +
            torch.norm(self.user_embedding_neg.weight) ** 2 +
            torch.norm(self.item_embedding_neg.weight) ** 2
        )

    def get_embeddings(self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        """Return all four embedding matrices (for inference / evaluation)."""
        pos_emb, neg_emb = self.forward(pos_edge_index, neg_edge_index)
        return self.split_embeddings(pos_emb, neg_emb)
