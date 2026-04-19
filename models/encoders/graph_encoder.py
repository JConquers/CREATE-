"""
Graph encoders for CREATE++: LightGCN and UltraGCN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
