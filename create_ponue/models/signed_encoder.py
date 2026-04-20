"""
Signed Graph Encoder for CREATE-Pone.
Implements dual-branch message passing inspired by Pone-GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightSignedConv(MessagePassing):
    """Lightweight signed graph convolution without feature transformation."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, pos_edge_index, neg_edge_index):
        """
        Args:
            x: Tuple of (pos_embeddings, neg_embeddings)
            pos_edge_index: Edge index for positive graph
            neg_edge_index: Edge index for negative graph
        """
        def get_norm(node_features, edge_index):
            row, col = edge_index
            deg = degree(col, node_features.size(0), dtype=node_features.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm

        pos_x, neg_x = x
        norm_pos = get_norm(pos_x, pos_edge_index)
        norm_neg = get_norm(neg_x, neg_edge_index)

        # Positive branch: aggregate from positive graph
        out_pos = self.propagate(pos_edge_index, x=pos_x, size=None, norm=norm_pos)

        # Negative branch: aggregate from negative graph
        out_neg = self.propagate(neg_edge_index, x=neg_x, size=None, norm=norm_neg)

        return out_pos, out_neg

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class SignedGraphEncoder(nn.Module):
    """
    Signed Graph Encoder with dual-branch message passing.

    Learns separate interest embeddings (from positive interactions)
    and disinterest embeddings (from negative interactions).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        reg_weight: float = 1e-4,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg_weight = reg_weight

        # Interest embeddings (positive graph)
        self.user_pos_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_pos_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))

        # Disinterest embeddings (negative graph)
        self.user_neg_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))

        # Initialize embeddings
        nn.init.xavier_normal_(self.user_pos_embedding)
        nn.init.xavier_normal_(self.item_pos_embedding)
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(LightSignedConv())

    def forward(self, pos_edge_index, neg_edge_index):
        """
        Perform dual-branch message passing.

        Args:
            pos_edge_index: Edge index for positive graph (2, E_pos)
            neg_edge_index: Edge index for negative graph (2, E_neg)

        Returns:
            user_pos_emb, item_pos_emb: Interest embeddings
            user_neg_emb, item_neg_emb: Disinterest embeddings
        """
        # Combine user and item embeddings
        pos_embeddings = torch.cat([self.user_pos_embedding, self.item_pos_embedding], dim=0)
        neg_embeddings = torch.cat([self.user_neg_embedding, self.item_neg_embedding], dim=0)

        alpha = 1.0 / (self.num_layers + 1)
        final_pos_emb = pos_embeddings * alpha
        final_neg_emb = neg_embeddings * alpha

        current_pos = pos_embeddings
        current_neg = neg_embeddings

        for layer in self.conv_layers:
            out_pos, out_neg = layer((current_pos, current_neg), pos_edge_index, neg_edge_index)
            # Residual connection with layer averaging
            final_pos_emb = final_pos_emb + out_pos * alpha
            final_neg_emb = final_neg_emb + out_neg * alpha
            current_pos, current_neg = out_pos, out_neg

        # Split user and item embeddings
        user_pos_emb = final_pos_emb[:self.num_users]
        item_pos_emb = final_pos_emb[self.num_users:]
        user_neg_emb = final_neg_emb[:self.num_users]
        item_neg_emb = final_neg_emb[self.num_users:]

        return user_pos_emb, item_pos_emb, user_neg_emb, item_neg_emb

    def get_embedding_regularization(self):
        """Compute L2 regularization loss for embeddings."""
        reg = (
            self.user_pos_embedding.pow(2).sum() +
            self.item_pos_embedding.pow(2).sum() +
            self.user_neg_embedding.pow(2).sum() +
            self.item_neg_embedding.pow(2).sum()
        )
        return self.reg_weight * reg
