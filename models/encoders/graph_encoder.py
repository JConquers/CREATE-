"""PoneGNN graph encoder implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGINConv(MessagePassing):
    """LightGCN-style GIN convolution with normalized aggregation."""

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
        def get_norm(node, edge_index):
            row, col = edge_index
            deg = degree(col, node.size(0), dtype=node.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm, deg_inv_sqrt

        def gin_norm(out, input_x, deg_inv_sqrt):
            norm_self = deg_inv_sqrt * deg_inv_sqrt
            norm_self = norm_self.unsqueeze(dim=1).repeat(1, input_x.size(1))
            return out + (1 + self.eps) * norm_self * input_x

        norm_pos, deg_inv_sqrt_pos = get_norm(x[0], pos_edge_index)
        norm_neg, deg_inv_sqrt_neg = get_norm(x[1], neg_edge_index)

        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x[0], norm=norm_pos)
            out_neg = self.propagate(neg_edge_index, x=x[0], norm=norm_neg)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[0], deg_inv_sqrt_neg)
            return out_pos, out_neg
        else:
            out_pos = self.propagate(pos_edge_index, x=x[0], norm=norm_pos)
            out_neg = self.propagate(pos_edge_index, x=x[1], norm=norm_pos)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[1], deg_inv_sqrt_pos)
            return out_pos, out_neg

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class PoneGNNEncoder(nn.Module):
    """PoneGNN encoder with signed graph convolutions.

    This implementation follows the PoneGNN paper with dual embeddings
    for positive and negative feedback learning.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 2,
        reg: float = 1e-4,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg = reg

        # Positive and negative embeddings
        self.user_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))
        self.user_neg_embedding = nn.Parameter(torch.empty(num_users, embedding_dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(num_items, embedding_dim))

        # Initialize embeddings
        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)

        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.conv_layers.append(LightGINConv(embedding_dim, embedding_dim, True))
            else:
                self.conv_layers.append(LightGINConv(embedding_dim, embedding_dim, False))

        # Cache for latest embeddings (used in loss computation)
        self.pos_emb = None
        self.neg_emb = None

    def forward(self, pos_edge_index: torch.Tensor,
                neg_edge_index: torch.Tensor) -> tuple:
        """
        Forward pass for PoneGNN encoder.

        Args:
            pos_edge_index: Positive edge indices (2, num_pos_edges)
            neg_edge_index: Negative edge indices (2, num_neg_edges)

        Returns:
            Tuple of (positive_embeddings, negative_embeddings)
        """
        alpha = 1.0 / (self.num_layers + 1)

        # Concatenate user and item embeddings
        ego_pos = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        ego_neg = torch.cat([self.user_neg_embedding, self.item_neg_embedding], dim=0)

        # Initialize layer outputs
        pos_emb = ego_pos * alpha
        neg_emb = ego_neg * alpha

        ego_embeddings = (ego_pos, ego_neg)

        # Apply graph convolutions
        for i in range(self.num_layers):
            ego_embeddings = self.conv_layers[i](
                ego_embeddings, pos_edge_index, neg_edge_index
            )
            pos_emb = pos_emb + ego_embeddings[0] * alpha
            neg_emb = neg_emb + ego_embeddings[1] * alpha

        # Cache embeddings for loss computation
        self.pos_emb = pos_emb
        self.neg_emb = neg_emb

        return pos_emb, neg_emb

    def get_embeddings(
        self, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor
    ) -> tuple:
        """
        Get user and item embeddings separately.

        Returns:
            Tuple of (user_pos, user_neg, item_pos, item_neg) embeddings
        """
        pos_emb, neg_emb = self(pos_edge_index, neg_edge_index)
        user_pos, item_pos = torch.split(pos_emb, [self.num_users, self.num_items], dim=0)
        user_neg, item_neg = torch.split(neg_emb, [self.num_users, self.num_items], dim=0)
        return user_pos, user_neg, item_pos, item_neg
