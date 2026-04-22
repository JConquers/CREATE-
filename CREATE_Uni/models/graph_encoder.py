"""
Graph Encoder for CREATE-Uni using UniGNN convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unignn_conv import (
    UniGCNConv,
    UniGINConv,
    UniSAGEConv,
    UniGATConv,
    UniGCNIIConv,
)


class UniGNNEncoder(nn.Module):
    """
    Graph encoder using UniGNN convolutions for user-item interactions.

    This encoder builds a bipartite graph between users and items,
    then applies UniGNN message passing to learn embeddings.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int = 2,
        conv_type: str = "UniSAGE",
        heads: int = 8,
        dropout: float = 0.1,
        use_norm: bool = True,
        first_aggregate: str = "mean",
        second_aggregate: str = "sum",
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.conv_type = conv_type

        # User and item embeddings
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        # Select convolution type
        conv_map = {
            "UniGCN": UniGCNConv,
            "UniGIN": UniGINConv,
            "UniSAGE": UniSAGEConv,
            "UniGAT": UniGATConv,
        }
        ConvClass = conv_map.get(conv_type, UniSAGEConv)

        # Build convolution layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_dim = embedding_dim if i == 0 else embedding_dim * heads
            out_dim = embedding_dim
            conv = ConvClass(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=heads if conv_type == "UniGAT" else 1,
                dropout=dropout,
                first_aggregate=first_aggregate,
                second_aggregate=second_aggregate,
                use_norm=use_norm,
            )
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.use_norm = use_norm

    def forward(
        self,
        vertex: torch.Tensor,
        edges: torch.Tensor,
        degE: torch.Tensor = None,
        degV: torch.Tensor = None,
    ):
        """
        Forward pass through the graph encoder.

        Args:
            vertex: Row indices of hypergraph incidence matrix (2*E,)
            edges: Column indices of hypergraph incidence matrix (2*E,)
            degE: Edge degree normalization (E,) - optional, defaults to uniform
            degV: Vertex degree normalization (N,) - optional, defaults to uniform

        Returns:
            user_embeddings: Updated user embeddings (num_users, D)
            item_embeddings: Updated item embeddings (num_items, D)
        """
        N = self.num_users + self.num_items
        E = edges.max().item() + 1  # Number of edges

        # Combine user and item embeddings
        ego_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )

        # Default degree normalization if not provided
        if degV is None:
            degV = torch.ones(N, device=ego_embeddings.device)
        if degE is None:
            degE = torch.ones(E, device=ego_embeddings.device)

        # Apply UniGNN layers
        for conv in self.convs:
            ego_embeddings = self.dropout(ego_embeddings)
            ego_embeddings = conv(
                X=ego_embeddings,
                vertex=vertex,
                edges=edges,
                degE=degE,
                degV=degV,
            )
            ego_embeddings = F.relu(ego_embeddings)

        # Split back into user and item embeddings
        user_final = ego_embeddings[: self.num_users]
        item_final = ego_embeddings[self.num_users :]

        return user_final, item_final


class LightGCNStyleEncoder(nn.Module):
    """
    Simplified graph encoder inspired by LightGCN with UniGNN-style propagation.

    Uses averaging of embeddings across layers like LightGCN.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        alpha: float = 0.9,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.alpha = alpha

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, adj: torch.sparse.FloatTensor):
        """
        Forward pass using sparse matrix multiplication.

        Args:
            adj: Normalized adjacency matrix (sparse)

        Returns:
            user_embeddings: Final user embeddings
            item_embeddings: Final item embeddings
        """
        ego_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)

        # Layer combination
        all_embeddings = torch.stack(all_embeddings, dim=-1)
        all_embeddings = all_embeddings.mean(dim=-1)

        user_final = all_embeddings[: self.num_users]
        item_final = all_embeddings[self.num_users :]

        return user_final, item_final
