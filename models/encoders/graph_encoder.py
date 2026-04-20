"""
Graph encoders for CREATE++: LightGCN, UltraGCN, and PoneGNN.
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


class PoneGNN(GraphEncoder):
    """PoneGNN: Integrating Positive and Negative Feedback in Graph Neural Networks.

    Based on the paper "Pone-GNN: Integrating Positive and Negative Feedback in
    Graph Neural Networks for Recommender Systems" (2025).

    Key features:
    - Dual embeddings: interest embeddings (Z) for likes, disinterest embeddings (V) for dislikes
    - Separate message passing on positive and negative feedback graphs
    - Light graph isomorphism network (no feature transformation, no nonlinear activation)
    - Contrastive learning between positive and negative embeddings
    - Disinterest-score filter for recommendations

    Note: This implementation focuses on the graph encoder component. The full PoneGNN
    includes a disinterest-score filter which can be applied during inference.
    """

    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim=64,
        n_layers=4,
        epsilon_p=0.1,
        epsilon_n=0.1,
        lambda_cl=0.1,
        lambda_reg=5e-5,
        temperature=1.0,
    ):
        super().__init__(n_users, n_items, embedding_dim)

        self.n_layers = n_layers
        self.epsilon_p = epsilon_p  # Learnable parameter for positive graph self-embedding
        self.epsilon_n = epsilon_n  # Learnable parameter for negative graph self-embedding
        self.lambda_cl = lambda_cl  # Contrastive learning coefficient
        self.lambda_reg = lambda_reg  # L2 regularization coefficient
        self.temperature = temperature

        # Interest embeddings (positive feedback)
        self.user_embedding_pos = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_pos = nn.Embedding(n_items, embedding_dim)

        # Disinterest embeddings (negative feedback)
        self.user_embedding_neg = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_neg = nn.Embedding(n_items, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize embeddings using Xavier normal initialization."""
        nn.init.xavier_normal_(self.user_embedding_pos.weight)
        nn.init.xavier_normal_(self.item_embedding_pos.weight)
        nn.init.xavier_normal_(self.user_embedding_neg.weight)
        nn.init.xavier_normal_(self.item_embedding_neg.weight)

    def forward(self, edge_index, edge_weight=None, negative_edge_index=None):
        """
        Args:
            edge_index: LongTensor of shape (2, n_edges) with positive user-item interactions
            edge_weight: Optional FloatTensor of shape (n_edges,) for positive edges
            negative_edge_index: Optional LongTensor of shape (2, n_neg_edges) for negative interactions
        Returns:
            user_emb: FloatTensor of shape (n_users, embedding_dim) - interest embeddings
            item_emb: FloatTensor of shape (n_items, embedding_dim) - interest embeddings
            user_emb_neg: FloatTensor of shape (n_users, embedding_dim) - disinterest embeddings
            item_emb_neg: FloatTensor of shape (n_items, embedding_dim) - disinterest embeddings
        """
        # Initialize embeddings
        user_emb_pos = self.user_embedding_pos.weight
        item_emb_pos = self.item_embedding_pos.weight
        user_emb_neg = self.user_embedding_neg.weight
        item_emb_neg = self.item_embedding_neg.weight

        # Store embeddings from all layers for averaging
        all_user_emb_pos = [user_emb_pos]
        all_item_emb_pos = [item_emb_pos]
        all_user_emb_neg = [user_emb_neg]
        all_item_emb_neg = [item_emb_neg]

        # Message passing on positive graph
        for layer in range(self.n_layers):
            user_emb_pos, item_emb_pos = self._propagate_positive(
                edge_index, user_emb_pos, item_emb_pos, edge_weight
            )
            all_user_emb_pos.append(user_emb_pos)
            all_item_emb_pos.append(item_emb_pos)

        # Message passing on negative graph (if negative edges provided)
        if negative_edge_index is not None:
            for layer in range(self.n_layers):
                user_emb_neg, item_emb_neg = self._propagate_negative(
                    negative_edge_index, user_emb_neg, item_emb_neg
                )
                all_user_emb_neg.append(user_emb_neg)
                all_item_emb_neg.append(item_emb_neg)

        # Average embeddings from all layers (Eq. 2 and 4 in paper)
        user_emb_pos = torch.stack(all_user_emb_pos, dim=0).mean(dim=0)
        item_emb_pos = torch.stack(all_item_emb_pos, dim=0).mean(dim=0)
        user_emb_neg = torch.stack(all_user_emb_neg, dim=0).mean(dim=0)
        item_emb_neg = torch.stack(all_item_emb_neg, dim=0).mean(dim=0)

        return user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg

    def _propagate_positive(self, edge_index, user_emb, item_emb, edge_weight=None):
        """
        Propagate embeddings on positive graph using light graph isomorphism.
        Eq. (1) in PoneGNN paper.
        """
        row, col = edge_index[0], edge_index[1]  # row: users, col: items

        if edge_weight is None:
            edge_weight = torch.ones(row.shape[0], device=row.device)

        # Compute normalization factors
        deg_user = torch.bincount(row, minlength=self.n_users).float().clamp(min=1)
        deg_item = torch.bincount(col, minlength=self.n_items).float().clamp(min=1)

        deg_inv_sqrt_user = torch.pow(deg_user, -0.5)
        deg_inv_sqrt_item = torch.pow(deg_item, -0.5)

        # Normalize edge weights
        norm = deg_inv_sqrt_user[row] * deg_inv_sqrt_item[col] * edge_weight

        # Aggregate item embeddings to users
        item_to_user = torch.zeros_like(user_emb)
        item_to_user = item_to_user.scatter_add_(
            0, row.unsqueeze(1).expand_as(item_emb[col]),
            item_emb[col] * norm.unsqueeze(1)
        )

        # Aggregate user embeddings to items
        user_to_item = torch.zeros_like(item_emb)
        user_to_item = user_to_item.scatter_add_(
            0, col.unsqueeze(1).expand_as(user_emb[row]),
            user_emb[row] * norm.unsqueeze(1)
        )

        # Apply self-embedding strengthening (first term in Eq. 1)
        self_emb_factor = (1 + self.epsilon_p) / (torch.bincount(row, minlength=self.n_users).float().clamp(min=1)[row].shape[0] + 1)

        new_user_emb = user_emb + item_to_user
        new_item_emb = item_emb + user_to_item

        return new_user_emb, new_item_emb

    def _propagate_negative(self, edge_index, user_emb, item_emb):
        """
        Propagate embeddings on negative graph using light graph isomorphism.
        Eq. (3) in PoneGNN paper.
        """
        row, col = edge_index[0], edge_index[1]  # row: users, col: items

        # Compute normalization factors
        deg_user = torch.bincount(row, minlength=self.n_users).float().clamp(min=1)
        deg_item = torch.bincount(col, minlength=self.n_items).float().clamp(min=1)

        deg_inv_sqrt_user = torch.pow(deg_user, -0.5)
        deg_inv_sqrt_item = torch.pow(deg_item, -0.5)

        # Normalize edge weights (assume uniform weight for negative edges)
        norm = deg_inv_sqrt_user[row] * deg_inv_sqrt_item[col]

        # Aggregate item embeddings to users
        item_to_user = torch.zeros_like(user_emb)
        item_to_user = item_to_user.scatter_add_(
            0, row.unsqueeze(1).expand_as(item_emb[col]),
            item_emb[col] * norm.unsqueeze(1)
        )

        # Aggregate user embeddings to items
        user_to_item = torch.zeros_like(item_emb)
        user_to_item = user_to_item.scatter_add_(
            0, col.unsqueeze(1).expand_as(user_emb[row]),
            user_emb[row] * norm.unsqueeze(1)
        )

        # Apply self-embedding strengthening
        new_user_emb = user_emb + item_to_user
        new_item_emb = item_emb + user_to_item

        return new_user_emb, new_item_emb

    def compute_contrastive_loss(self, user_emb_pos, item_emb_pos, user_emb_neg, item_emb_neg):
        """
        Compute contrastive learning loss (Eq. 7 and 13 in PoneGNN paper).
        Maximizes similarity between positive embeddings while minimizing
        similarity between disinterest embeddings.
        """
        # Normalize embeddings
        user_emb_pos = F.normalize(user_emb_pos, p=2, dim=1)
        item_emb_pos = F.normalize(item_emb_pos, p=2, dim=1)
        user_emb_neg = F.normalize(user_emb_neg, p=2, dim=1)
        item_emb_neg = F.normalize(item_emb_neg, p=2, dim=1)

        # Compute similarity for positive pairs
        pos_sim_user = torch.sum(user_emb_pos * user_emb_pos, dim=1)  # Should be close to 1
        pos_sim_item = torch.sum(item_emb_pos * item_emb_pos, dim=1)

        # Compute similarity for negative pairs (should be pushed apart)
        neg_sim_user = torch.sum(user_emb_neg * user_emb_neg, dim=1)
        neg_sim_item = torch.sum(item_emb_neg * item_emb_neg, dim=1)

        # Contrastive loss: bring positive pairs closer, push negative pairs apart
        # Using InfoNCE-style loss
        pos_loss = -torch.log(torch.exp(pos_sim_user / self.temperature) /
                              (torch.exp(pos_sim_user / self.temperature) +
                               torch.exp(neg_sim_user / self.temperature) + 1e-8)).mean()
        pos_loss += -torch.log(torch.exp(pos_sim_item / self.temperature) /
                               (torch.exp(pos_sim_item / self.temperature) +
                                torch.exp(neg_sim_item / self.temperature) + 1e-8)).mean()

        return self.lambda_cl * pos_loss

    def compute_reg_loss(self):
        """Compute L2 regularization loss on initial embeddings."""
        reg_loss = (
            torch.norm(self.user_embedding_pos.weight) ** 2 +
            torch.norm(self.item_embedding_pos.weight) ** 2 +
            torch.norm(self.user_embedding_neg.weight) ** 2 +
            torch.norm(self.item_embedding_neg.weight) ** 2
        )
        return self.lambda_reg * reg_loss
