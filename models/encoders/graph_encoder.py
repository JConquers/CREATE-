"""PoneGNN graph encoder implementation with dual embeddings and contrastive learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGINConv(MessagePassing):
    """LightGCN-style GIN convolution with normalized aggregation for signed graphs.

    Propagates messages through positive and negative edges separately,
    enabling learning from both positive and negative user feedback.
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
    """PoneGNN encoder with signed graph convolutions and contrastive learning.

    This implementation follows the PoneGNN paper with dual embeddings
    for positive and negative feedback learning, plus contrastive loss
    for aligning the two embedding spaces.

    Args:
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimension
        num_layers: Number of graph convolution layers
        reg: L2 regularization coefficient
        temperature: Temperature parameter for contrastive loss
        contrastive_weight: Weight for contrastive loss
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

    def compute_contrastive_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE-style contrastive loss for aligning positive and negative embedding spaces.

        Following the Pone-GNN paper, this loss encourages:
        - Positive pairs (user, positive_item) to be similar in both spaces
        - Negative pairs to be dissimilar

        Args:
            users: User indices (batch_size,)
            pos_items: Positive item indices (batch_size,)
            neg_items: Negative sample indices (batch_size,)

        Returns:
            Contrastive loss scalar
        """
        if self.pos_emb is None or self.neg_emb is None:
            return torch.tensor(0.0, device=self.user_embedding.device)

        # Get embeddings for the batch
        # Users are in the first num_users rows, items are in the remaining num_items rows
        u_pos = self.pos_emb[:self.num_users][users]
        u_neg = self.neg_emb[:self.num_users][users]
        i_pos = self.pos_emb[self.num_users:][pos_items]
        i_neg = self.neg_emb[self.num_users:][pos_items]
        n_pos = self.pos_emb[self.num_users:][neg_items]
        n_neg = self.neg_emb[self.num_users:][neg_items]

        # Normalize embeddings
        u_pos_norm = F.normalize(u_pos, dim=1)
        u_neg_norm = F.normalize(u_neg, dim=1)
        i_pos_norm = F.normalize(i_pos, dim=1)
        i_neg_norm = F.normalize(i_neg, dim=1)

        # Compute similarities
        pos_similarity = (u_pos_norm * i_pos_norm).sum(dim=1)  # Positive space alignment
        neg_similarity = (u_neg_norm * i_neg_norm).sum(dim=1)  # Negative space alignment

        # InfoNCE-style loss: pull positive pairs together, push negative pairs apart
        # exp(pos_sim / tau) / (exp(pos_sim / tau) + exp(neg_sim / tau))
        pos_exp = torch.exp(pos_similarity / self.temperature)
        neg_exp = torch.exp(neg_similarity / self.temperature)

        contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8)).mean()

        return self.contrastive_weight * contrastive_loss

    def compute_dual_feedback_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> tuple:
        """
        Compute dual-feedback BPR loss for positive and negative interactions.

        Positive feedback (rating > 3.5): Learn to rank positive items higher
        Negative feedback (rating < 3.5): Learn to rank negative items lower

        Args:
            users: User indices
            pos_items: Positive item indices
            neg_items: Negative sample indices
            weights: Optional rating weights (offset from 3.5)

        Returns:
            Tuple of (positive_bpr_loss, negative_bpr_loss, total_loss)
        """
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
        pos_bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Negative BPR: maximize u_neg · n_neg - u_neg · i_neg
        # Push away from negative items
        neg_scores_neg = (u_neg * n_neg).sum(dim=1)
        pos_scores_neg = (u_neg * i_neg).sum(dim=1)
        neg_bpr_loss = -F.logsigmoid(neg_scores_neg - pos_scores_neg).mean()

        total_loss = pos_bpr_loss + neg_bpr_loss

        return pos_bpr_loss, neg_bpr_loss, total_loss

    def compute_orthogonal_loss(self, users: torch.Tensor) -> torch.Tensor:
        """
        Compute orthogonal loss to decorrelate positive and negative user embeddings.

        Encourages positive and negative embeddings to be orthogonal (uncorrelated),
        enabling them to learn distinct representations.

        Args:
            users: User indices

        Returns:
            Orthogonal regularization loss scalar
        """
        if self.pos_emb is None or self.neg_emb is None:
            return torch.tensor(0.0, device=self.user_embedding.device)

        u_pos = self.pos_emb[:self.num_users][users]
        u_neg = self.neg_emb[:self.num_users][users]

        # Normalize
        u_pos_norm = F.normalize(u_pos, dim=1)
        u_neg_norm = F.normalize(u_neg, dim=1)

        # Cosine similarity - should be close to 0 (orthogonal)
        similarity = (u_pos_norm * u_neg_norm).sum(dim=1)

        # Penalize non-orthogonality
        return (similarity ** 2).mean()

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
        """
        Compute total loss for PoneGNN pre-training.

        Combines dual-feedback loss, orthogonal loss, and contrastive loss.

        Args:
            users: User indices
            pos_items: Positive item indices
            weights: Rating weights (unused in current implementation)
            neg_items: Negative sample indices
            pos_edge_index: Positive edge indices
            neg_edge_index: Negative edge indices
            epoch: Current epoch (for contrastive loss triggering)

        Returns:
            Total loss scalar
        """
        # Forward pass
        pos_emb, neg_emb = self(pos_edge_index, neg_edge_index)

        # Dual-feedback loss
        _, _, dual_loss = self.compute_dual_feedback_loss(
            users=users,
            pos_items=pos_items,
            neg_items=neg_items,
        )

        # Orthogonal loss
        ortho_loss = self.compute_orthogonal_loss(users=users)

        # Contrastive loss (every 10 epochs)
        contrastive_loss = self.compute_contrastive_loss(
            users=users,
            pos_items=pos_items,
            neg_items=neg_items,
        )

        # Only apply contrastive loss every 10 epochs
        if epoch % 10 != 1 and epoch != 1:
            contrastive_loss = torch.tensor(0.0, device=users.device)

        return dual_loss + ortho_loss + contrastive_loss
