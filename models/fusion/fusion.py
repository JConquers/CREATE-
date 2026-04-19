"""
Fusion modules for CREATE++ to combine sequential and graph representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModule(nn.Module):
    """Base class for fusion modules."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim


class LinearFusion(FusionModule):
    """Linear weighted sum fusion (original CREATE)."""

    def __init__(self, embedding_dim, w_local=1.0, w_global=1.0):
        super().__init__(embedding_dim)
        self.w_local = w_local
        self.w_global = w_global

    def forward(self, local_emb, global_emb):
        return self.w_local * local_emb + self.w_global * global_emb


class MLPFusion(FusionModule):
    """Non-linear MLP + GELU fusion (CREATE++ improvement).

    Uses a two-layer MLP with GELU activation to learn a non-linear combination
    of sequential (local) and graph (global) embeddings.
    """

    def __init__(self, embedding_dim, hidden_dim=None):
        super().__init__(embedding_dim)
        hidden_dim = hidden_dim or embedding_dim * 2

        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, local_emb, global_emb):
        """
        Args:
            local_emb: FloatTensor from sequential encoder (n_items, embedding_dim)
            global_emb: FloatTensor from graph encoder (n_items, embedding_dim)
        Returns:
            fused_emb: FloatTensor (n_items, embedding_dim)
        """
        combined = torch.cat([local_emb, global_emb], dim=-1)
        return self.fusion_mlp(combined)
