"""
Sequential Encoder for CREATE-Pone.
Implements SASRec-style transformer encoder with cross-representation alignment.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for sequential recommendations."""

    def __init__(self, embedding_dim: int, max_len: int = 500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        return x + self.pe[:x.size(1)]


class SequentialEncoder(nn.Module):
    """
    SASRec-style sequential encoder with alignment support.

    Uses a transformer encoder to model sequential patterns in user interactions.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Item embeddings (shared with graph encoder)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_normal_(self.item_embedding.weight)

    def create_causal_mask(self, seq_len, device):
        """Create causal attention mask for transformer."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, item_sequence, attention_mask=None):
        """
        Encode item sequence.

        Args:
            item_sequence: Tensor of shape (batch_size, seq_len) with item IDs
            attention_mask: Boolean mask of shape (batch_size, seq_len), True for valid tokens

        Returns:
            sequence_output: Tensor of shape (batch_size, seq_len, embedding_dim)
            user_output: Tensor of shape (batch_size, embedding_dim) - last position embedding
        """
        # Get item embeddings
        item_emb = self.item_embedding(item_sequence)  # (B, L, D)
        item_emb *= math.sqrt(self.embedding_dim)

        # Add positional encoding
        item_emb = self.pos_encoding(item_emb)
        item_emb = self.dropout(item_emb)

        # Create causal mask
        seq_len = item_sequence.size(1)
        causal_mask = self.create_causal_mask(seq_len, device=item_sequence.device)

        # Apply transformer
        transformer_out = self.transformer(
            src=item_emb,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask if attention_mask is not None else None,
        )

        # Layer normalization
        transformer_out = self.layer_norm(transformer_out)

        # Get user representation (last valid position)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            batch_indices = torch.arange(item_sequence.size(0), device=item_sequence.device)
            user_output = transformer_out[batch_indices, lengths.clamp(min=0)]
        else:
            user_output = transformer_out[:, -1]

        return transformer_out, user_output

    def get_item_embeddings(self):
        """Return full item embedding matrix."""
        return self.item_embedding.weight


class AlignmentModule(nn.Module):
    """
    Cross-representation alignment module (Barlow Twins style).

    Aligns sequential encoder representations with graph encoder representations
    while maintaining orthogonality to disinterest embeddings.

    Implements Eq. 15 from CREATE++ paper:
    L_align = sum_i (1 - C_hz_ii)^2 + lambda * sum_{i!=j} (C_hz_ij)^2 + mu * sum_{i,j} (C_hv_ij)^2

    Where:
    - C_hz: cross-correlation between seq_emb and graph interest (pos) embeddings
    - C_hv: cross-correlation between seq_emb and graph disinterest (neg) embeddings
    - First term: invariance (alignment)
    - Second term: redundancy reduction (push off-diagonal to 0)
    - Third term: orthogonality (push seq away from disinterest)
    """

    def __init__(self, embedding_dim: int, lambda_param: float = 0.1, mu_param: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lambda_param = lambda_param  # Weight for off-diagonal (redundancy reduction)
        self.mu_param = mu_param  # Weight for orthogonality to disinterest

    def forward(self, seq_emb, graph_pos_emb, graph_neg_emb=None):
        """
        Compute alignment and orthogonality losses according to CREATE++ Eq. 15.

        Args:
            seq_emb: Sequential encoder embeddings (B, D)
            graph_pos_emb: Graph encoder interest embeddings (B, D)
            graph_neg_emb: Graph encoder disinterest embeddings (B, D) - optional

        Returns:
            barlow_twins_loss: Alignment + redundancy reduction (Eq. 15 first two terms)
            orthogonality_loss: Orthogonality to disinterest (Eq. 15 third term)
        """
        # Normalize embeddings (batch normalization without affine parameters)
        seq_emb_norm = nn.functional.normalize(seq_emb, dim=1, p=2)
        graph_pos_emb_norm = nn.functional.normalize(graph_pos_emb, dim=1, p=2)

        # Cross-correlation matrix C_hz between seq and graph interest embeddings
        batch_size = seq_emb.size(0)
        cross_corr_hz = (seq_emb_norm.T @ graph_pos_emb_norm) / batch_size  # (D, D)

        # Term 1: Invariance - push diagonal toward 1 (alignment)
        on_diag_hz = torch.diagonal(cross_corr_hz).add(-1).pow(2).sum()

        # Term 2: Redundancy reduction - push off-diagonal toward 0
        off_diag_hz = self._off_diagonal(cross_corr_hz).pow(2).sum()

        # Barlow Twins loss (alignment + redundancy reduction)
        barlow_twins_loss = on_diag_hz + self.lambda_param * off_diag_hz

        # Term 3: Orthogonality - push seq away from disinterest embeddings
        orthogonality_loss = torch.tensor(0.0, device=seq_emb.device)
        if graph_neg_emb is not None:
            graph_neg_emb_norm = nn.functional.normalize(graph_neg_emb, dim=1, p=2)
            # Cross-correlation C_hv between seq and graph disinterest embeddings
            cross_corr_hv = (seq_emb_norm.T @ graph_neg_emb_norm) / batch_size
            # Push ALL elements toward 0 (orthogonality)
            orthogonality_loss = cross_corr_hv.pow(2).sum()

        return barlow_twins_loss, orthogonality_loss

    def _off_diagonal(self, x):
        """Return flattened off-diagonal elements of square matrix."""
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
