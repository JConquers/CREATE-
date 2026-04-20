"""
Sequential encoders for CREATE++: SASRec and BERT4Rec.
"""

import torch
import torch.nn as nn
import math


class SequentialEncoder(nn.Module):
    """Base class for sequential encoders."""

    def __init__(self, n_items, embedding_dim, max_seq_len, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        self.item_embedding = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len + 1, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout

    def get_attention_mask(self, seq):
        """Create causal attention mask. Shape (seq_len, seq_len), no batch dim."""
        seq_len = seq.size(1)
        return torch.tril(torch.ones(seq_len, seq_len, device=seq.device))


class SASRec(SequentialEncoder):
    """SASRec: Self-Attentive Sequential Recommendation.

    Uses causal (lower triangular) attention for next-item prediction.
    """

    def __init__(self, n_items, embedding_dim=64, max_seq_len=50, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__(n_items, embedding_dim, max_seq_len, n_heads, n_layers, dropout)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        ) for _ in range(n_layers)])

    def forward(self, seq):
        """
        Args:
            seq: LongTensor of shape (batch_size, seq_len) with item indices
        Returns:
            item_embeddings: FloatTensor of shape (batch_size, seq_len, embedding_dim)
        """
        seq_len = seq.size(1)
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand_as(seq)
        pos_emb = self.pos_embedding(positions)

        embeddings = self.item_embedding(seq) + pos_emb
        embeddings = self.dropout(embeddings)

        attention_mask = self.get_attention_mask(seq)

        for i in range(self.n_layers):
            attn_out, _ = self.attention_layers[i](embeddings, embeddings, embeddings,
                                                     attn_mask=attention_mask)
            attn_out = self.dropout(attn_out)
            embeddings = self.norm_layers[i](embeddings + attn_out)

            ffn_out = self.ffn_layers[i](embeddings)
            embeddings = self.norm_layers[i](embeddings + ffn_out)

        return embeddings


class BERT4Rec(SequentialEncoder):
    """BERT4Rec: Sequential Recommendation with Bidirectional Transformer.

    Uses bidirectional attention (no causal mask) and is trained with masked item prediction.
    """

    def __init__(self, n_items, embedding_dim=64, max_seq_len=50, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__(n_items, embedding_dim, max_seq_len, n_heads, n_layers, dropout)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(n_layers * 2)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        ) for _ in range(n_layers)])

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, seq):
        """
        Args:
            seq: LongTensor of shape (batch_size, seq_len) with item indices
        Returns:
            item_embeddings: FloatTensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = seq.shape

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        positions = torch.arange(seq_len + 1, device=seq.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)

        embeddings = self.item_embedding(seq)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        embeddings = embeddings + pos_emb
        embeddings = self.dropout(embeddings)

        for i in range(self.n_layers):
            attn_out, _ = self.attention_layers[i](embeddings, embeddings, embeddings)
            attn_out = self.dropout(attn_out)
            embeddings = self.norm_layers[i * 2](embeddings + attn_out)

            ffn_out = self.ffn_layers[i](embeddings)
            embeddings = self.norm_layers[i * 2 + 1](embeddings + ffn_out)

        return embeddings[:, 1:, :]
