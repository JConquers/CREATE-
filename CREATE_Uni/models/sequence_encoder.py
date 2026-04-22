"""
Sequence Encoder for CREATE-Uni.
Implements SASRec and BERT4Rec style transformers for sequential recommendation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, embedding_dim)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SequentialEncoder(nn.Module):
    """
    SASRec-style sequential encoder using transformer encoder with causal masking.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_sequence_length: int = 50,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Item embeddings
        self.item_embeddings = nn.Embedding(
            num_items + 2, embedding_dim, padding_idx=0
        )  # 0=PAD, 1=MASK

        # Position embeddings (learned, reverse order like SASRec)
        self.position_embeddings = nn.Embedding(
            max_sequence_length + 1, embedding_dim
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (triangular) mask for transformer."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        return mask

    def forward(
        self,
        item_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_last: bool = True,
    ):
        """
        Forward pass through sequence encoder.

        Args:
            item_ids: Tensor of shape (batch_size, seq_len) with item indices
            attention_mask: Boolean tensor of shape (batch_size, seq_len), True = real token
            return_last: If True, return only last position embedding; else return all

        Returns:
            sequence_output: Tensor of shape (batch_size, embedding_dim) if return_last else (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = item_ids.shape

        # Get item embeddings
        item_emb = self.item_embeddings(item_ids)  # (B, L, D)
        item_emb *= math.sqrt(self.embedding_dim)

        # Position embeddings (reverse order like SASRec)
        positions = (
            torch.arange(seq_len - 1, -1, -1, device=item_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        pos_mask = positions < attention_mask.sum(dim=1, keepdim=True)
        positions = positions[pos_mask]
        pos_emb = self.position_embeddings(positions)

        # Create masked position embeddings tensor
        pos_emb_full = torch.zeros(
            batch_size, seq_len, self.embedding_dim, device=item_ids.device
        )
        pos_emb_full[pos_mask] = pos_emb

        # Combine embeddings
        seq_emb = item_emb + pos_emb_full
        seq_emb = self.layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # Apply transformer with causal mask
        causal_mask = self._create_causal_mask(seq_len, item_ids.device)
        seq_emb = self.transformer_encoder(
            src=seq_emb,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask,
        )

        if return_last:
            # Get last valid position for each sequence
            lengths = attention_mask.sum(dim=1) - 1  # (B,)
            last_mask = attention_mask.gather(dim=1, index=lengths.unsqueeze(1))
            last_emb = seq_emb.gather(
                dim=1, index=lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.embedding_dim)
            ).squeeze(1)
            return last_emb

        return seq_emb

    def predict(
        self,
        item_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_ids: torch.Tensor = None,
    ):
        """
        Generate predictions for next item.

        Args:
            item_ids: Input sequence (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            candidate_ids: Optional candidate items for scoring

        Returns:
            scores: Item scores (batch_size, num_items) or (batch_size, len(candidate_ids))
        """
        last_emb = self.forward(item_ids, attention_mask, return_last=True)
        scores = last_emb @ self.item_embeddings.weight.T

        if candidate_ids is not None:
            candidate_emb = self.item_embeddings(candidate_ids)
            scores = last_emb @ candidate_emb.T

        return scores


class Bert4RecEncoder(nn.Module):
    """
    BERT4Rec-style bidirectional sequence encoder with MLM objective.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_sequence_length: int = 50,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.mask_id = num_items + 1

        # Item embeddings
        self.item_embeddings = nn.Embedding(
            num_items + 2, embedding_dim, padding_idx=0
        )  # 0=PAD, 1=MASK

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            max_sequence_length + 1, embedding_dim
        )

        # Transformer encoder (bidirectional)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.position_embeddings.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mlm_mask: torch.Tensor = None,
    ):
        """
        Forward pass through BERT4Rec encoder.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with item indices (may contain MASK tokens)
            attention_mask: Boolean tensor of shape (batch_size, seq_len)
            mlm_mask: Boolean tensor indicating positions to predict (batch_size, seq_len)

        Returns:
            masked_embeddings: Embeddings at masked positions (num_masked, embedding_dim)
            all_embeddings: All position embeddings (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        item_emb = self.item_embeddings(input_ids)
        item_emb *= math.sqrt(self.embedding_dim)

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embeddings(positions)

        # Combine
        seq_emb = item_emb + pos_emb
        seq_emb = self.layer_norm(seq_emb)
        seq_emb = self.dropout(seq_emb)

        # Transformer (bidirectional, no causal mask)
        seq_emb = self.transformer_encoder(
            src=seq_emb,
            src_key_padding_mask=~attention_mask,
        )

        if mlm_mask is not None:
            masked_emb = seq_emb[mlm_mask]
            return masked_emb, seq_emb

        return seq_emb

    def predict(
        self,
        item_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Predict next item by masking last position.

        Args:
            item_ids: Input sequence (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            scores: Item scores (batch_size, num_items)
        """
        batch_size, seq_len = item_ids.shape
        device = item_ids.device

        # Create extended input with MASK at the end
        extended_ids = torch.full(
            (batch_size, seq_len + 1), self.mask_id, dtype=torch.long, device=device
        )
        extended_ids[:, :seq_len] = item_ids

        extended_mask = torch.zeros(
            (batch_size, seq_len + 1), dtype=torch.bool, device=device
        )
        extended_mask[:, : seq_len + 1] = True

        # Get embedding at last position
        emb, _ = self.forward(extended_ids, extended_mask, mlm_mask=None)
        last_emb = emb[:, -1, :]  # (B, D)

        # Score against all items
        scores = last_emb @ self.item_embeddings.weight.T
        scores[:, 0] = -torch.inf  # PAD
        scores[:, self.mask_id:] = -torch.inf  # MASK and beyond

        return scores
