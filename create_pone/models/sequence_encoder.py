"""SASRec-style causal transformer encoder for CREATE-Pone."""

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """Causal transformer encoder for next-item prediction."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_sequence_length: int,
    ):
        super().__init__()

        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = item_embeddings.shape

        positions = (
            torch.arange(seq_len, device=item_embeddings.device)
            .unsqueeze(0)
            .expand(batch_size, seq_len)
        )
        sequence_inputs = item_embeddings + self.position_embedding(positions)
        sequence_inputs = self.layer_norm(sequence_inputs)
        sequence_inputs = self.dropout(sequence_inputs)
        sequence_inputs = sequence_inputs.masked_fill(
            ~attention_mask.unsqueeze(-1),
            0.0,
        )

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=item_embeddings.device, dtype=torch.bool),
            diagonal=1,
        )

        encoded = self.encoder(
            src=sequence_inputs,
            mask=causal_mask,
            src_key_padding_mask=~attention_mask,
        )

        return encoded

    @staticmethod
    def get_last_hidden(encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(encoded.size(0), device=encoded.device)
        return encoded[batch_index, lengths]
