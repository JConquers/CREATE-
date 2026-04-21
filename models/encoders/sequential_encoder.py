"""SASRec sequential encoder implementation."""

import torch
from torch import nn
import torch.nn.functional as F


class SASRecEncoder(nn.Module):
    """SASRec encoder using Transformer encoder layers.

    This implementation follows the original SASRec paper with self-attention
    for sequential pattern modeling in user behavior sequences.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_sequence_length: int = 50,
    ):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length

        # Item and position embeddings
        self.item_embedding = nn.Embedding(num_items + 2, embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_sequence_length + 1, embedding_dim)

        # Pre-compute and cache position indices for common lengths (register as non-learnable buffer)
        self._register_positions(max_sequence_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _register_positions(self, max_len):
        """Pre-compute position indices for common sequence lengths."""
        # Store reverse positions [seq_len-1, seq_len-2, ..., 0] for various lengths
        self.register_buffer('_pos_indices_cache', {}, persistent=False)

    def _get_cached_positions(self, seq_len: int) -> torch.Tensor:
        """Get cached position indices or compute if not cached."""
        if seq_len not in self._pos_indices_cache:
            positions = torch.arange(seq_len - 1, -1, step=-1, device=self.device)
            self._pos_indices_cache[seq_len] = positions
        return self._pos_indices_cache[seq_len]

    def _init_weights(self):
        """Initialize embeddings with Xavier initialization."""
        nn.init.xavier_normal_(self.item_embedding.weight)
        nn.init.xavier_normal_(self.position_embedding.weight)

    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device

    def _create_attention_mask(self, seq_length: int) -> torch.Tensor:
        """Create causal attention mask for self-attention."""
        mask = torch.tril(
            torch.ones(seq_length, seq_length, device=self.device)
        ).bool()
        return ~mask  # Invert: True means mask out

    def forward(self, item_sequences: torch.Tensor,
                mask: torch.Tensor = None) -> dict:
        """
        Forward pass for SASRec encoder.

        Args:
            item_sequences: Batch of item sequences (batch_size, seq_len)
            mask: Boolean mask for valid positions (batch_size, seq_len)

        Returns:
            Dictionary containing:
                - 'sequence_output': Output embeddings for each position
                - 'user_embedding': Final user representation (last item)
                - 'item_scores': Scores for all items
        """
        batch_size, seq_len = item_sequences.shape

        # Get embeddings with scaling
        item_emb = self.item_embedding(item_sequences) * (self.embedding_dim ** 0.5)

        # Use cached or compute positions
        positions = self._get_cached_positions(seq_len)
        pos_emb = self.position_embedding(positions.unsqueeze(0).expand(batch_size, -1))

        # Combine and normalize
        hidden = self.layer_norm(item_emb + pos_emb)
        hidden = self.dropout(hidden)

        # Create attention mask
        attn_mask = self._create_attention_mask(seq_len)

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(
            hidden,
            mask=attn_mask,
            src_key_padding_mask=~mask if mask is not None else None,
        )

        # Get last valid embedding for each user using optimized gather
        lengths = mask.sum(dim=-1) - 1  # Last valid position
        batch_indices = torch.arange(batch_size, device=self.device)
        user_embeddings = transformer_output[batch_indices, lengths]

        # Compute scores for all items
        item_scores = user_embeddings @ self.item_embedding.weight.T

        return {
            'sequence_output': transformer_output,
            'user_embedding': user_embeddings,
            'item_scores': item_scores,
        }

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get embedding for specific item IDs."""
        return self.item_embedding(item_ids)

    def get_all_item_embeddings(self) -> torch.Tensor:
        """Get embeddings for all items."""
        return self.item_embedding.weight
