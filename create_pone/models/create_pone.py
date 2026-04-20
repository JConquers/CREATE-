"""Integrated CREATE-Pone model: signed graph encoder + sequential encoder."""

import torch
from torch import nn

from create_pone.dataset.signed_graph import SignedGraph

from .sequence_encoder import SequenceEncoder
from .signed_gnn import SignedDualGNN


class CreatePoneModel(nn.Module):
    """CREATE-Pone model following the CREATE++ signed variant design."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        gnn_layers: int,
        gnn_dropout: float,
        max_sequence_length: int,
        transformer_heads: int,
        transformer_layers: int,
        transformer_ff_dim: int,
        transformer_dropout: float,
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.pad_id = num_items

        self.signed_gnn = SignedDualGNN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            num_layers=gnn_layers,
            dropout=gnn_dropout,
        )
        self.sequence_encoder = SequenceEncoder(
            embedding_dim=embedding_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            max_sequence_length=max_sequence_length,
        )

    def forward(
        self,
        batch: dict,
        signed_graph: SignedGraph,
        run_sequence: bool,
    ) -> dict:
        (
            interest_user_embeddings,
            disinterest_user_embeddings,
            interest_item_embeddings,
            disinterest_item_embeddings,
        ) = self.signed_gnn(signed_graph)

        outputs = {
            "interest_user_embeddings": interest_user_embeddings,
            "disinterest_user_embeddings": disinterest_user_embeddings,
            "interest_item_embeddings": interest_item_embeddings,
            "disinterest_item_embeddings": disinterest_item_embeddings,
        }

        if not run_sequence:
            return outputs

        pad_row = torch.zeros(
            1,
            interest_item_embeddings.size(1),
            dtype=interest_item_embeddings.dtype,
            device=interest_item_embeddings.device,
        )
        sequence_item_table = torch.cat([interest_item_embeddings, pad_row], dim=0)

        input_item_embeddings = sequence_item_table[batch["input_ids"]]
        encoded_sequence = self.sequence_encoder(
            item_embeddings=input_item_embeddings,
            attention_mask=batch["attention_mask"],
        )

        logits = encoded_sequence @ interest_item_embeddings.t()
        last_hidden = self.sequence_encoder.get_last_hidden(
            encoded=encoded_sequence,
            attention_mask=batch["attention_mask"],
        )

        outputs.update(
            {
                "sequence_hidden": encoded_sequence,
                "sequence_logits": logits,
                "sequence_user_embedding": last_hidden,
            }
        )

        return outputs

    @torch.no_grad()
    def predict_topk(
        self,
        batch: dict,
        signed_graph: SignedGraph,
        topk: int,
    ) -> torch.Tensor:
        self.eval()
        outputs = self.forward(batch=batch, signed_graph=signed_graph, run_sequence=True)
        scores = outputs["sequence_user_embedding"] @ outputs["interest_item_embeddings"].t()
        _, indices = torch.topk(scores, k=min(topk, self.num_items), dim=-1)
        return indices
