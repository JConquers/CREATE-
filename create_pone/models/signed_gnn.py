"""Dual-branch signed GNN used in CREATE-Pone."""

import torch
import torch.nn.functional as F
from torch import nn

from create_pone.dataset.signed_graph import SignedGraph


class SignedDualGNN(nn.Module):
    """Two independent GNN branches for interest and disinterest propagation."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.dropout = dropout

        self.user_interest_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_interest_embedding = nn.Embedding(num_items, embedding_dim)

        self.user_disinterest_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_disinterest_embedding = nn.Embedding(num_items, embedding_dim)

        # Eq. (7) uses k = 0 .. K-1 states; this needs K-1 propagation steps.
        eps_size = max(num_layers - 1, 1)
        self.epsilon_pos = nn.Parameter(torch.zeros(eps_size))
        self.epsilon_neg = nn.Parameter(torch.zeros(eps_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.user_interest_embedding.weight)
        nn.init.xavier_uniform_(self.item_interest_embedding.weight)
        nn.init.xavier_uniform_(self.user_disinterest_embedding.weight)
        nn.init.xavier_uniform_(self.item_disinterest_embedding.weight)

    @staticmethod
    def _propagate(
        node_embeddings: torch.Tensor,
        adjacency: torch.Tensor,
        degree_inverse: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        neighborhood_messages = torch.sparse.mm(adjacency, node_embeddings)
        residual = (1.0 + epsilon) * degree_inverse.unsqueeze(-1) * node_embeddings
        return neighborhood_messages + residual

    def forward(
        self,
        signed_graph: SignedGraph,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        interest_nodes = torch.cat(
            [
                self.user_interest_embedding.weight,
                self.item_interest_embedding.weight,
            ],
            dim=0,
        )
        disinterest_nodes = torch.cat(
            [
                self.user_disinterest_embedding.weight,
                self.item_disinterest_embedding.weight,
            ],
            dim=0,
        )

        all_interest = [interest_nodes]
        all_disinterest = [disinterest_nodes]

        # Produce exactly K states z^(0)..z^(K-1), v^(0)..v^(K-1) as in Eq. (8).
        for layer_idx in range(self.num_layers - 1):
            interest_nodes = self._propagate(
                node_embeddings=interest_nodes,
                adjacency=signed_graph.pos_adj,
                degree_inverse=signed_graph.pos_deg_inv,
                epsilon=self.epsilon_pos[layer_idx],
            )
            disinterest_nodes = self._propagate(
                node_embeddings=disinterest_nodes,
                adjacency=signed_graph.neg_adj,
                degree_inverse=signed_graph.neg_deg_inv,
                epsilon=self.epsilon_neg[layer_idx],
            )

            if self.dropout > 0:
                interest_nodes = F.dropout(
                    interest_nodes,
                    p=self.dropout,
                    training=self.training,
                )
                disinterest_nodes = F.dropout(
                    disinterest_nodes,
                    p=self.dropout,
                    training=self.training,
                )

            all_interest.append(interest_nodes)
            all_disinterest.append(disinterest_nodes)

        final_interest = torch.stack(all_interest, dim=0).sum(dim=0) / self.num_layers
        final_disinterest = (
            torch.stack(all_disinterest, dim=0).sum(dim=0) / self.num_layers
        )

        interest_user, interest_item = torch.split(
            final_interest,
            [self.num_users, self.num_items],
            dim=0,
        )
        disinterest_user, disinterest_item = torch.split(
            final_disinterest,
            [self.num_users, self.num_items],
            dim=0,
        )

        return interest_user, disinterest_user, interest_item, disinterest_item
