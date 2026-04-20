"""Signed graph construction and mini-batch triplet sampling for CREATE-Pone."""

import random
from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd
import torch


@dataclass
class SignedGraph:
    """Sparse signed graph tensors used by the dual GNN branches."""

    pos_adj: torch.Tensor
    neg_adj: torch.Tensor
    pos_deg_inv: torch.Tensor
    neg_deg_inv: torch.Tensor


def _empty_sparse_matrix(num_nodes: int, device: torch.device) -> torch.Tensor:
    indices = torch.empty((2, 0), dtype=torch.long, device=device)
    values = torch.empty((0,), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()


def _build_normalized_bipartite_adjacency(
    interaction_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = num_users + num_items

    if interaction_df.empty:
        return _empty_sparse_matrix(num_nodes, device), torch.zeros(
            num_nodes, dtype=torch.float32, device=device
        )

    dedup_df = interaction_df[["user_id", "item_id"]].drop_duplicates()
    user_nodes = torch.tensor(dedup_df["user_id"].values, dtype=torch.long, device=device)
    item_nodes = (
        torch.tensor(dedup_df["item_id"].values, dtype=torch.long, device=device) + num_users
    )

    src = torch.cat([user_nodes, item_nodes], dim=0)
    dst = torch.cat([item_nodes, user_nodes], dim=0)

    degree = torch.bincount(src, minlength=num_nodes).float()

    deg_inv_sqrt = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))
    deg_inv = torch.where(degree > 0, degree.reciprocal(), torch.zeros_like(degree))

    norm_values = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([src, dst], dim=0)

    adjacency = torch.sparse_coo_tensor(
        indices,
        norm_values,
        (num_nodes, num_nodes),
    ).coalesce()

    return adjacency, deg_inv


def build_signed_graph(
    train_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    pos_threshold: float,
    neg_threshold: float,
    device: torch.device,
) -> SignedGraph:
    """Create normalized sparse adjacency matrices for positive and negative graphs."""

    positive_df = train_df[train_df["rating"] >= pos_threshold]
    negative_df = train_df[train_df["rating"] <= neg_threshold]

    pos_adj, pos_deg_inv = _build_normalized_bipartite_adjacency(
        positive_df,
        num_users,
        num_items,
        device,
    )
    neg_adj, neg_deg_inv = _build_normalized_bipartite_adjacency(
        negative_df,
        num_users,
        num_items,
        device,
    )

    return SignedGraph(
        pos_adj=pos_adj,
        neg_adj=neg_adj,
        pos_deg_inv=pos_deg_inv,
        neg_deg_inv=neg_deg_inv,
    )


class SignedTripleSampler:
    """Sample CREATE++ style positive and negative triplets for each mini-batch."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
        pos_threshold: float,
        neg_threshold: float,
        seed: int,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self._rng = random.Random(seed)

        self._pos_sets: List[Set[int]] = [set() for _ in range(num_users)]
        self._neg_sets: List[Set[int]] = [set() for _ in range(num_users)]

        positive_df = train_df[train_df["rating"] >= pos_threshold][["user_id", "item_id"]]
        negative_df = train_df[train_df["rating"] <= neg_threshold][["user_id", "item_id"]]

        for row in positive_df.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)
            self._pos_sets[user_id].add(item_id)

        for row in negative_df.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)
            self._neg_sets[user_id].add(item_id)

        self._pos_items_by_user = [sorted(items) for items in self._pos_sets]
        self._neg_items_by_user = [sorted(items) for items in self._neg_sets]

    def _sample_from(self, items: List[int]) -> int:
        index = self._rng.randrange(len(items))
        return items[index]

    def _sample_excluding(self, excluded: Set[int]) -> int | None:
        if len(excluded) >= self.num_items:
            return None

        for _ in range(25):
            candidate = self._rng.randrange(self.num_items)
            if candidate not in excluded:
                return candidate

        for candidate in range(self.num_items):
            if candidate not in excluded:
                return candidate

        return None

    @staticmethod
    def _to_tensor(values: List[int], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(values, dtype=torch.long, device=device)

    def sample(self, user_ids: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample mini-batch triplets Bp and Bn from batch users."""

        pos_users: List[int] = []
        pos_items: List[int] = []
        pos_negs: List[int] = []

        neg_users: List[int] = []
        neg_items: List[int] = []
        neg_negs: List[int] = []

        for user_id in user_ids.detach().cpu().tolist():
            user_id = int(user_id)

            positive_items = self._pos_items_by_user[user_id]
            if positive_items:
                positive_item = self._sample_from(positive_items)
                positive_negative = self._sample_excluding(self._pos_sets[user_id])
                if positive_negative is not None:
                    pos_users.append(user_id)
                    pos_items.append(positive_item)
                    pos_negs.append(positive_negative)

            negative_items = self._neg_items_by_user[user_id]
            if negative_items:
                negative_item = self._sample_from(negative_items)
                negative_negative = self._sample_excluding(self._neg_sets[user_id])
                if negative_negative is not None:
                    neg_users.append(user_id)
                    neg_items.append(negative_item)
                    neg_negs.append(negative_negative)

        return {
            "pos_users": self._to_tensor(pos_users, device),
            "pos_items": self._to_tensor(pos_items, device),
            "pos_negs": self._to_tensor(pos_negs, device),
            "neg_users": self._to_tensor(neg_users, device),
            "neg_items": self._to_tensor(neg_items, device),
            "neg_negs": self._to_tensor(neg_negs, device),
        }
