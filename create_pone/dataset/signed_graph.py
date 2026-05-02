"""Signed graph construction and mini-batch triplet sampling for CREATE-Pone."""

import random
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
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
        num_negs: int = 1,
        neg_sampling_power: float = 0.0,
        exclude_seen: bool = True,
    ):
        self.num_users = num_users
        self.num_items = num_items
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.num_negs = max(1, int(num_negs))
        self.exclude_seen = exclude_seen

        self._pos_sets: List[Set[int]] = [set() for _ in range(num_users)]
        self._neg_sets: List[Set[int]] = [set() for _ in range(num_users)]
        self._seen_sets: List[Set[int]] = [set() for _ in range(num_users)]

        self._pos_items_by_user: List[List[int]] = [[] for _ in range(num_users)]
        self._pos_ratings_by_user: List[List[float]] = [[] for _ in range(num_users)]
        self._neg_items_by_user: List[List[int]] = [[] for _ in range(num_users)]
        self._neg_ratings_by_user: List[List[float]] = [[] for _ in range(num_users)]

        positive_df = train_df[train_df["rating"] >= pos_threshold][
            ["user_id", "item_id", "rating"]
        ]
        negative_df = train_df[train_df["rating"] <= neg_threshold][
            ["user_id", "item_id", "rating"]
        ]

        for row in positive_df.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)
            rating = float(row.rating)
            self._pos_sets[user_id].add(item_id)
            self._seen_sets[user_id].add(item_id)
            self._pos_items_by_user[user_id].append(item_id)
            self._pos_ratings_by_user[user_id].append(rating)

        for row in negative_df.itertuples(index=False):
            user_id = int(row.user_id)
            item_id = int(row.item_id)
            rating = float(row.rating)
            self._neg_sets[user_id].add(item_id)
            self._seen_sets[user_id].add(item_id)
            self._neg_items_by_user[user_id].append(item_id)
            self._neg_ratings_by_user[user_id].append(rating)

        self._all_items = np.arange(self.num_items, dtype=np.int64)
        item_counts = (
            train_df["item_id"].value_counts().reindex(range(self.num_items), fill_value=0)
        )
        weights = item_counts.to_numpy(dtype=np.float64)
        if neg_sampling_power > 0:
            weights = np.power(weights, float(neg_sampling_power))
        if weights.sum() <= 0:
            weights = np.ones(self.num_items, dtype=np.float64)
        self._sampling_weights = weights
        self._sampling_probs = self._sampling_weights / self._sampling_weights.sum()

    def _sample_from(self, items: List[int]) -> int:
        index = self._rng.randrange(len(items))
        return items[index]

    def _sample_negatives(self, excluded: Set[int], k: int) -> list[int]:
        if k <= 0 or len(excluded) >= self.num_items:
            return []

        excluded_set = excluded
        samples: list[int] = []

        # Fast path: rejection sampling from global distribution.
        max_rounds = max(2, k * 3)
        draw_size = max(8, k * 2)
        for _ in range(max_rounds):
            draw = self._np_rng.choice(
                self._all_items,
                size=draw_size,
                replace=True,
                p=self._sampling_probs,
            )
            for cand in draw.tolist():
                if cand not in excluded_set:
                    samples.append(int(cand))
                    if len(samples) >= k:
                        return samples[:k]

        # Fallback: exact candidate set if rejection struggles.
        excluded_arr = np.fromiter(excluded_set, dtype=np.int64)
        candidates = np.setdiff1d(self._all_items, excluded_arr, assume_unique=False)
        if candidates.size == 0:
            return samples[:k]

        weights = self._sampling_weights[candidates]
        probs = weights / weights.sum() if weights.sum() > 0 else None
        extra = self._np_rng.choice(candidates, size=k - len(samples), replace=True, p=probs)
        samples.extend(extra.tolist())
        return samples[:k]

    @staticmethod
    def _to_tensor(values: List[int], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(values, dtype=torch.long, device=device)

    def _to_tensor_2d(self, values: List[List[int]], device: torch.device) -> torch.Tensor:
        if not values:
            return torch.empty((0, self.num_negs), dtype=torch.long, device=device)
        return torch.tensor(values, dtype=torch.long, device=device)

    def sample(self, user_ids: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """Sample mini-batch triplets Bp and Bn from batch users."""

        pos_users: List[int] = []
        pos_items: List[int] = []
        pos_negs: List[List[int]] = []
        pos_ratings: List[float] = []

        neg_users: List[int] = []
        neg_items: List[int] = []
        neg_negs: List[List[int]] = []
        neg_ratings: List[float] = []

        for user_id in user_ids.detach().cpu().tolist():
            user_id = int(user_id)

            positive_items = self._pos_items_by_user[user_id]
            if positive_items:
                index = self._rng.randrange(len(positive_items))
                positive_item = positive_items[index]
                positive_rating = self._pos_ratings_by_user[user_id][index]
                excluded = self._seen_sets[user_id] if self.exclude_seen else self._pos_sets[user_id]
                positive_negatives = self._sample_negatives(excluded, self.num_negs)
                if positive_negatives:
                    pos_users.append(user_id)
                    pos_items.append(positive_item)
                    pos_negs.append(positive_negatives)
                    pos_ratings.append(positive_rating)

            negative_items = self._neg_items_by_user[user_id]
            if negative_items:
                index = self._rng.randrange(len(negative_items))
                negative_item = negative_items[index]
                negative_rating = self._neg_ratings_by_user[user_id][index]
                excluded = self._seen_sets[user_id] if self.exclude_seen else self._neg_sets[user_id]
                negative_negatives = self._sample_negatives(excluded, self.num_negs)
                if negative_negatives:
                    neg_users.append(user_id)
                    neg_items.append(negative_item)
                    neg_negs.append(negative_negatives)
                    neg_ratings.append(negative_rating)

        return {
            "pos_users": self._to_tensor(pos_users, device),
            "pos_items": self._to_tensor(pos_items, device),
            "pos_negs": self._to_tensor_2d(pos_negs, device),
            "pos_ratings": torch.tensor(
                pos_ratings, dtype=torch.float32, device=device
            )
            if pos_ratings
            else torch.empty(0, dtype=torch.float32, device=device),
            "neg_users": self._to_tensor(neg_users, device),
            "neg_items": self._to_tensor(neg_items, device),
            "neg_negs": self._to_tensor_2d(neg_negs, device),
            "neg_ratings": torch.tensor(
                neg_ratings, dtype=torch.float32, device=device
            )
            if neg_ratings
            else torch.empty(0, dtype=torch.float32, device=device),
        }
