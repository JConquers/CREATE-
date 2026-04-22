"""
Evaluation metrics for sequential recommendation.

Implements standard metrics:
- Hit Rate (HR@K): Fraction of users for whom the correct item appears in top-K
- NDCG@K: Normalized Discounted Cumulative Gain
- Precision@K, Recall@K, MAP@K
"""

import torch
from typing import Dict, List, Optional


def hit_rate_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Compute Hit Rate@K.

    Args:
        predictions: Item scores (batch_size, num_items)
        targets: Target item IDs (batch_size,)
        k: Cut-off for top-K evaluation
        pad_id: Padding item ID to exclude

    Returns:
        hr: Hit rate per sample (batch_size,)
    """
    _, topk_indices = torch.topk(predictions, k, dim=-1)
    hits = (topk_indices == targets.unsqueeze(-1)).any(dim=-1).float()
    return hits


def ndcg_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
    pad_id: int = 0,
) -> torch.Tensor:
    """
    Compute NDCG@K.

    Args:
        predictions: Item scores (batch_size, num_items)
        targets: Target item IDs (batch_size,)
        k: Cut-off for top-K evaluation
        pad_id: Padding item ID to exclude

    Returns:
        ndcg: NDCG per sample (batch_size,)
    """
    _, topk_indices = torch.topk(predictions, k, dim=-1)

    # DCG: relevance at position i divided by log2(i+1)
    # For binary relevance (hit=1, miss=0), DCG = sum of 1/log2(i+1) for hits
    hits = (topk_indices == targets.unsqueeze(-1)).float()
    discounts = torch.log2(torch.arange(2, k + 2, device=predictions.device).float())
    dcg = (hits / discounts).sum(dim=-1)

    # Ideal DCG is 1 (if hit at position 1)
    idcg = torch.ones_like(dcg)

    ndcg = dcg / idcg
    return ndcg


def precision_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute Precision@K.

    For single-target recommendation, Precision@K = HR@K / K.
    """
    hr = hit_rate_at_k(predictions, targets, k)
    return hr / k


def recall_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute Recall@K.

    For single-target recommendation, Recall@K = HR@K.
    """
    return hit_rate_at_k(predictions, targets, k)


def map_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute Mean Average Precision@K.

    For single-target recommendation, AP@K = HR@K / rank_of_hit.
    """
    _, topk_indices = torch.topk(predictions, k, dim=-1)

    # Find rank of first hit (0-indexed, so add 1 for 1-indexed rank)
    hits = (topk_indices == targets.unsqueeze(-1))
    first_hit_rank = hits.float().argmax(dim=-1) + 1

    # If no hit, set rank to k (will give 0 AP after division)
    no_hit = ~hits.any(dim=-1)
    first_hit_rank[no_hit] = k

    # AP = 1 / rank for single target
    ap = 1.0 / first_hit_rank.float()
    ap[no_hit] = 0.0

    return ap


class Metric:
    """Base class for metrics."""

    def __init__(self, k: int, name: str = "metric"):
        self.k = k
        self.name = name
        self.reset()

    def reset(self):
        self.values = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError

    def compute(self) -> float:
        raise NotImplementedError

    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """Compute metric for a batch."""
        self.update(predictions, targets)
        return self.compute_batch(predictions, targets)

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """Compute metric values for individual samples in batch."""
        raise NotImplementedError


class HitRateMetric(Metric):
    """Hit Rate@K metric."""

    def __init__(self, k: int = 10):
        super().__init__(k, name=f"hr@{k}")

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        return hit_rate_at_k(predictions, targets, self.k).cpu().tolist()

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class NDCGMetric(Metric):
    """NDCG@K metric."""

    def __init__(self, k: int = 10):
        super().__init__(k, name=f"ndcg@{k}")

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        return ndcg_at_k(predictions, targets, self.k).cpu().tolist()

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class PrecisionMetric(Metric):
    """Precision@K metric."""

    def __init__(self, k: int = 10):
        super().__init__(k, name=f"precision@{k}")

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        return precision_at_k(predictions, targets, self.k).cpu().tolist()

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class RecallMetric(Metric):
    """Recall@K metric."""

    def __init__(self, k: int = 10):
        super().__init__(k, name=f"recall@{k}")

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        return recall_at_k(predictions, targets, self.k).cpu().tolist()

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class MAPMetric(Metric):
    """Mean Average Precision@K metric."""

    def __init__(self, k: int = 10):
        super().__init__(k, name=f"map@{k}")

    def compute_batch(self, predictions: torch.Tensor, targets: torch.Tensor) -> List[float]:
        return map_at_k(predictions, targets, self.k).cpu().tolist()

    def compute(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


def create_metrics(k_values: List[int] = [5, 10, 20]) -> Dict[str, Metric]:
    """
    Create a dictionary of metrics for evaluation.

    Args:
        k_values: List of K values for evaluation

    Returns:
        metrics: Dictionary of metric instances
    """
    metrics = {}
    for k in k_values:
        metrics[f"hr@{k}"] = HitRateMetric(k=k)
        metrics[f"ndcg@{k}"] = NDCGMetric(k=k)
        metrics[f"precision@{k}"] = PrecisionMetric(k=k)
        metrics[f"recall@{k}"] = RecallMetric(k=k)
        metrics[f"map@{k}"] = MAPMetric(k=k)
    return metrics


def evaluate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate model predictions against targets.

    Args:
        predictions: Item scores (batch_size, num_items)
        targets: Target item IDs (batch_size,)
        k_values: List of K values for evaluation

    Returns:
        results: Dictionary of metric values
    """
    results = {}
    metrics = create_metrics(k_values)

    for metric_name, metric in metrics.items():
        values = metric(predictions, targets)
        results[metric_name] = sum(values) / len(values) if values else 0.0

    return results
