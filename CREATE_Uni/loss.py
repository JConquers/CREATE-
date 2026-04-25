"""
Loss functions for CREATE-Uni.

Implements multiple objectives:
1. LocalObjective: Standard cross-entropy for next-item prediction
2. GlobalObjective: BPR-style ranking loss for collaborative filtering
3. BarlowTwinsObjective: Redundancy reduction loss
4. Combined losses for different model variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalObjective(nn.Module):
    """
    Local objective: Cross-entropy loss for next-item prediction.
    This is the standard sequential recommendation loss.
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute local objective loss.

        Args:
            logits: Prediction scores (batch_size, num_items) or (num_masked, num_items)
            labels: Target item IDs (batch_size,) or (num_masked,)

        Returns:
            loss: Scalar tensor
        """
        assert logits.shape[0] == labels.shape[0], (
            f"Logits and labels must have same batch size: "
            f"{logits.shape[0]} vs {labels.shape[0]}"
        )

        loss = self.loss_fn(logits, labels)
        return loss


class GlobalObjective(nn.Module):
    """
    Global objective: BPR-style pairwise ranking loss.

    Optimizes the relative order of positive and negative items.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        positive_scores: torch.Tensor,
        negative_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute global objective loss.

        Args:
            positive_scores: Scores for positive items (batch_size,)
            negative_scores: Scores for negative items (batch_size,)

        Returns:
            loss: Scalar tensor
        """
        # BPR loss: -log(sigmoid(pos - neg))
        diff = positive_scores - negative_scores
        loss = -F.logsigmoid(diff + self.margin).mean()
        return loss


class BarlowTwinsObjective(nn.Module):
    """
    Barlow Twins objective: Redundancy reduction for self-supervised learning.

    Encourages invariance between graph and sequence views while
    reducing redundancy between features.

    Expects already-projected embeddings (model handles projection).
    """

    def __init__(
        self,
        lambda_param: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.lambda_param = lambda_param
        self.batch_norm = batch_norm

    @staticmethod
    def off_diagonal(x: torch.Tensor) -> torch.Tensor:
        """Return flattened off-diagonal elements of square matrix."""
        n, m = x.shape
        assert n == m, "Matrix must be square"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(
        self,
        fst_embeddings: torch.Tensor,
        snd_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Barlow Twins loss.

        Args:
            fst_embeddings: First view embeddings (batch_size, dim) - already projected
            snd_embeddings: Second view embeddings (batch_size, dim) - already projected

        Returns:
            loss: Scalar tensor
        """
        # Embeddings are already projected by model's graph_projector/seq_projector
        z_i = fst_embeddings
        z_j = snd_embeddings

        # L2 normalize
        z_i = F.normalize(z_i, p=2, dim=-1)
        z_j = F.normalize(z_j, p=2, dim=-1)

        batch_size = z_i.shape[0]

        # Batch normalize (standardize) - use larger eps for stability
        if self.batch_norm:
            z_i_mean = z_i.mean(0)
            z_i_std = z_i.std(0)
            z_j_mean = z_j.mean(0)
            z_j_std = z_j.std(0)
            z_i_norm = (z_i - z_i_mean) / (z_i_std + 1e-6)
            z_j_norm = (z_j - z_j_mean) / (z_j_std + 1e-6)
        else:
            z_i_norm = z_i
            z_j_norm = z_j

        # Cross-correlation matrix
        C = z_i_norm.T @ z_j_norm / batch_size

        # Loss: diagonal should be 1, off-diagonal should be 0
        feature_dim = C.shape[0]
        on_diag = torch.diagonal(C).add(-1).pow(2).sum() / feature_dim
        off_diag = self.off_diagonal(C).pow(2).sum() / feature_dim

        loss = on_diag + self.lambda_param * off_diag
        return loss


class CREATEUniLoss(nn.Module):
    """
    Combined loss for CREATE-Uni model.

    Combines local, global, and barlow-twins/alignment objectives
    with configurable weights. Supports warmup epochs (CREATE-style)
    where local loss is not applied.
    """

    def __init__(
        self,
        local_coef: float = 1.0,
        global_coef: float = 0.1,
        barlow_twins_coef: float = 0.01,
        # Local objective params
        label_smoothing: float = 0.0,
        # Global objective params
        global_margin: float = 0.0,
        # Barlow Twins params
        barlow_lambda: float = 0.1,
        proj_dim: int = 64,
        embedding_dim: int = 64,
        # Warmup params (CREATE-style)
        warmup_epochs: int = 0,
    ):
        super().__init__()
        self.local_coef = local_coef
        self.global_coef = global_coef
        self.barlow_twins_coef = barlow_twins_coef
        self.warmup_epochs = warmup_epochs

        # Initialize objectives
        self.local_objective = LocalObjective(label_smoothing=label_smoothing)
        self.global_objective = GlobalObjective(margin=global_margin)

        if barlow_twins_coef > 0:
            self.barlow_twins_objective = BarlowTwinsObjective(
                lambda_param=barlow_lambda
            )
        else:
            self.barlow_twins_objective = None

    def forward(
        self,
        batch: dict,
        model_outputs: dict,
        epoch_num: int = 0,
        warmup_epochs: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            batch: Input batch with labels
            model_outputs: Model outputs with predictions and embeddings
            epoch_num: Current epoch
            warmup_epochs: Number of warmup epochs (if None, uses self.warmup_epochs)

        Returns:
            total_loss: Combined loss scalar
        """
        warmup = warmup_epochs if warmup_epochs is not None else self.warmup_epochs
        total_loss = 0.0
        loss_dict = {"local": 0.0, "global": 0.0, "align": 0.0}

        # Local objective: only apply after warmup epochs (CREATE-style)
        if epoch_num >= warmup:
            if "local_prediction" in model_outputs and "labels.ids" in batch:
                local_loss = self.local_objective(
                    model_outputs["local_prediction"],
                    batch["labels.ids"],
                )
                local_weighted = self.local_coef * local_loss
                total_loss += local_weighted
                loss_dict["local"] = local_weighted.item()

        # Global objective (if available)
        if (
            self.global_coef > 0
            and "global_positive" in model_outputs
            and "global_negative" in model_outputs
        ):
            global_loss = self.global_objective(
                model_outputs["global_positive"],
                model_outputs["global_negative"],
            )
            global_weighted = self.global_coef * global_loss
            total_loss += global_weighted
            loss_dict["global"] = global_weighted.item()

        # Barlow Twins objective: only apply after warmup epochs
        if epoch_num >= warmup:
            if (
                self.barlow_twins_coef > 0
                and self.barlow_twins_objective is not None
                and "alignment_fst_embeddings" in model_outputs
                and "alignment_snd_embeddings" in model_outputs
            ):
                barlow_loss = self.barlow_twins_objective(
                    model_outputs["alignment_fst_embeddings"],
                    model_outputs["alignment_snd_embeddings"],
                )
                barlow_weighted = self.barlow_twins_coef * barlow_loss
                total_loss += barlow_weighted
                loss_dict["align"] = barlow_weighted.item()

        return total_loss, loss_dict
