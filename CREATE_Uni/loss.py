"""
Loss functions for CREATE-Uni.

Implements multiple objectives:
1. LocalObjective: Standard cross-entropy for next-item prediction
2. GlobalObjective: BPR-style ranking loss for collaborative filtering
3. FusionObjective: Fusion task loss
4. ContrastiveObjective: InfoNCE contrastive loss
5. BarlowTwinsObjective: Redundancy reduction loss
6. Combined losses for different model variants
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


class FusionObjective(nn.Module):
    """
    Fusion objective: Additional ranking loss for fused representations.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        fusion_positive: torch.Tensor,
        fusion_negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute fusion objective loss.

        Args:
            fusion_positive: Scores for positive items from fusion
            fusion_negative: Scores for negative items from fusion

        Returns:
            loss: Scalar tensor
        """
        diff = fusion_positive - fusion_negative
        loss = -F.logsigmoid(diff + self.margin).mean()
        return loss


class ContrastiveObjective(nn.Module):
    """
    Contrastive objective: InfoNCE loss for contrastive learning.

    Encourages agreement between graph and sequence representations
    of the same user/session.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        normalize_embeddings: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        fst_embeddings: torch.Tensor,
        snd_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive objective loss.

        Args:
            fst_embeddings: First view embeddings (batch_size, dim)
            snd_embeddings: Second view embeddings (batch_size, dim)

        Returns:
            loss: Scalar tensor
        """
        batch_size = fst_embeddings.shape[0]
        assert batch_size == snd_embeddings.shape[0]

        # Concatenate embeddings
        combined = torch.cat([fst_embeddings, snd_embeddings], dim=0)

        if self.normalize_embeddings:
            combined = F.normalize(combined, p=2, dim=-1)

        # Compute similarity matrix
        similarity = combined @ combined.T / self.temperature

        # Positive pairs are diagonal offsets by batch_size
        positives = torch.cat(
            [
                torch.diagonal(similarity, offset=batch_size),
                torch.diagonal(similarity, offset=-batch_size),
            ]
        ).reshape(2 * batch_size, 1)

        # Create mask to exclude positives and self-similarity
        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)
        mask = mask.fill_diagonal_(False)
        for i in range(batch_size):
            mask[i, batch_size + i] = False
            mask[batch_size + i, i] = False

        # Get negative samples
        negatives = similarity[mask].reshape(2 * batch_size, -1)

        # Combine positives and negatives
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=logits.device)

        loss = self.loss_fn(logits, labels) / 2
        return loss


class BarlowTwinsObjective(nn.Module):
    """
    Barlow Twins objective: Redundancy reduction for self-supervised learning.

    Encourages invariance between graph and sequence views while
    reducing redundancy between features.
    """

    def __init__(
        self,
        projector: nn.Module,
        lambda_param: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.projector = projector
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
            fst_embeddings: First view embeddings (batch_size, dim)
            snd_embeddings: Second view embeddings (batch_size, dim)

        Returns:
            loss: Scalar tensor
        """
        # Project embeddings
        z_i = self.projector(fst_embeddings)
        z_j = self.projector(snd_embeddings)

        # L2 normalize
        z_i = F.normalize(z_i, p=2, dim=-1)
        z_j = F.normalize(z_j, p=2, dim=-1)

        batch_size = z_i.shape[0]

        # Batch normalize (standardize)
        if self.batch_norm:
            z_i_norm = (z_i - z_i.mean(0)) / (z_i.std(0) + 1e-8)
            z_j_norm = (z_j - z_j.mean(0)) / (z_j.std(0) + 1e-8)
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

    Combines local, global, fusion, and contrastive objectives
    with configurable weights. Supports warmup epochs (CREATE-style)
    where local loss is not applied.
    """

    def __init__(
        self,
        local_coef: float = 1.0,
        global_coef: float = 0.1,
        fusion_coef: float = 0.1,
        contrastive_coef: float = 0.01,
        barlow_twins_coef: float = 0.0,
        # Local objective params
        label_smoothing: float = 0.0,
        # Global objective params
        global_margin: float = 0.0,
        # Contrastive objective params
        contrastive_temperature: float = 1.0,
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
        self.fusion_coef = fusion_coef
        self.contrastive_coef = contrastive_coef
        self.barlow_twins_coef = barlow_twins_coef
        self.warmup_epochs = warmup_epochs

        # Initialize objectives
        self.local_objective = LocalObjective(label_smoothing=label_smoothing)
        self.global_objective = GlobalObjective(margin=global_margin)
        self.fusion_objective = FusionObjective(margin=global_margin)
        self.contrastive_objective = ContrastiveObjective(
            temperature=contrastive_temperature
        )

        if barlow_twins_coef > 0:
            projector = nn.Sequential(
                nn.Linear(embedding_dim, proj_dim),
                nn.BatchNorm1d(proj_dim, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim),
            )
            self.barlow_twins_objective = BarlowTwinsObjective(
                projector=projector, lambda_param=barlow_lambda
            )
        else:
            self.barlow_twins_objective = None

    def forward(
        self,
        batch: dict,
        model_outputs: dict,
        epoch_num: int = 0,
        warmup_epochs: Optional[int] = None,
    ) -> torch.Tensor:
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

        # Local objective: only apply after warmup epochs (CREATE-style)
        if epoch_num >= warmup:
            if "local_prediction" in model_outputs and "labels.ids" in batch:
                local_loss = self.local_objective(
                    model_outputs["local_prediction"],
                    batch["labels.ids"],
                )
                total_loss += self.local_coef * local_loss

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
            total_loss += self.global_coef * global_loss

        # Fusion objective (if available)
        if (
            self.fusion_coef > 0
            and "fusion_positive" in model_outputs
            and "fusion_negative" in model_outputs
        ):
            fusion_loss = self.fusion_objective(
                model_outputs["fusion_positive"],
                model_outputs["fusion_negative"],
            )
            total_loss += self.fusion_coef * fusion_loss

        # Contrastive objective (if embeddings provided)
        if (
            self.contrastive_coef > 0
            and "contrastive_fst_embeddings" in model_outputs
            and "contrastive_snd_embeddings" in model_outputs
        ):
            contrastive_loss = self.contrastive_objective(
                model_outputs["contrastive_fst_embeddings"],
                model_outputs["contrastive_snd_embeddings"],
            )
            total_loss += self.contrastive_coef * contrastive_loss

        # Barlow Twins objective (if configured)
        if (
            self.barlow_twins_coef > 0
            and self.barlow_twins_objective is not None
            and "contrastive_fst_embeddings" in model_outputs
            and "contrastive_snd_embeddings" in model_outputs
        ):
            barlow_loss = self.barlow_twins_objective(
                model_outputs["contrastive_fst_embeddings"],
                model_outputs["contrastive_snd_embeddings"],
            )
            total_loss += self.barlow_twins_coef * barlow_loss

        return total_loss


class MRGSRecLoss(nn.Module):
    """
    Loss function from MRGSRec/UnderDog models.
    Included for compatibility with CREATE baseline.
    """

    def __init__(
        self,
        local_coef: float = 1.0,
        global_coef: float = 0.1,
        fusion_coef: float = 0.1,
        contrastive_coef: float = 0.01,
        contrastive_tau: float = 1.0,
    ):
        super().__init__()
        self.local_coef = local_coef
        self.global_coef = global_coef
        self.fusion_coef = fusion_coef
        self.contrastive_coef = contrastive_coef

        self.local_objective = LocalObjective()
        self.global_objective = GlobalObjective()
        self.fusion_objective = FusionObjective()
        self.contrastive_objective = ContrastiveObjective(temperature=contrastive_tau)

    def forward(
        self,
        batch: dict,
        model_outputs: dict,
    ) -> torch.Tensor:
        loss = (
            self.local_coef * self.local_objective(
                model_outputs["local_prediction"],
                batch["labels.ids"],
            )
            + self.global_coef * self.global_objective(
                model_outputs["global_positive"],
                model_outputs["global_negative"],
            )
            + self.fusion_coef * self.fusion_objective(
                model_outputs["fusion_positive"],
                model_outputs["fusion_negative"],
            )
            + self.contrastive_coef * self.contrastive_objective(
                model_outputs["contrastive_fst_embeddings"],
                model_outputs["contrastive_snd_embeddings"],
            )
        )
        return loss
