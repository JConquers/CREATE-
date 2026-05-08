"""Loss functions for the CREATE-Pone variant."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            "Logits and labels must have same batch size: "
            f"{logits.shape[0]} vs {labels.shape[0]}"
        )

        loss = self.loss_fn(logits, labels)
        return loss


class CreatePoneLoss:
    """Implements CREATE++ objectives for the signed CREATE-Pone variant."""

    def __init__(
        self,
        w_global: float,
        w_align: float,
        barlow_lambda: float,
        orthogonal_mu: float,
        contrastive_tau: float,
        neg_branch_scale: float,
    ):
        self.w_global = w_global
        self.w_align = w_align
        self.barlow_lambda = barlow_lambda
        self.orthogonal_mu = orthogonal_mu
        self.contrastive_tau = contrastive_tau
        self.neg_branch_scale = neg_branch_scale
        self.local_objective = LocalObjective()

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _local_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        valid_mask = attention_mask & (target_ids >= 0)
        if not valid_mask.any():
            reference = outputs.get("sequence_hidden")
            if reference is None:
                reference = outputs["interest_item_embeddings"]
            return self._zero_like(reference)

        logits = outputs.get("sequence_logits")
        if logits is None:
            sequence_hidden = outputs["sequence_hidden"]
            item_embeddings = outputs["interest_item_embeddings"]
            logits = sequence_hidden @ item_embeddings.t()

        return self.local_objective(logits[valid_mask], target_ids[valid_mask])

    def _dual_feedback_loss(
        self,
        outputs: dict,
        triplets: dict,
        include_negative: bool = True,
    ) -> torch.Tensor:
        interest_user = outputs["interest_user_embeddings"]
        disinterest_user = outputs["disinterest_user_embeddings"]
        interest_item = outputs["interest_item_embeddings"]
        disinterest_item = outputs["disinterest_item_embeddings"]

        loss = self._zero_like(interest_user)

        pos_users = triplets["pos_users"]
        pos_negs = triplets.get("pos_negs")
        if pos_users.numel() > 0 and pos_negs is not None and pos_negs.numel() > 0:
            pos_items = triplets["pos_items"]
            z_u = interest_user[pos_users]
            z_i = interest_item[pos_items]
            if pos_negs.dim() == 1:
                pos_negs = pos_negs.unsqueeze(1)
            z_j = interest_item[pos_negs]

            y_ui = (z_u * z_i).sum(dim=1)
            y_uj = (z_u.unsqueeze(1) * z_j).sum(dim=2)
            diff = y_ui.unsqueeze(1) - y_uj
            loss = loss - F.logsigmoid(diff).mean()

        if include_negative:
            neg_users = triplets["neg_users"]
            neg_negs = triplets.get("neg_negs")
            if neg_users.numel() > 0 and neg_negs is not None and neg_negs.numel() > 0:
                neg_items = triplets["neg_items"]
                v_u = disinterest_user[neg_users]
                v_i = disinterest_item[neg_items]
                if neg_negs.dim() == 1:
                    neg_negs = neg_negs.unsqueeze(1)
                v_j = disinterest_item[neg_negs]

                y_ui = self.neg_branch_scale * (v_u * v_i).sum(dim=1)
                y_uj = (v_u.unsqueeze(1) * v_j).sum(dim=2)
                diff = y_uj - y_ui.unsqueeze(1)
                loss = loss - F.logsigmoid(diff).mean()

        return loss

    def _contrastive_loss(self, outputs: dict, triplets: dict) -> torch.Tensor:
        interest_user = outputs["interest_user_embeddings"]
        disinterest_user = outputs["disinterest_user_embeddings"]
        interest_item = outputs["interest_item_embeddings"]
        disinterest_item = outputs["disinterest_item_embeddings"]

        pos_users = triplets["pos_users"]
        pos_items = triplets["pos_items"]
        neg_users = triplets["neg_users"]
        neg_items = triplets["neg_items"]

        if pos_users.numel() == 0 or neg_users.numel() == 0:
            return self._zero_like(interest_user)

        # Eq. (11): pair all positive/negative samples for each user u.
        pos_index_by_user: dict[int, list[int]] = {}
        for idx, user_id in enumerate(pos_users.detach().cpu().tolist()):
            user_id = int(user_id)
            pos_index_by_user.setdefault(user_id, []).append(idx)

        neg_index_by_user: dict[int, list[int]] = {}
        for idx, user_id in enumerate(neg_users.detach().cpu().tolist()):
            user_id = int(user_id)
            neg_index_by_user.setdefault(user_id, []).append(idx)

        common_users = [
            user_id for user_id in pos_index_by_user if user_id in neg_index_by_user
        ]
        if not common_users:
            return self._zero_like(interest_user)

        pos_pair_indices: list[int] = []
        neg_pair_indices: list[int] = []
        for user_id in common_users:
            for pos_idx in pos_index_by_user[user_id]:
                for neg_idx in neg_index_by_user[user_id]:
                    pos_pair_indices.append(pos_idx)
                    neg_pair_indices.append(neg_idx)

        if not pos_pair_indices:
            return self._zero_like(interest_user)

        pos_indices = torch.tensor(
            pos_pair_indices,
            dtype=torch.long,
            device=interest_user.device,
        )
        neg_indices = torch.tensor(
            neg_pair_indices,
            dtype=torch.long,
            device=interest_user.device,
        )

        z_u = interest_user[pos_users[pos_indices]]
        z_i = interest_item[pos_items[pos_indices]]

        v_u = disinterest_user[neg_users[neg_indices]]
        v_i = disinterest_item[neg_items[neg_indices]]

        pos_scores = (z_u * z_i).sum(dim=1) / self.contrastive_tau
        neg_scores = (v_u * v_i).sum(dim=1) / self.contrastive_tau

        logits = torch.stack([pos_scores, neg_scores], dim=1)
        return -(pos_scores - torch.logsumexp(logits, dim=1)).mean()

    def _alignment_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        sequence_user = outputs["sequence_user_embedding"]

        user_ids = batch["user_ids"]
        interest_user = outputs["interest_user_embeddings"][user_ids]
        disinterest_user = outputs["disinterest_user_embeddings"][user_ids]

        batch_size = sequence_user.size(0)
        if batch_size < 2:
            return self._zero_like(sequence_user)

        seq_std = (sequence_user - sequence_user.mean(dim=0)) / (
            sequence_user.std(dim=0) + 1e-9
        )
        int_std = (interest_user - interest_user.mean(dim=0)) / (
            interest_user.std(dim=0) + 1e-9
        )
        dis_std = (disinterest_user - disinterest_user.mean(dim=0)) / (
            disinterest_user.std(dim=0) + 1e-9
        )

        c_hz = (seq_std.T @ int_std) / batch_size
        c_hv = (seq_std.T @ dis_std) / batch_size

        on_diag = (torch.diagonal(c_hz) - 1.0).pow(2).sum()

        diagonal = torch.diag(torch.diagonal(c_hz))
        off_diag = (c_hz - diagonal).pow(2).sum()

        orthogonal = c_hv.pow(2).sum()

        return on_diag + self.barlow_lambda * off_diag + self.orthogonal_mu * orthogonal

    def __call__(
        self,
        outputs: dict,
        batch: dict,
        triplets: dict,
        warmup: bool,
        include_negative: bool = True,
        include_contrastive: bool = True,
    ) -> dict:
        global_df = self._dual_feedback_loss(
            outputs,
            triplets,
            include_negative=include_negative,
        )
        if include_contrastive:
            global_cl = self._contrastive_loss(outputs, triplets)
        else:
            global_cl = self._zero_like(outputs["interest_user_embeddings"])
        global_loss = global_df + global_cl

        if warmup:
            total_loss = global_loss
            local_loss = self._zero_like(global_loss)
            align_loss = self._zero_like(global_loss)
        else:
            local_loss = self._local_loss(outputs, batch)
            align_loss = self._alignment_loss(outputs, batch)
            total_loss = local_loss + self.w_global * global_loss + self.w_align * align_loss

        return {
            "total": total_loss,
            "local": local_loss,
            "global": global_loss,
            "global_df": global_df,
            "global_cl": global_cl,
            "align": align_loss,
        }
