"""Loss functions for the CREATE-Pone variant."""

import torch
import torch.nn.functional as F


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
        local_loss_chunk_size: int = 0,
        rating_offset: float = 0.0,
    ):
        self.w_global = w_global
        self.w_align = w_align
        self.barlow_lambda = barlow_lambda
        self.orthogonal_mu = orthogonal_mu
        self.contrastive_tau = contrastive_tau
        self.neg_branch_scale = neg_branch_scale
        self.local_loss_chunk_size = local_loss_chunk_size
        self.rating_offset = rating_offset

    @staticmethod
    def _zero_like(reference: torch.Tensor) -> torch.Tensor:
        return reference.sum() * 0.0

    def _rating_weight(self, ratings: torch.Tensor | None, positive: bool) -> torch.Tensor | None:
        if ratings is None or ratings.numel() == 0:
            return None
        sign = torch.sign(ratings - self.rating_offset)
        if positive:
            return (-0.5 * sign + 1.5)
        return (0.5 * sign + 1.5)

    def _local_loss(self, outputs: dict, batch: dict) -> torch.Tensor:
        target_ids = batch["target_ids"]
        attention_mask = batch["attention_mask"]

        valid_mask = attention_mask & (target_ids >= 0)
        if not valid_mask.any():
            reference = outputs.get("sequence_hidden")
            if reference is None:
                reference = outputs["sequence_logits"]
            return self._zero_like(reference)

        if self.local_loss_chunk_size <= 0 and "sequence_logits" in outputs:
            logits = outputs["sequence_logits"]
            return F.cross_entropy(logits[valid_mask], target_ids[valid_mask], reduction="mean")

        return self._local_loss_chunked(outputs=outputs, valid_mask=valid_mask, target_ids=target_ids)

    def _local_loss_chunked(
        self,
        outputs: dict,
        valid_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        sequence_hidden = outputs["sequence_hidden"]
        item_embeddings = outputs["interest_item_embeddings"]

        hidden = sequence_hidden[valid_mask]
        targets = target_ids[valid_mask]

        if hidden.numel() == 0:
            return self._zero_like(sequence_hidden)

        target_vectors = item_embeddings[targets]
        target_logits = (hidden * target_vectors).sum(dim=1)

        chunk_size = max(1, int(self.local_loss_chunk_size))
        num_items = item_embeddings.size(0)

        max_scores = torch.full_like(target_logits, -float("inf"))
        sum_exp = torch.zeros_like(target_logits)

        for start in range(0, num_items, chunk_size):
            end = min(start + chunk_size, num_items)
            item_chunk = item_embeddings[start:end]
            chunk_scores = hidden @ item_chunk.t()

            chunk_max = chunk_scores.max(dim=1).values
            updated_max = torch.maximum(max_scores, chunk_max)

            sum_exp = (
                sum_exp * torch.exp(max_scores - updated_max)
                + torch.exp(chunk_scores - updated_max.unsqueeze(1)).sum(dim=1)
            )
            max_scores = updated_max

        log_denom = max_scores + torch.log(sum_exp + 1e-12)
        return -(target_logits - log_denom).mean()

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
        pos_ratings = triplets.get("pos_ratings")
        if pos_users.numel() > 0 and pos_negs is not None and pos_negs.numel() > 0:
            pos_items = triplets["pos_items"]
            z_u = interest_user[pos_users]
            z_i = interest_item[pos_items]
            if pos_negs.dim() == 1:
                pos_negs = pos_negs.unsqueeze(1)
            z_j = interest_item[pos_negs]

            y_ui = (z_u * z_i).sum(dim=1)
            y_uj = (z_u.unsqueeze(1) * z_j).sum(dim=2)
            weight = self._rating_weight(pos_ratings, positive=True)
            if weight is None:
                weight = torch.ones_like(y_ui)
            diff = weight.unsqueeze(1) * y_ui.unsqueeze(1) - y_uj
            loss = loss - F.logsigmoid(diff).mean()

        if include_negative:
            neg_users = triplets["neg_users"]
            neg_negs = triplets.get("neg_negs")
            neg_ratings = triplets.get("neg_ratings")
            if neg_users.numel() > 0 and neg_negs is not None and neg_negs.numel() > 0:
                neg_items = triplets["neg_items"]
                v_u = disinterest_user[neg_users]
                v_i = disinterest_item[neg_items]
                if neg_negs.dim() == 1:
                    neg_negs = neg_negs.unsqueeze(1)
                v_j = disinterest_item[neg_negs]

                y_ui = self.neg_branch_scale * (v_u * v_i).sum(dim=1)
                y_uj = (v_u.unsqueeze(1) * v_j).sum(dim=2)
                weight = self._rating_weight(neg_ratings, positive=False)
                if weight is None:
                    weight = torch.ones_like(y_ui)
                diff = y_uj - weight.unsqueeze(1) * y_ui.unsqueeze(1)
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

        # Eq. (11): pair positive and negative terms for the same user u.
        pos_index_by_user = {}
        for idx, user_id in enumerate(pos_users.detach().cpu().tolist()):
            user_id = int(user_id)
            if user_id not in pos_index_by_user:
                pos_index_by_user[user_id] = idx

        neg_index_by_user = {}
        for idx, user_id in enumerate(neg_users.detach().cpu().tolist()):
            user_id = int(user_id)
            if user_id not in neg_index_by_user:
                neg_index_by_user[user_id] = idx

        common_users = [
            user_id for user_id in pos_index_by_user if user_id in neg_index_by_user
        ]
        if not common_users:
            return self._zero_like(interest_user)

        pos_indices = torch.tensor(
            [pos_index_by_user[user_id] for user_id in common_users],
            dtype=torch.long,
            device=interest_user.device,
        )
        neg_indices = torch.tensor(
            [neg_index_by_user[user_id] for user_id in common_users],
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
