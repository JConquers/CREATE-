"""Pone-GNN baseline implementation (standalone, no external repo dependency)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGINConv2(MessagePassing):
    """Lightweight signed GIN-style convolution used in Pone-GNN."""

    def __init__(self, first_aggr: bool):
        super().__init__(aggr="add")
        self.first_aggr = first_aggr
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def get_norm(node: torch.Tensor, edge_index: torch.Tensor):
            row, col = edge_index
            deg = degree(col, node.size(0), dtype=node.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm, deg_inv_sqrt

        def gin_norm(out: torch.Tensor, input_x: torch.Tensor, deg_inv_sqrt: torch.Tensor):
            norm_self = deg_inv_sqrt[torch.arange(input_x.size(0), device=input_x.device)]
            norm_self = norm_self.unsqueeze(dim=1).repeat(1, input_x.size(1))
            return out + (1 + self.eps) * norm_self * input_x

        norm_pos, deg_inv_sqrt_pos = get_norm(x[0], pos_edge_index)
        norm_neg, deg_inv_sqrt_neg = get_norm(x[0], neg_edge_index)

        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=x[0], norm=norm_pos)
            out_neg = self.propagate(neg_edge_index, x=x[0], norm=norm_neg)
            out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
            out_neg = gin_norm(out_neg, x[0], deg_inv_sqrt_neg)
            return out_pos, out_neg

        out_pos = self.propagate(pos_edge_index, x=x[0], norm=norm_pos)
        out_neg = self.propagate(pos_edge_index, x=x[1], norm=norm_pos)
        out_pos = gin_norm(out_pos, x[0], deg_inv_sqrt_pos)
        out_neg = gin_norm(out_neg, x[1], deg_inv_sqrt_pos)
        return out_pos, out_neg

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        return norm.view(-1, 1) * x_j


class PoneGNN(nn.Module):
    """Signed GNN baseline mirroring the Pone-GNN repo logic."""

    def __init__(self, num_users: int, num_items: int, num_layers: int = 4, dim: int = 64, reg: float = 5e-5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.dim = dim
        self.reg = reg
        self.temperature = 1.0
        self.contrastive_weight = 1.0

        self.user_embedding = nn.Parameter(torch.empty(num_users, dim))
        self.item_embedding = nn.Parameter(torch.empty(num_items, dim))
        self.user_neg_embedding = nn.Parameter(torch.empty(num_users, dim))
        self.item_neg_embedding = nn.Parameter(torch.empty(num_items, dim))

        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        nn.init.xavier_normal_(self.user_neg_embedding)
        nn.init.xavier_normal_(self.item_neg_embedding)

        self.conv = nn.ModuleList()
        for i in range(num_layers):
            self.conv.append(LightGINConv2(first_aggr=(i == 0)))

    def forward(self, data_p: Data, data_n: Data) -> tuple[torch.Tensor, torch.Tensor]:
        pos_edges = data_p.edge_index
        neg_edges = data_n.edge_index
        alpha = 1.0 / (self.num_layers + 1)

        ego_pos = torch.cat((self.user_embedding, self.item_embedding), dim=0)
        ego_neg = torch.cat((self.user_neg_embedding, self.item_neg_embedding), dim=0)

        pos_embeddings = ego_pos * alpha
        neg_embeddings = ego_neg * alpha
        ego_embeddings = (ego_pos, ego_pos)

        for layer in self.conv:
            ego_embeddings = layer(ego_embeddings, pos_edges, neg_edges)
            pos_embeddings = pos_embeddings + ego_embeddings[0] * alpha
            neg_embeddings = neg_embeddings + ego_embeddings[1] * alpha

        return pos_embeddings, neg_embeddings

    def loss(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        weights: torch.Tensor,
        negative_samples: torch.Tensor,
        data_p: Data,
        data_n: Data,
        trigger_negative: bool,
    ) -> torch.Tensor:
        pos_emb, neg_emb = self(data_p, data_n)

        u_p = pos_emb[users]
        i_p = pos_emb[items]
        n_p = pos_emb[negative_samples]

        positive_batch = torch.mul(u_p, i_p)
        negative_batch = torch.mul(u_p.view(len(u_p), 1, self.dim), n_p)
        pos_coeff = (-0.5 * torch.sign(weights) + 1.5).view(len(u_p), 1)

        pos_bpr = F.logsigmoid(
            pos_coeff * positive_batch.sum(dim=1).view(len(u_p), 1)
            - negative_batch.sum(dim=2)
        ).sum(dim=1)
        loss = -pos_bpr.mean()

        reg_loss = 0.5 * (u_p ** 2).sum() + 0.5 * (i_p ** 2).sum() + 0.5 * (n_p ** 2).sum()
        loss = loss + self.reg * reg_loss

        if trigger_negative:
            u_n = neg_emb[users]
            i_n = neg_emb[items]
            n_n = neg_emb[negative_samples]

            positive_batch = torch.mul(u_n, i_n)
            negative_batch = torch.mul(u_n.view(len(u_n), 1, self.dim), n_n)
            neg_coeff = (0.5 * torch.sign(weights) + 1.5).view(len(u_n), 1)

            neg_bpr = F.logsigmoid(
                negative_batch.sum(dim=2)
                - neg_coeff * positive_batch.sum(dim=1).view(len(u_n), 1)
            ).sum(dim=1)
            loss = loss - neg_bpr.mean()

            reg_loss = 0.5 * (u_n ** 2).sum() + 0.5 * (i_n ** 2).sum() + 0.5 * (n_n ** 2).sum()
            loss = loss + self.reg * reg_loss

            u_p_norm = F.normalize(u_p, dim=1)
            i_p_norm = F.normalize(i_p, dim=1)
            u_n_norm = F.normalize(u_n, dim=1)
            i_n_norm = F.normalize(i_n, dim=1)

            pos_sim = torch.sum(u_p_norm * i_p_norm, dim=1)
            neg_sim = torch.sum(u_n_norm * i_n_norm, dim=1)

            pos_pair = torch.exp(pos_sim / self.temperature)
            neg_pair = torch.exp(neg_sim / self.temperature)
            contrastive = -torch.log(pos_pair / (pos_pair + neg_pair)).mean()
            loss = loss + self.contrastive_weight * contrastive

        return loss

    @torch.no_grad()
    def get_ui_embeddings(self, data_p: Data, data_n: Data) -> tuple[torch.Tensor, ...]:
        pos_embeddings, neg_embeddings = self(data_p, data_n)
        u_p, i_p = torch.split(pos_embeddings, [self.num_users, self.num_items], dim=0)
        u_n, i_n = torch.split(neg_embeddings, [self.num_users, self.num_items], dim=0)
        return u_p, u_n, i_p, i_n


class PoneBipartiteDataset(Dataset):
    def __init__(
        self,
        train_df,
        neg_dist: np.ndarray,
        offset: float,
        num_users: int,
        num_items: int,
        num_negs: int,
    ):
        self.edge_1 = torch.tensor(train_df["user_id"].values, dtype=torch.long)
        self.edge_2 = torch.tensor(train_df["item_id"].values, dtype=torch.long) + num_users
        self.edge_3 = torch.tensor(train_df["rating"].values, dtype=torch.float32) - offset

        self.neg_dist = neg_dist
        self.num_negs = num_negs
        self.num_users = num_users
        self.num_items = num_items
        self._all_items = np.arange(num_items)
        self.train = train_df

    def negs_gen_epoch(self) -> None:
        self.edge_4 = torch.empty((len(self.edge_1), self.num_negs), dtype=torch.long)
        for user_id in np.unique(self.train["user_id"].values):
            pos = self.train[self.train["user_id"] == user_id]["item_id"].values
            neg = np.setdiff1d(self._all_items, pos)
            if neg.size == 0:
                continue
            weights = self.neg_dist[neg]
            weights = weights / weights.sum() if weights.sum() > 0 else None
            total = len(pos) * self.num_negs
            sampled = np.random.choice(neg, total, replace=True, p=weights)
            sampled = torch.tensor(sampled, dtype=torch.long) + self.num_users
            self.edge_4[self.edge_1 == user_id] = sampled.view(len(pos), self.num_negs)

    def __len__(self) -> int:
        return len(self.edge_1)

    def __getitem__(self, idx: int):
        u = self.edge_1[idx]
        v = self.edge_2[idx]
        w = self.edge_3[idx]
        negs = self.edge_4[idx]
        return u, v, w, negs


def deg_dist(train_df, num_items: int) -> np.ndarray:
    counts = np.bincount(train_df["item_id"].values, minlength=num_items).astype(np.float64)
    weights = np.power(counts, 0.75)
    if weights.sum() <= 0:
        weights = np.ones(num_items, dtype=np.float64)
    return weights


def build_pone_edges(
    train_df,
    num_users: int,
    pos_offset: float,
    neg_offset: float,
) -> tuple[Data, Data]:
    pos_train = train_df[train_df["rating"] > pos_offset]
    neg_train = train_df[train_df["rating"] < neg_offset]

    edge_user = torch.tensor(pos_train["user_id"].values, dtype=torch.long)
    edge_item = torch.tensor(pos_train["item_id"].values, dtype=torch.long) + num_users
    edge_p = torch.stack(
        [torch.cat([edge_user, edge_item]), torch.cat([edge_item, edge_user])],
        dim=0,
    )

    edge_user_n = torch.tensor(neg_train["user_id"].values, dtype=torch.long)
    edge_item_n = torch.tensor(neg_train["item_id"].values, dtype=torch.long) + num_users
    edge_n = torch.stack(
        [torch.cat([edge_user_n, edge_item_n]), torch.cat([edge_item_n, edge_user_n])],
        dim=0,
    )

    return Data(edge_index=edge_p), Data(edge_index=edge_n)


def gen_top_k_new3(
    train_df,
    test_df,
    r_hat_p: torch.Tensor,
    r_hat_n: torch.Tensor,
    num_items: int,
    k: int,
    threshold: float = 0.0,
) -> np.ndarray:
    all_items = set(range(num_items))
    tot_items = set(train_df["item_id"]).union(set(test_df["item_id"]))
    no_items = list(all_items - tot_items)
    if no_items:
        r_hat_p[:, no_items] = -float("inf")

    for u, i in train_df[["user_id", "item_id"]].values:
        r_hat_p[int(u), int(i)] = -float("inf")

    if threshold is not None:
        mask = r_hat_n > threshold
        r_hat_p[mask] = -float("inf")

    _, reco = torch.topk(r_hat_p, k)
    return reco.cpu().numpy()


def build_eval_masks(
    train_df,
    test_df,
    num_users: int,
    num_items: int,
) -> tuple[list[int], list[list[int]]]:
    all_items = set(range(num_items))
    tot_items = set(train_df["item_id"]).union(set(test_df["item_id"]))
    no_items = list(all_items - tot_items)

    seen_by_user: list[list[int]] = [[] for _ in range(num_users)]
    for row in train_df.itertuples(index=False):
        seen_by_user[int(row.user_id)].append(int(row.item_id))

    return no_items, seen_by_user


def generate_reco_chunked(
    emb_u: torch.Tensor,
    emb_n_u: torch.Tensor,
    emb_v: torch.Tensor,
    emb_n_v: torch.Tensor,
    train_df,
    test_df,
    num_users: int,
    num_items: int,
    topk: int,
    threshold: float,
    chunk_size: int,
) -> np.ndarray:
    no_items, seen_by_user = build_eval_masks(train_df, test_df, num_users, num_items)
    reco = np.zeros((num_users, topk), dtype=np.int64)

    for start in range(0, num_users, chunk_size):
        end = min(start + chunk_size, num_users)
        scores = emb_u[start:end] @ emb_v.t()

        if threshold is not None:
            neg_scores = emb_n_u[start:end] @ emb_n_v.t()
            scores = scores.masked_fill(neg_scores > threshold, float("-inf"))

        if no_items:
            scores[:, no_items] = float("-inf")

        for idx, user_id in enumerate(range(start, end)):
            seen_items = seen_by_user[user_id]
            if seen_items:
                scores[idx, seen_items] = float("-inf")

        topk_indices = torch.topk(scores, topk, dim=1).indices
        reco[start:end] = topk_indices.cpu().numpy()

    return reco


def compute_pone_metrics(
    reco: np.ndarray,
    train_df,
    test_df,
    num_users: int,
    n_list: list[int] | None = None,
    threshold: float = 3.0,
    partition: tuple[int, int] = (20, 50),
) -> Dict[str, Dict[str, list[float]]]:
    if n_list is None:
        n_list = [1, 5, 10, 15, 20]

    max_k = int(min(reco.shape[1], max(n_list, default=0)))
    n_list = [n for n in n_list if n <= max_k]
    if not n_list:
        empty = [0.0] * 0
        return {
            "n_list": [],
            "precision": {"total": empty},
            "recall": {"total": empty},
            "ndcg": {"total": empty},
            "hr": {"total": empty},
        }

    positives = test_df[test_df["rating"] >= threshold]
    gt: Dict[int, set[int]] = {}
    for row in positives.itertuples(index=False):
        gt.setdefault(int(row.user_id), set()).add(int(row.item_id))

    if not gt:
        zeros = [0.0] * len(n_list)
        return {
            "n_list": n_list,
            "precision": {"total": zeros},
            "recall": {"total": zeros},
            "ndcg": {"total": zeros},
            "hr": {"total": zeros},
        }

    user_counts = np.bincount(train_df["user_id"].values, minlength=num_users)
    group1_mask = user_counts < partition[0]
    group2_mask = (user_counts < partition[1]) & (~group1_mask)
    group3_mask = ~(user_counts < partition[1])

    group_map = np.zeros(num_users, dtype=np.int64)
    group_map[group2_mask] = 1
    group_map[group3_mask] = 2
    group_names = ["group1", "group2", "group3", "total"]

    metrics = {
        "precision": {name: np.zeros(len(n_list)) for name in group_names},
        "recall": {name: np.zeros(len(n_list)) for name in group_names},
        "ndcg": {name: np.zeros(len(n_list)) for name in group_names},
        "hr": {name: np.zeros(len(n_list)) for name in group_names},
    }
    counts = {name: 0 for name in group_names}

    discount = 1.0 / np.log2(np.arange(2, max_k + 2))
    idcg_base = np.cumsum(discount)
    n_indices = [n - 1 for n in n_list]

    for user_id, targets in gt.items():
        if user_id >= num_users:
            continue
        group_idx = int(group_map[user_id])
        group_name = group_names[group_idx]

        rec = reco[user_id][:max_k]
        hits = np.array([1.0 if int(item) in targets else 0.0 for item in rec], dtype=np.float64)
        hits_cum = np.cumsum(hits)
        hit_binary = (hits_cum > 0).astype(np.float64)
        dcg_cum = np.cumsum(hits * discount)

        ideal_len = min(len(targets), max_k)
        if ideal_len > 0:
            idcg = idcg_base.copy()
            idcg[ideal_len:] = idcg[ideal_len - 1]
        else:
            idcg = np.zeros(max_k, dtype=np.float64)

        for idx, k_idx in enumerate(n_indices):
            k = n_list[idx]
            precision = hits_cum[k_idx] / max(1, k)
            recall = hits_cum[k_idx] / max(1, len(targets))
            hr = hit_binary[k_idx]
            ndcg = (dcg_cum[k_idx] / idcg[k_idx]) if idcg[k_idx] > 0 else 0.0

            metrics["precision"][group_name][idx] += precision
            metrics["recall"][group_name][idx] += recall
            metrics["hr"][group_name][idx] += hr
            metrics["ndcg"][group_name][idx] += ndcg

            metrics["precision"]["total"][idx] += precision
            metrics["recall"]["total"][idx] += recall
            metrics["hr"]["total"][idx] += hr
            metrics["ndcg"]["total"][idx] += ndcg

        counts[group_name] += 1
        counts["total"] += 1

    for name in group_names:
        denom = max(1, counts[name])
        for metric in metrics.values():
            metric[name] = (metric[name] / denom).tolist()

    metrics["n_list"] = n_list
    return metrics


def compute_metrics(
    reco: np.ndarray,
    test_df,
    topk: int,
    min_rating: float,
) -> Dict[str, float]:
    positives = test_df[test_df["rating"] >= min_rating]
    gt: Dict[int, set[int]] = {}
    for row in positives.itertuples(index=False):
        gt.setdefault(int(row.user_id), set()).add(int(row.item_id))

    if not gt:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "hr": 0.0}

    precision_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0
    hr_total = 0.0

    for user_id, targets in gt.items():
        rec = reco[user_id][:topk]
        hits = 0
        dcg = 0.0
        for rank, item_id in enumerate(rec):
            if int(item_id) in targets:
                hits += 1
                dcg += 1.0 / math.log2(rank + 2.0)

        precision_total += hits / max(1, topk)
        recall_total += hits / max(1, len(targets))
        hr_total += 1.0 if hits > 0 else 0.0

        ideal_count = min(len(targets), topk)
        idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_count))
        if idcg > 0:
            ndcg_total += dcg / idcg

    user_count = len(gt)
    return {
        "precision": precision_total / user_count,
        "recall": recall_total / user_count,
        "ndcg": ndcg_total / user_count,
        "hr": hr_total / user_count,
    }


def train_pone_gnn_baseline(
    args,
    bundle,
    output_dir,
    logger,
    device: torch.device,
) -> tuple[Path, Path]:
    train_df = bundle.train_df
    test_df = bundle.test_df

    neg_dist = deg_dist(train_df, bundle.num_items)
    dataset = PoneBipartiteDataset(
        train_df,
        neg_dist,
        args.pone_offset,
        bundle.num_users,
        bundle.num_items,
        args.neg_sample_k,
    )

    data_p, data_n = build_pone_edges(train_df, bundle.num_users, args.pone_offset, 3.5)
    data_p = data_p.to(device)
    data_n = data_n.to(device)

    model = PoneGNN(
        bundle.num_users,
        bundle.num_items,
        num_layers=args.pone_num_layers,
        dim=args.embedding_dim,
        reg=args.pone_reg,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pone_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 200], gamma=0.2)

    history = []
    best_ndcg = -float("inf")

    eval_every = max(1, args.pone_eval_every)
    pone_eval_n = [1, 5, 10, 15, 20]
    for epoch in range(1, args.epochs + 1):
        dataset.negs_gen_epoch()
        ds = DataLoader(dataset, batch_size=args.pone_batch_size, shuffle=True)

        model.train()
        total_loss = 0.0
        for u, v, w, negs in ds:
            u = u.to(device)
            v = v.to(device)
            w = w.to(device)
            negs = negs.to(device)

            trigger = True
            if args.pone_neg_every > 0:
                trigger = (epoch % args.pone_neg_every) == args.pone_neg_offset

            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(u, v, w, negs, data_p, data_n, trigger)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())

        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "loss": total_loss / max(1, len(ds)),
        }

        if eval_every == 1 or epoch % eval_every == 1:
            model.eval()
            emb_u, emb_n_u, emb_v, emb_n_v = model.get_ui_embeddings(data_p, data_n)
            chunk_size = max(1, args.eval_batch_size)
            topk = min(max(pone_eval_n), emb_v.size(0))
            reco = generate_reco_chunked(
                emb_u,
                emb_n_u,
                emb_v,
                emb_n_v,
                train_df,
                test_df,
                bundle.num_users,
                bundle.num_items,
                topk,
                threshold=0.0,
                chunk_size=chunk_size,
            )
            metrics = compute_pone_metrics(
                reco,
                train_df,
                test_df,
                bundle.num_users,
                n_list=pone_eval_n,
                threshold=args.eval_min_rating,
            )

            epoch_metrics.update({
                f"precision@{k}": metrics["precision"]["total"][idx]
                for idx, k in enumerate(metrics["n_list"])
            })
            epoch_metrics.update({
                f"recall@{k}": metrics["recall"]["total"][idx]
                for idx, k in enumerate(metrics["n_list"])
            })
            epoch_metrics.update({
                f"ndcg@{k}": metrics["ndcg"]["total"][idx]
                for idx, k in enumerate(metrics["n_list"])
            })
            epoch_metrics.update({
                f"hr@{k}": metrics["hr"]["total"][idx]
                for idx, k in enumerate(metrics["n_list"])
            })

            for idx, k in enumerate(metrics["n_list"]):
                logger.info(
                    "Pone-GNN Eval@%s | P=%.4f R=%.4f NDCG=%.4f HR=%.4f",
                    k,
                    metrics["precision"]["total"][idx],
                    metrics["recall"]["total"][idx],
                    metrics["ndcg"]["total"][idx],
                    metrics["hr"]["total"][idx],
                )

            if args.eval_topk in metrics["n_list"]:
                idx = metrics["n_list"].index(args.eval_topk)
            else:
                idx = -1
            best_ndcg = max(best_ndcg, metrics["ndcg"]["total"][idx])

        history.append(epoch_metrics)

    checkpoint_path = output_dir / f"pone_gnn_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    history_path = output_dir / f"pone_gnn_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_history.json"

    payload = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "num_users": bundle.num_users,
        "num_items": bundle.num_items,
        "best_ndcg": best_ndcg,
    }
    torch.save(payload, checkpoint_path)
    with history_path.open("w", encoding="utf-8") as file_obj:
        import json

        json.dump(history, file_obj, indent=2)

    return checkpoint_path, history_path
