"""Training pipeline for CREATE-Pone."""

import argparse
import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Helps reduce CUDA memory fragmentation on long-running jobs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from create_pone.dataset import (
    NextItemCollator,
    SignedTripleSampler,
    UserSequenceDataset,
    build_signed_graph,
    load_dataset_bundle,
)
from create_pone.losses import CreatePoneLoss
from create_pone.models import CreatePoneModel


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_kaggle_runtime() -> bool:
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or os.environ.get("KAGGLE_URL_BASE") is not None
        or Path("/kaggle").exists()
    )


def resolve_device(device_name: str, allow_kaggle_cpu: bool) -> torch.device:
    cuda_available = torch.cuda.is_available()

    if device_name == "auto":
        if cuda_available:
            return torch.device("cuda")

        if is_kaggle_runtime() and not allow_kaggle_cpu:
            raise RuntimeError(
                "Kaggle CUDA is not available. Enable GPU accelerator in notebook settings "
                "and restart the kernel, or pass --allow-kaggle-cpu to continue on CPU."
            )

        return torch.device("cpu")

    if device_name.startswith("cuda") and not cuda_available:
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is False. "
            "Enable Kaggle GPU accelerator and restart the kernel."
        )

    return torch.device(device_name)


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
    return moved


def resolve_effective_batch_size(
    requested_batch_size: int,
    max_seq_len: int,
    num_items: int,
    device: torch.device,
    max_logit_elements: int,
    use_dense_sequence_logits: bool,
) -> int:
    if device.type != "cuda" or max_logit_elements <= 0 or not use_dense_sequence_logits:
        return requested_batch_size

    per_batch_elements = requested_batch_size * max_seq_len * max(1, num_items)
    if per_batch_elements <= max_logit_elements:
        return requested_batch_size

    safe_batch = max(1, max_logit_elements // (max_seq_len * max(1, num_items)))
    return min(requested_batch_size, safe_batch)


def resolve_amp_config(
    use_mixed_precision: bool,
    amp_dtype_name: str,
    device: torch.device,
) -> tuple[bool, torch.dtype]:
    if not use_mixed_precision or device.type != "cuda":
        return False, torch.float32

    if amp_dtype_name == "bf16":
        supports_bf16 = bool(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
        )
        if supports_bf16:
            return True, torch.bfloat16
        print("Requested bf16 AMP but device does not support bf16. Falling back to fp16 AMP.")

    return True, torch.float16


def build_grad_scaler(use_amp: bool, device: torch.device):
    if device.type != "cuda":
        return None

    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def amp_autocast_context(
    use_amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
):
    if not use_amp or device.type != "cuda":
        return nullcontext()

    try:
        return torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype)


def backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    grad_clip: float,
    scaler,
) -> None:
    optimizer.zero_grad(set_to_none=True)

    if scaler is not None:
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        return

    loss.backward()
    if grad_clip > 0:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()


def build_empty_triplets(device: torch.device) -> dict[str, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.long, device=device)
    return {
        "pos_users": empty,
        "pos_items": empty,
        "pos_negs": empty,
        "neg_users": empty,
        "neg_items": empty,
        "neg_negs": empty,
    }


def sample_global_user_ids(num_users: int, sample_size: int) -> torch.Tensor:
    if sample_size <= 0 or sample_size >= num_users:
        return torch.arange(num_users, dtype=torch.long)
    return torch.randperm(num_users)[:sample_size]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def free_disk_gb(path: Path) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except OSError:
        return -1.0


def cuda_free_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return -1.0
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device=device)
        return free_bytes / (1024 ** 3)
    except RuntimeError:
        return -1.0


def prune_epoch_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0 or not checkpoint_dir.exists():
        return

    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))
    excess = len(checkpoint_files) - keep_last
    if excess <= 0:
        return

    for checkpoint_path in checkpoint_files[:excess]:
        try:
            checkpoint_path.unlink(missing_ok=True)
            print(f"Removed old checkpoint to free space: {checkpoint_path}")
        except OSError as exc:
            print(f"Warning: failed to remove old checkpoint {checkpoint_path}: {exc}")


def safe_torch_save(payload: dict, path: Path, label: str) -> bool:
    try:
        torch.save(payload, path)
        return True
    except (RuntimeError, OSError) as exc:
        print(f"Warning: failed to save {label} at {path}: {exc}")
        return False


def build_eval_examples(
    bundle,
    context_sequences: dict[int, list[int]],
    split: str,
    max_users: int,
    min_positive_rating: float,
) -> list[tuple[int, list[int], list[int]]]:
    eval_df = bundle.val_df if split == "val" else bundle.test_df
    if eval_df is None or eval_df.empty:
        return []

    positive_df = eval_df
    if "rating" in eval_df.columns:
        positive_df = eval_df[eval_df["rating"] >= min_positive_rating]
    if positive_df.empty:
        return []

    ordered_df = positive_df
    if "timestamp" in positive_df.columns:
        ordered_df = positive_df.sort_values(["user_id", "timestamp"])

    target_df = ordered_df.groupby("user_id", as_index=False).agg({"item_id": list})

    examples: list[tuple[int, list[int], list[int]]] = []
    for row in target_df.itertuples(index=False):
        user_id = int(getattr(row, "user_id"))
        target_items = [int(item_id) for item_id in dict.fromkeys(getattr(row, "item_id"))]
        if not target_items:
            continue

        context = context_sequences.get(user_id)
        if not context:
            continue

        examples.append((user_id, context, target_items))

    if max_users > 0 and len(examples) > max_users:
        examples = examples[:max_users]

    return examples


def build_seen_items_lookup(
    train_df,
    user_ids: set[int],
) -> dict[int, list[int]]:
    if not user_ids:
        return {}

    subset = train_df.loc[
        train_df["user_id"].isin(user_ids),
        ["user_id", "item_id"],
    ].drop_duplicates()

    lookup: dict[int, list[int]] = {}
    for row in subset.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)
        lookup.setdefault(user_id, []).append(item_id)

    return lookup


def build_user_sequences_from_df(
    interactions_df: pd.DataFrame,
    max_sequence_length: int,
) -> dict[int, list[int]]:
    if interactions_df is None or interactions_df.empty:
        return {}

    ordered_df = interactions_df
    if "timestamp" in interactions_df.columns:
        ordered_df = interactions_df.sort_values(["user_id", "timestamp"])

    sequences: dict[int, list[int]] = {}
    for user_id, group in ordered_df.groupby("user_id", sort=False):
        item_ids = [int(item_id) for item_id in group["item_id"].tolist()]
        if item_ids:
            sequences[int(user_id)] = item_ids[-max_sequence_length:]

    return sequences


@torch.no_grad()
def evaluate_ranking(
    model: CreatePoneModel,
    signed_graph,
    eval_examples: list[tuple[int, list[int], list[int]]],
    pad_id: int,
    device: torch.device,
    topk: int,
    batch_size: int,
    split: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
    filter_seen: bool,
    seen_items_lookup: dict[int, list[int]] | None,
) -> dict:
    if not eval_examples:
        return {
            "split": split,
            "users": 0,
            "topk": max(1, topk),
            "hr": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
            "avg_targets": 0.0,
        }

    model.eval()

    graph_outputs = model(batch={}, signed_graph=signed_graph, run_sequence=False)
    interest_item_embeddings = graph_outputs["interest_item_embeddings"]

    eval_topk = min(max(1, topk), interest_item_embeddings.size(0))

    pad_row = torch.zeros(
        1,
        interest_item_embeddings.size(1),
        dtype=interest_item_embeddings.dtype,
        device=interest_item_embeddings.device,
    )
    sequence_item_table = torch.cat([interest_item_embeddings, pad_row], dim=0)

    hits_total = 0.0
    precision_total = 0.0
    recall_total = 0.0
    ndcg_total = 0.0
    target_count_total = 0
    user_count = len(eval_examples)

    for start_idx in range(0, user_count, max(1, batch_size)):
        batch_examples = eval_examples[start_idx:start_idx + max(1, batch_size)]
        batch_len = len(batch_examples)
        max_len = max(len(example[1]) for example in batch_examples)

        input_ids = torch.full(
            (batch_len, max_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros((batch_len, max_len), dtype=torch.bool, device=device)

        for row_idx, (_, context, _) in enumerate(batch_examples):
            context_tensor = torch.tensor(context, dtype=torch.long, device=device)
            seq_len = context_tensor.numel()
            input_ids[row_idx, :seq_len] = context_tensor
            attention_mask[row_idx, :seq_len] = True

        autocast_ctx = amp_autocast_context(
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            device=device,
        )
        with autocast_ctx:
            input_item_embeddings = sequence_item_table[input_ids]
            encoded_sequence = model.sequence_encoder(
                item_embeddings=input_item_embeddings,
                attention_mask=attention_mask,
            )
            last_hidden = model.sequence_encoder.get_last_hidden(
                encoded=encoded_sequence,
                attention_mask=attention_mask,
            )
            scores = last_hidden @ interest_item_embeddings.t()

        if filter_seen:
            # Match common recommendation protocol: remove seen context items from candidates.
            for row_idx, (user_id, context, target_items) in enumerate(batch_examples):
                target_tensor = torch.tensor(target_items, dtype=torch.long, device=device)
                target_scores = scores[row_idx, target_tensor].clone()
                seen_items = seen_items_lookup.get(user_id) if seen_items_lookup is not None else context
                if seen_items:
                    scores[row_idx, seen_items] = float("-inf")
                scores[row_idx, target_tensor] = target_scores

        _, topk_indices = torch.topk(scores, k=eval_topk, dim=1)

        for row_idx, (_, _, target_items) in enumerate(batch_examples):
            target_set = set(target_items)
            target_count = len(target_set)
            recommended_items = topk_indices[row_idx].tolist()

            hit_count = 0
            dcg = 0.0
            for rank, item_id in enumerate(recommended_items):
                if item_id in target_set:
                    hit_count += 1
                    dcg += 1.0 / math.log2(rank + 2.0)

            ideal_count = min(target_count, eval_topk)
            idcg = sum(1.0 / math.log2(rank + 2.0) for rank in range(ideal_count))

            hits_total += 1.0 if hit_count > 0 else 0.0
            precision_total += hit_count / max(1, eval_topk)
            recall_total += hit_count / max(1, target_count)
            ndcg_total += (dcg / idcg) if idcg > 0 else 0.0
            target_count_total += target_count

    model.train()

    return {
        "split": split,
        "users": user_count,
        "topk": eval_topk,
        "hr": hits_total / max(1, user_count),
        "precision": precision_total / max(1, user_count),
        "recall": recall_total / max(1, user_count),
        "ndcg": ndcg_total / max(1, user_count),
        "avg_targets": target_count_total / max(1, user_count),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CREATE-Pone (CREATE++ signed variant)")

    parser.add_argument("--dataset", choices=["beauty", "books"], required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/create_pone")

    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-dropout", type=float, default=0.1)

    parser.add_argument("--transformer-heads", type=int, default=2)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--transformer-dropout", type=float, default=0.2)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--w-global", type=float, default=0.6)
    parser.add_argument("--w-align", type=float, default=0.1)
    parser.add_argument("--barlow-lambda", type=float, default=0.1)
    parser.add_argument("--orthogonal-mu", type=float, default=0.1)
    parser.add_argument("--contrastive-tau", type=float, default=1.0)
    parser.add_argument("--neg-branch-scale", type=float, default=1.0)

    parser.add_argument("--pos-threshold", type=float, default=4.0)
    parser.add_argument("--neg-threshold", type=float, default=3.0)
    parser.add_argument(
        "--allow-kaggle-cpu",
        action="store_true",
        help="Allow CPU fallback on Kaggle when CUDA is unavailable.",
    )
    parser.add_argument(
        "--max-logit-elements",
        type=int,
        default=120_000_000,
        help="CUDA safety cap for batch*seq_len*num_items when dense logits are enabled.",
    )
    parser.add_argument(
        "--local-loss-chunk-size",
        type=int,
        default=4096,
        help="Chunk size across items for local full-softmax CE (<=0 uses dense logits).",
    )
    parser.add_argument(
        "--global-user-sample",
        type=int,
        default=50_000,
        help="Users sampled per epoch for the global signed loss. Use <=0 for all users.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print batch progress every N mini-batches during joint epochs.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=0,
        help="Limit joint-epoch mini-batches (0 means full epoch).",
    )
    parser.add_argument(
        "--warmup-global-steps",
        type=int,
        default=200,
        help="Number of global signed-graph optimizer steps per warmup epoch.",
    )
    parser.add_argument(
        "--global-steps-per-epoch",
        type=int,
        default=1,
        help="Number of global signed-graph optimizer steps per non-warmup epoch.",
    )
    parser.add_argument(
        "--joint-refresh-every",
        type=int,
        default=500,
        help="Run a full graph+sequence joint step every N local batches (<=0 disables).",
    )
    parser.add_argument(
        "--min-free-gb-for-joint-refresh",
        type=float,
        default=4.0,
        help="Skip full joint refresh when free CUDA memory falls below this threshold.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="test",
        help="Dataset split used for checkpoint-time ranking evaluation.",
    )
    parser.add_argument(
        "--eval-topk",
        type=int,
        default=10,
        help="Top-K cutoff used for Precision/Recall/NDCG evaluation at checkpoints.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Batch size used for checkpoint-time ranking evaluation.",
    )
    parser.add_argument(
        "--eval-max-users",
        type=int,
        default=0,
        help="Maximum users evaluated per checkpoint (<=0 means all).",
    )
    parser.add_argument(
        "--eval-min-rating",
        type=float,
        default=3.0,
        help="Minimum rating treated as relevant for evaluation targets.",
    )
    parser.add_argument(
        "--eval-include-val-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When evaluating on test split, include validation interactions in user context.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Run standalone evaluation every N epochs (checkpoint-triggered eval still runs).",
    )
    parser.add_argument(
        "--eval-filter-seen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask seen context items during ranking evaluation.",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA AMP for sequence-heavy training steps.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="AMP dtype when --mixed-precision is enabled.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save an epoch checkpoint every N epochs. Use <=0 to disable periodic saves.",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=3,
        help="Keep only the most recent periodic checkpoints on disk. <=0 keeps all.",
    )
    parser.add_argument(
        "--save-optimizer-periodic",
        action="store_true",
        help="Include optimizer state in periodic epoch checkpoints (larger files).",
    )
    parser.add_argument(
        "--save-optimizer-best",
        action="store_true",
        help="Include optimizer state in best checkpoint (larger file).",
    )
    parser.add_argument(
        "--save-optimizer-final",
        action="store_true",
        help="Include optimizer state in final checkpoint (larger file).",
    )

    return parser


def train_create_pone(args: argparse.Namespace) -> tuple[Path, Path]:
    set_random_seed(args.seed)
    device = resolve_device(args.device, allow_kaggle_cpu=args.allow_kaggle_cpu)
    use_amp, amp_dtype = resolve_amp_config(
        use_mixed_precision=args.mixed_precision,
        amp_dtype_name=args.amp_dtype,
        device=device,
    )
    scaler = build_grad_scaler(use_amp=use_amp, device=device)

    print(f"Using device: {device}")
    if device.type == "cuda":
        cuda_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_index)
        print(
            f"CUDA device {cuda_index}: {props.name} | "
            f"VRAM={props.total_memory / (1024 ** 3):.1f} GB"
        )
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    if use_amp:
        print(f"Using mixed precision AMP with dtype={args.amp_dtype}")

    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_sequence_length=args.max_seq_len,
    )

    amp_element_multiplier = 2 if use_amp else 1
    effective_logit_elements = int(args.max_logit_elements * amp_element_multiplier)
    compute_dense_sequence_logits = args.local_loss_chunk_size <= 0

    effective_batch_size = resolve_effective_batch_size(
        requested_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_items=bundle.num_items,
        device=device,
        max_logit_elements=effective_logit_elements,
        use_dense_sequence_logits=compute_dense_sequence_logits,
    )
    if effective_batch_size < args.batch_size:
        print(
            f"Reducing batch size from {args.batch_size} to {effective_batch_size} "
            "for CUDA memory safety on full-softmax logits."
        )
    elif not compute_dense_sequence_logits:
        print("Using chunked local loss; skipping dense-logit batch-size cap.")

    train_dataset = UserSequenceDataset(bundle.user_sequences)
    if len(train_dataset) == 0:
        raise RuntimeError("No valid user sequences with at least two interactions were found.")

    collator = NextItemCollator(
        max_sequence_length=args.max_seq_len,
        pad_id=bundle.num_items,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=(device.type == "cuda"),
    )

    signed_graph = build_signed_graph(
        train_df=bundle.train_df,
        num_users=bundle.num_users,
        num_items=bundle.num_items,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        device=device,
    )

    triplet_sampler = SignedTripleSampler(
        train_df=bundle.train_df,
        num_users=bundle.num_users,
        num_items=bundle.num_items,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
        seed=args.seed,
    )

    model = CreatePoneModel(
        num_users=bundle.num_users,
        num_items=bundle.num_items,
        embedding_dim=args.embedding_dim,
        gnn_layers=args.gnn_layers,
        gnn_dropout=args.gnn_dropout,
        max_sequence_length=args.max_seq_len,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
    ).to(device)

    criterion = CreatePoneLoss(
        w_global=args.w_global,
        w_align=args.w_align,
        barlow_lambda=args.barlow_lambda,
        orthogonal_mu=args.orthogonal_mu,
        contrastive_tau=args.contrastive_tau,
        neg_branch_scale=args.neg_branch_scale,
        local_loss_chunk_size=args.local_loss_chunk_size,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_prefix = f"create_pone_{args.dataset}_{stamp}"

    checkpoint_dir = output_dir / f"{run_prefix}_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = output_dir / f"{run_prefix}_best.pt"
    best_total = float("inf")

    eval_context_df = bundle.train_df
    if (
        args.eval_split == "test"
        and args.eval_include_val_context
        and bundle.val_df is not None
        and not bundle.val_df.empty
    ):
        eval_context_df = pd.concat([bundle.train_df, bundle.val_df], ignore_index=True)

    eval_context_sequences = build_user_sequences_from_df(
        interactions_df=eval_context_df,
        max_sequence_length=args.max_seq_len,
    )

    eval_examples = build_eval_examples(
        bundle=bundle,
        context_sequences=eval_context_sequences,
        split=args.eval_split,
        max_users=args.eval_max_users,
        min_positive_rating=args.eval_min_rating,
    )
    eval_seen_items = None
    if args.eval_filter_seen and eval_examples:
        eval_user_ids = {user_id for user_id, _, _ in eval_examples}
        eval_seen_items = build_seen_items_lookup(
            train_df=eval_context_df,
            user_ids=eval_user_ids,
        )
    last_eval_metrics = None
    free_gb_at_start = free_disk_gb(output_dir)

    print(
        "Loaded dataset:",
        f"users={bundle.num_users}",
        f"items={bundle.num_items}",
        f"train_interactions={len(bundle.train_df)}",
        f"train_sequences={len(train_dataset)}",
        f"batch_size={effective_batch_size}",
        f"steps_per_epoch={len(train_loader)}",
        f"warmup_global_steps={max(1, args.warmup_global_steps)}",
        f"global_steps={max(1, args.global_steps_per_epoch)}",
        f"joint_refresh_every={args.joint_refresh_every}",
        f"min_free_gb_for_refresh={args.min_free_gb_for_joint_refresh}",
        f"eval_split={args.eval_split}",
        f"eval_users={len(eval_examples)}",
        f"eval_topk={args.eval_topk}",
        f"eval_min_rating={args.eval_min_rating}",
        f"eval_include_val_context={args.eval_include_val_context}",
        f"eval_filter_seen={args.eval_filter_seen}",
        f"eval_filter_seen_users={0 if eval_seen_items is None else len(eval_seen_items)}",
        f"mixed_precision={use_amp}",
        f"local_loss_chunk_size={args.local_loss_chunk_size}",
        f"effective_logit_cap={effective_logit_elements}",
        f"keep_last_checkpoints={args.keep_last_checkpoints}",
        f"save_opt_periodic={args.save_optimizer_periodic}",
        f"save_opt_best={args.save_optimizer_best}",
        f"save_opt_final={args.save_optimizer_final}",
        f"free_disk_gb={free_gb_at_start:.2f}",
    )
    if 0 < free_gb_at_start < 5.0:
        print(
            "Warning: low free disk space detected. "
            "Consider removing old outputs or lowering checkpoint retention."
        )
    if args.warmup_epochs > 0 and args.eval_every > 0:
        print(
            "Note: warmup epochs optimize global loss only; "
            "ranking metrics are often low until joint training starts."
        )

    empty_triplets = build_empty_triplets(device)

    def build_checkpoint_payload(
        epoch_index: int,
        eval_data,
        include_optimizer: bool,
        best_total_value: float | None = None,
        best_path: Path | None = None,
    ) -> dict:
        payload = {
            "epoch": epoch_index,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "num_users": bundle.num_users,
            "num_items": bundle.num_items,
            "pad_id": bundle.num_items,
            "eval_metrics": eval_data,
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if best_total_value is not None:
            payload["best_total"] = best_total_value
        if best_path is not None:
            payload["best_checkpoint_path"] = str(best_path)
        return payload

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        warmup = epoch < args.warmup_epochs
        model.train()

        global_steps = max(1, args.warmup_global_steps if warmup else args.global_steps_per_epoch)
        global_value_sum = 0.0
        global_df_sum = 0.0
        global_cl_sum = 0.0

        for _ in range(global_steps):
            sampled_user_ids = sample_global_user_ids(
                num_users=bundle.num_users,
                sample_size=args.global_user_sample,
            )
            global_triplets = triplet_sampler.sample(sampled_user_ids, device=device)

            global_outputs = model(batch={}, signed_graph=signed_graph, run_sequence=False)
            global_losses = criterion(
                outputs=global_outputs,
                batch={},
                triplets=global_triplets,
                warmup=True,
            )

            global_step_loss = (
                global_losses["global"] if warmup else args.w_global * global_losses["global"]
            )

            backward_and_step(
                loss=global_step_loss,
                optimizer=optimizer,
                model=model,
                grad_clip=args.grad_clip,
                scaler=scaler,
            )

            global_value_sum += float(global_losses["global"].detach().item())
            global_df_sum += float(global_losses["global_df"].detach().item())
            global_cl_sum += float(global_losses["global_cl"].detach().item())

        global_value = global_value_sum / global_steps
        global_df_value = global_df_sum / global_steps
        global_cl_value = global_cl_sum / global_steps

        local_total_sum = 0.0
        local_loss_sum = 0.0
        align_loss_sum = 0.0
        step_count = 0

        joint_refresh_steps = 0
        skipped_refresh_low_mem = 0
        oom_recovery_steps = 0
        oom_skipped_batches = 0

        if not warmup:
            model.eval()
            with torch.no_grad():
                cached_outputs = model(batch={}, signed_graph=signed_graph, run_sequence=False)
            model.train()

            interest_user_embeddings = cached_outputs["interest_user_embeddings"]
            disinterest_user_embeddings = cached_outputs["disinterest_user_embeddings"]
            interest_item_embeddings = cached_outputs["interest_item_embeddings"]
            disinterest_item_embeddings = cached_outputs["disinterest_item_embeddings"]

            pad_row = torch.zeros(
                1,
                interest_item_embeddings.size(1),
                dtype=interest_item_embeddings.dtype,
                device=interest_item_embeddings.device,
            )
            sequence_item_table = torch.cat([interest_item_embeddings, pad_row], dim=0)

            steps_target = len(train_loader)
            if args.max_steps_per_epoch > 0:
                steps_target = min(steps_target, args.max_steps_per_epoch)

            local_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader, start=1):
                if args.max_steps_per_epoch > 0 and batch_idx > args.max_steps_per_epoch:
                    break

                batch = move_batch_to_device(batch, device)

                run_full_joint = (
                    args.joint_refresh_every > 0
                    and (batch_idx % args.joint_refresh_every == 0)
                )

                if (
                    run_full_joint
                    and device.type == "cuda"
                    and args.min_free_gb_for_joint_refresh > 0
                ):
                    free_gb_before_refresh = cuda_free_gb(device)
                    if 0 < free_gb_before_refresh < args.min_free_gb_for_joint_refresh:
                        run_full_joint = False
                        skipped_refresh_low_mem += 1

                if run_full_joint:
                    # Keep signed sparse propagation in fp32; sparse CUDA matmul does not support fp16.
                    autocast_ctx = nullcontext()
                    with autocast_ctx:
                        outputs = model(
                            batch=batch,
                            signed_graph=signed_graph,
                            run_sequence=True,
                            compute_sequence_logits=compute_dense_sequence_logits,
                        )
                        losses = criterion(
                            outputs=outputs,
                            batch=batch,
                            triplets=empty_triplets,
                            warmup=False,
                        )
                else:
                    autocast_ctx = amp_autocast_context(
                        use_amp=use_amp,
                        amp_dtype=amp_dtype,
                        device=device,
                    )
                    with autocast_ctx:
                        input_item_embeddings = sequence_item_table[batch["input_ids"]]
                        encoded_sequence = model.sequence_encoder(
                            item_embeddings=input_item_embeddings,
                            attention_mask=batch["attention_mask"],
                        )

                        last_hidden = model.sequence_encoder.get_last_hidden(
                            encoded=encoded_sequence,
                            attention_mask=batch["attention_mask"],
                        )

                        outputs = {
                            "interest_user_embeddings": interest_user_embeddings,
                            "disinterest_user_embeddings": disinterest_user_embeddings,
                            "interest_item_embeddings": interest_item_embeddings,
                            "disinterest_item_embeddings": disinterest_item_embeddings,
                            "sequence_hidden": encoded_sequence,
                            "sequence_user_embedding": last_hidden,
                        }
                        if compute_dense_sequence_logits:
                            outputs["sequence_logits"] = encoded_sequence @ interest_item_embeddings.t()

                        losses = criterion(
                            outputs=outputs,
                            batch=batch,
                            triplets=empty_triplets,
                            warmup=False,
                        )

                used_full_joint_step = run_full_joint

                try:
                    backward_and_step(
                        loss=losses["total"],
                        optimizer=optimizer,
                        model=model,
                        grad_clip=args.grad_clip,
                        scaler=scaler,
                    )
                except torch.OutOfMemoryError as exc:
                    if device.type == "cuda":
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()

                    if run_full_joint:
                        oom_recovery_steps += 1
                        used_full_joint_step = False
                        print(
                            f"Warning: OOM on full-joint refresh at batch {batch_idx}; "
                            "retrying with sequence-only step."
                        )

                        autocast_ctx = amp_autocast_context(
                            use_amp=use_amp,
                            amp_dtype=amp_dtype,
                            device=device,
                        )
                        with autocast_ctx:
                            input_item_embeddings = sequence_item_table[batch["input_ids"]]
                            encoded_sequence = model.sequence_encoder(
                                item_embeddings=input_item_embeddings,
                                attention_mask=batch["attention_mask"],
                            )

                            last_hidden = model.sequence_encoder.get_last_hidden(
                                encoded=encoded_sequence,
                                attention_mask=batch["attention_mask"],
                            )

                            outputs = {
                                "interest_user_embeddings": interest_user_embeddings,
                                "disinterest_user_embeddings": disinterest_user_embeddings,
                                "interest_item_embeddings": interest_item_embeddings,
                                "disinterest_item_embeddings": disinterest_item_embeddings,
                                "sequence_hidden": encoded_sequence,
                                "sequence_user_embedding": last_hidden,
                            }
                            if compute_dense_sequence_logits:
                                outputs["sequence_logits"] = encoded_sequence @ interest_item_embeddings.t()

                            losses = criterion(
                                outputs=outputs,
                                batch=batch,
                                triplets=empty_triplets,
                                warmup=False,
                            )

                        try:
                            backward_and_step(
                                loss=losses["total"],
                                optimizer=optimizer,
                                model=model,
                                grad_clip=args.grad_clip,
                                scaler=scaler,
                            )
                        except torch.OutOfMemoryError:
                            if device.type == "cuda":
                                optimizer.zero_grad(set_to_none=True)
                                torch.cuda.empty_cache()
                            oom_skipped_batches += 1
                            print(
                                f"Warning: skipped batch {batch_idx} after OOM retry. "
                                "Consider larger --joint-refresh-every or lower batch."
                            )
                            continue
                    else:
                        oom_skipped_batches += 1
                        print(
                            f"Warning: skipped batch {batch_idx} due to CUDA OOM: {exc}."
                        )
                        continue

                if used_full_joint_step:
                    joint_refresh_steps += 1
                    model.eval()
                    with torch.no_grad():
                        cached_outputs = model(batch={}, signed_graph=signed_graph, run_sequence=False)
                    model.train()

                    interest_user_embeddings = cached_outputs["interest_user_embeddings"]
                    disinterest_user_embeddings = cached_outputs["disinterest_user_embeddings"]
                    interest_item_embeddings = cached_outputs["interest_item_embeddings"]
                    disinterest_item_embeddings = cached_outputs["disinterest_item_embeddings"]

                    pad_row = torch.zeros(
                        1,
                        interest_item_embeddings.size(1),
                        dtype=interest_item_embeddings.dtype,
                        device=interest_item_embeddings.device,
                    )
                    sequence_item_table = torch.cat([interest_item_embeddings, pad_row], dim=0)

                local_total_sum += float(losses["total"].detach().item())
                local_loss_sum += float(losses["local"].detach().item())
                align_loss_sum += float(losses["align"].detach().item())
                step_count += 1

                should_log = (
                    args.log_every > 0
                    and (
                        step_count == 1
                        or step_count % args.log_every == 0
                        or step_count == steps_target
                    )
                )
                if should_log:
                    elapsed = time.time() - local_start_time
                    steps_per_second = step_count / max(elapsed, 1e-9)
                    eta_seconds = (steps_target - step_count) / max(steps_per_second, 1e-9)
                    print(
                        f"  batch {step_count}/{steps_target} | "
                        f"total={losses['total'].detach().item():.4f} | "
                        f"local={losses['local'].detach().item():.4f} | "
                        f"align={losses['align'].detach().item():.4f} | "
                        f"refresh_steps={joint_refresh_steps} | "
                        f"refresh_skipped={skipped_refresh_low_mem} | "
                        f"oom_recovered={oom_recovery_steps} | "
                        f"oom_skipped={oom_skipped_batches} | "
                        f"eta={format_duration(eta_seconds)}"
                    )

        if warmup:
            epoch_metrics = {
                "epoch": epoch + 1,
                "warmup": warmup,
                "total": global_value,
                "local": 0.0,
                "global": global_value,
                "global_df": global_df_value,
                "global_cl": global_cl_value,
                "align": 0.0,
                "local_steps": 0,
                "joint_refresh_steps": 0,
                "refresh_skipped": 0,
                "oom_recovered": 0,
                "oom_skipped": 0,
            }
        else:
            step_count = max(step_count, 1)
            local_total_avg = local_total_sum / step_count
            local_avg = local_loss_sum / step_count
            align_avg = align_loss_sum / step_count
            epoch_metrics = {
                "epoch": epoch + 1,
                "warmup": warmup,
                "total": local_total_avg + args.w_global * global_value,
                "local": local_avg,
                "global": global_value,
                "global_df": global_df_value,
                "global_cl": global_cl_value,
                "align": align_avg,
                "local_steps": step_count,
                "joint_refresh_steps": joint_refresh_steps,
                "refresh_skipped": skipped_refresh_low_mem,
                "oom_recovered": oom_recovery_steps,
                "oom_skipped": oom_skipped_batches,
            }

        should_save_best = epoch_metrics["total"] < best_total
        should_save_periodic = (
            args.checkpoint_every > 0
            and ((epoch + 1) % args.checkpoint_every == 0)
        )
        should_eval_standalone = (
            args.eval_every > 0
            and ((epoch + 1) % args.eval_every == 0)
        )
        needs_eval_for_checkpoint = should_save_best or should_save_periodic

        eval_metrics = None
        if eval_examples and (should_eval_standalone or needs_eval_for_checkpoint):
            eval_start_time = time.time()
            eval_metrics = evaluate_ranking(
                model=model,
                signed_graph=signed_graph,
                eval_examples=eval_examples,
                pad_id=bundle.num_items,
                device=device,
                topk=args.eval_topk,
                batch_size=args.eval_batch_size,
                split=args.eval_split,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                filter_seen=args.eval_filter_seen,
                seen_items_lookup=eval_seen_items,
            )
            eval_elapsed = time.time() - eval_start_time
            print(
                f"Eval {eval_metrics['split']}@{eval_metrics['topk']} | "
                f"users={eval_metrics['users']} | "
                f"P={eval_metrics['precision']:.4f} | "
                f"R={eval_metrics['recall']:.4f} | "
                f"NDCG={eval_metrics['ndcg']:.4f} | "
                f"HR={eval_metrics['hr']:.4f} | "
                f"targets/user={eval_metrics['avg_targets']:.2f} | "
                f"time={format_duration(eval_elapsed)}"
            )
            last_eval_metrics = eval_metrics

        checkpoint_eval_metrics = eval_metrics
        if checkpoint_eval_metrics is None and needs_eval_for_checkpoint:
            checkpoint_eval_metrics = last_eval_metrics

        epoch_metrics["eval"] = eval_metrics

        history.append(epoch_metrics)

        epoch_elapsed = time.time() - epoch_start_time

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"warmup={warmup} | "
            f"total={epoch_metrics['total']:.4f} | "
            f"local={epoch_metrics['local']:.4f} | "
            f"global={epoch_metrics['global']:.4f} | "
            f"align={epoch_metrics['align']:.4f} | "
            f"local_steps={epoch_metrics['local_steps']} | "
            f"refresh_steps={epoch_metrics['joint_refresh_steps']} | "
            f"refresh_skipped={epoch_metrics['refresh_skipped']} | "
            f"oom_recovered={epoch_metrics['oom_recovered']} | "
            f"oom_skipped={epoch_metrics['oom_skipped']} | "
            f"time={format_duration(epoch_elapsed)}"
        )

        if should_save_best:
            candidate_best = epoch_metrics["total"]
            best_payload = build_checkpoint_payload(
                epoch_index=epoch + 1,
                eval_data=checkpoint_eval_metrics,
                include_optimizer=args.save_optimizer_best,
                best_total_value=candidate_best,
            )
            if safe_torch_save(best_payload, best_checkpoint_path, "best checkpoint"):
                best_total = candidate_best
                print(f"Saved best checkpoint to: {best_checkpoint_path}")

        if should_save_periodic:
            if args.keep_last_checkpoints > 0:
                prune_epoch_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    keep_last=max(0, args.keep_last_checkpoints - 1),
                )

            epoch_checkpoint_path = checkpoint_dir / f"{run_prefix}_epoch_{epoch + 1:03d}.pt"
            periodic_payload = build_checkpoint_payload(
                epoch_index=epoch + 1,
                eval_data=checkpoint_eval_metrics,
                include_optimizer=args.save_optimizer_periodic,
            )
            if safe_torch_save(periodic_payload, epoch_checkpoint_path, "periodic checkpoint"):
                print(f"Saved epoch checkpoint to: {epoch_checkpoint_path}")
                if args.keep_last_checkpoints > 0:
                    prune_epoch_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        keep_last=args.keep_last_checkpoints,
                    )

    checkpoint_path = output_dir / f"{run_prefix}.pt"
    history_path = output_dir / f"{run_prefix}_history.json"

    final_payload = build_checkpoint_payload(
        epoch_index=args.epochs,
        eval_data=last_eval_metrics,
        include_optimizer=args.save_optimizer_final,
        best_total_value=best_total,
        best_path=best_checkpoint_path,
    )
    final_saved = safe_torch_save(final_payload, checkpoint_path, "final checkpoint")
    if final_saved:
        print(f"Saved checkpoint to: {checkpoint_path}")
    else:
        print("Warning: final checkpoint was not saved due to storage write failure.")

    with history_path.open("w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)

    print(f"Saved history to: {history_path}")

    return checkpoint_path, history_path
