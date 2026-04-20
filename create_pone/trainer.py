"""Training pipeline for CREATE-Pone."""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
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
) -> int:
    if device.type != "cuda" or max_logit_elements <= 0:
        return requested_batch_size

    per_batch_elements = requested_batch_size * max_seq_len * max(1, num_items)
    if per_batch_elements <= max_logit_elements:
        return requested_batch_size

    safe_batch = max(1, max_logit_elements // (max_seq_len * max(1, num_items)))
    return min(requested_batch_size, safe_batch)


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
        default=350_000_000,
        help="CUDA safety cap for batch*seq_len*num_items; batch is reduced automatically.",
    )

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    return parser


def train_create_pone(args: argparse.Namespace) -> tuple[Path, Path]:
    set_random_seed(args.seed)
    device = resolve_device(args.device, allow_kaggle_cpu=args.allow_kaggle_cpu)

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

    bundle = load_dataset_bundle(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        max_sequence_length=args.max_seq_len,
    )

    effective_batch_size = resolve_effective_batch_size(
        requested_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_items=bundle.num_items,
        device=device,
        max_logit_elements=args.max_logit_elements,
    )
    if effective_batch_size < args.batch_size:
        print(
            f"Reducing batch size from {args.batch_size} to {effective_batch_size} "
            "for CUDA memory safety on full-softmax logits."
        )

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
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []

    print(
        "Loaded dataset:",
        f"users={bundle.num_users}",
        f"items={bundle.num_items}",
        f"train_interactions={len(bundle.train_df)}",
        f"train_sequences={len(train_dataset)}",
        f"batch_size={effective_batch_size}",
    )

    for epoch in range(args.epochs):
        warmup = epoch < args.warmup_epochs
        model.train()

        running = {
            "total": 0.0,
            "local": 0.0,
            "global": 0.0,
            "global_df": 0.0,
            "global_cl": 0.0,
            "align": 0.0,
        }
        step_count = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            triplets = triplet_sampler.sample(batch["user_ids"], device=device)

            outputs = model(
                batch=batch,
                signed_graph=signed_graph,
                run_sequence=not warmup,
            )
            losses = criterion(
                outputs=outputs,
                batch=batch,
                triplets=triplets,
                warmup=warmup,
            )

            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            for key in running:
                running[key] += float(losses[key].detach().item())
            step_count += 1

        step_count = max(step_count, 1)
        epoch_metrics = {
            "epoch": epoch + 1,
            "warmup": warmup,
            **{key: value / step_count for key, value in running.items()},
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"warmup={warmup} | "
            f"total={epoch_metrics['total']:.4f} | "
            f"local={epoch_metrics['local']:.4f} | "
            f"global={epoch_metrics['global']:.4f} | "
            f"align={epoch_metrics['align']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = output_dir / f"create_pone_{args.dataset}_{stamp}.pt"
    history_path = output_dir / f"create_pone_{args.dataset}_{stamp}_history.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "num_users": bundle.num_users,
            "num_items": bundle.num_items,
            "pad_id": bundle.num_items,
        },
        checkpoint_path,
    )

    with history_path.open("w", encoding="utf-8") as file_obj:
        json.dump(history, file_obj, indent=2)

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved history to: {history_path}")

    return checkpoint_path, history_path
