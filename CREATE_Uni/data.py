"""
Data loaders and collators for CREATE-Uni.

Provides:
- SequenceDataset: Dataset for sequential recommendation
- Collator: Batch collation with padding and masking
- create_dataloaders: Factory function for train/val/test loaders
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SequenceDataset(Dataset):
    """
    Dataset for sequential recommendation.

    Loads user-item interaction sequences from CSV and prepares
    them for training/evaluation.
    """

    def __init__(
        self,
        data_path: str,
        max_sequence_length: int = 50,
        mode: str = "train",
        user_col: str = "user_id",
        item_col: str = "item_id",
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to CSV file with interactions
            max_sequence_length: Maximum sequence length for padding
            mode: One of "train", "validation", "test"
            user_col: Name of user ID column
            item_col: Name of item ID column
        """
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.max_sequence_length = max_sequence_length
        self.user_col = user_col
        self.item_col = item_col

        self._index = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load and process data from CSV."""
        df = pd.read_csv(data_path)

        # Sort by timestamp if available
        if "timestamp" in df.columns:
            df = df.sort_values([self.user_col, "timestamp"])
        else:
            df = df.sort_values([self.user_col])

        # Group by user
        user_items = df.groupby(self.user_col)[self.item_col].apply(list).to_dict()

        for user_idx, item_ids in sorted(user_items.items(), key=lambda x: x[0]):
            # Truncate to max length
            item_sequence = item_ids[-self.max_sequence_length :]

            # For next-item prediction, we need at least 2 items (1 context + 1 target)
            # For val/test: need at least 2 items
            # For train: need at least 2 items (context + target)
            if len(item_sequence) < 2:
                continue

            self._index.append(
                {
                    "user.ids": user_idx,
                    "item.ids": item_sequence,
                }
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict:
        sample = self._index[index].copy()
        return sample


class Collator:
    """
    Collator for sequential recommendation batches.

    Handles:
    - Padding sequences to max length in batch
    - Creating attention masks
    - Preparing labels for training
    - MLM masking for BERT4Rec-style training
    """

    def __init__(
        self,
        pad_id: int = 0,
        mask_id: int = 1,
        num_items: int = 0,
        mode: str = "train",
        mlm_prob: float = 0.15,
        replace_mask_prob: float = 0.8,
        replace_random_prob: float = 0.1,
        ensure_one_mask: bool = True,
        seq_encoder_type: str = "sasrec",
    ):
        """
        Initialize collator.

        Args:
            pad_id: Padding token ID
            mask_id: Mask token ID (for BERT4Rec)
            num_items: Number of items in dataset
            mode: One of "train", "validation", "test"
            mlm_prob: Probability of masking a token (BERT4Rec)
            replace_mask_prob: Probability of replacing with [MASK]
            replace_random_prob: Probability of replacing with random item
            ensure_one_mask: Ensure at least one masked token per sequence
            seq_encoder_type: "sasrec" or "bert4rec"
        """
        assert mode in ["train", "validation", "test"]
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.num_items = num_items
        self.mode = mode
        self.mlm_prob = mlm_prob
        self.replace_mask_prob = replace_mask_prob
        self.replace_random_prob = replace_random_prob
        self.ensure_one_mask = ensure_one_mask
        self.seq_encoder_type = seq_encoder_type

    @staticmethod
    def _shift_item_ids(item_ids: List[int]) -> List[int]:
        """Map raw item ids 0..N-1 to token ids 1..N."""
        return [item_id + 1 for item_id in item_ids]

    def _pad_sequence(
        self, sequences: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad sequences to max length in batch.

        Returns:
            padded: Padded tensor (batch_size, max_len)
            mask: Attention mask (batch_size, max_len), True = real token
        """
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        padded = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_id,
            dtype=torch.long,
        )
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            padded[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            mask[i, :seq_len] = True

        return padded, mask

    def _mask_bert4rec_input(
        self,
        token_input_ids: torch.Tensor,
        raw_item_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply MLM masking for BERT4Rec training.

        Returns:
            masked_input_ids: Input with masked tokens
            mlm_labels: Target labels for masked positions (-100 for non-masked)
            mlm_mask: Boolean mask for MLM positions
        """
        device = token_input_ids.device
        B, L = token_input_ids.shape

        # Candidates for masking: real tokens only
        can_mask = (
            attn_mask
            & (token_input_ids != self.pad_id)
            & (token_input_ids != self.mask_id)
        )

        # Sample masking positions
        probs = torch.full((B, L), self.mlm_prob, device=device)
        probs = probs * can_mask.float()
        mlm_mask = torch.bernoulli(probs).bool()

        # Ensure at least one mask per sequence
        if self.ensure_one_mask:
            no_mask_rows = mlm_mask.sum(dim=1) == 0
            if no_mask_rows.any():
                valid_pos = can_mask[no_mask_rows]
                if valid_pos.any():
                    r = torch.rand_like(valid_pos.float())
                    r[~valid_pos] = -1.0
                    pos = r.argmax(dim=1)
                    mlm_mask[no_mask_rows, pos] = True

        # Labels: true IDs at masked positions
        mlm_labels = raw_item_ids.clone()
        mlm_labels[~mlm_mask] = -100  # Ignore index

        # Create masked input
        masked_input_ids = token_input_ids.clone()
        rand = torch.rand((B, L), device=device)

        # 80% -> [MASK]
        replace_with_mask = mlm_mask & (rand < self.replace_mask_prob)
        masked_input_ids[replace_with_mask] = self.mask_id

        # 10% -> random item
        replace_with_random = (
            mlm_mask
            & (rand >= self.replace_mask_prob)
            & (rand < self.replace_mask_prob + self.replace_random_prob)
        )
        if replace_with_random.any():
            random_ids = torch.randint(
                low=1,  # Real item token ids are 1..num_items
                high=self.num_items + 1,
                size=(replace_with_random.sum().item(),),
                device=device,
                dtype=torch.long,
            )
            masked_input_ids[replace_with_random] = random_ids

        return masked_input_ids, mlm_labels, mlm_mask

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            processed: Dictionary of tensors for model input
        """
        processed = {
            "user.ids": [],
            "item.ids": [],
            "item.raw_ids": [],
            "item.length": [],
            "labels.ids": [],
            "graph_pos_user.ids": [],
            "graph_pos_item.ids": [],
        }

        for sample in batch:
            user_id = sample["user.ids"]
            raw_items = sample["item.ids"]
            processed["user.ids"].append(user_id)

            if self.seq_encoder_type == "bert4rec" and self.mode == "train":
                # BERT4Rec: use full sequence for MLM
                token_items = self._shift_item_ids(raw_items)
                processed["item.ids"].extend(token_items)
                processed["item.raw_ids"].extend(raw_items)
                processed["item.length"].append(len(token_items))
                processed["graph_pos_user.ids"].extend([user_id] * len(raw_items))
                processed["graph_pos_item.ids"].extend(raw_items)
            elif self.mode == "train":
                # SASRec: supervise every next-item position, like CREATE.
                context_raw_items = raw_items[:-1]
                context_token_items = self._shift_item_ids(context_raw_items)
                next_items = raw_items[1:]
                processed["item.ids"].extend(context_token_items)
                processed["item.raw_ids"].extend(context_raw_items)
                processed["item.length"].append(len(context_token_items))
                processed["labels.ids"].extend(next_items)
                processed["graph_pos_user.ids"].extend([user_id] * len(next_items))
                processed["graph_pos_item.ids"].extend(next_items)
            else:
                # Validation/Test: use all but last item
                # Ensure at least 1 item in context (handle edge case of single-item sequences)
                if len(raw_items) > 1:
                    context_raw_items = raw_items[:-1]
                else:
                    # Single-item sequence: use that item as context, no label (will be filtered by mask)
                    context_raw_items = raw_items
                context_token_items = self._shift_item_ids(context_raw_items)
                processed["item.ids"].extend(context_token_items)
                processed["item.raw_ids"].extend(context_raw_items)
                processed["item.length"].append(len(context_token_items))
                # Target is last item
                processed["labels.ids"].append(raw_items[-1])

        # Convert to tensors
        processed["user.ids"] = torch.tensor(processed["user.ids"], dtype=torch.long)
        processed["item.length"] = torch.tensor(processed["item.length"], dtype=torch.long)

        # Pad sequences
        item_sequences = []
        current_idx = 0
        for i, seq_len in enumerate(processed["item.length"].tolist()):
            current_idx += seq_len
            item_sequences.append(processed["item.ids"][current_idx - seq_len : current_idx])

        raw_item_sequences = []
        current_idx = 0
        for seq_len in processed["item.length"].tolist():
            current_idx += seq_len
            raw_item_sequences.append(
                processed["item.raw_ids"][current_idx - seq_len : current_idx]
            )

        padded_items, attention_mask = self._pad_sequence(item_sequences)
        padded_raw_items, _ = self._pad_sequence(raw_item_sequences)
        processed["item.ids"] = padded_items
        processed["item.raw_ids"] = padded_raw_items
        processed["mask"] = attention_mask

        if processed["graph_pos_user.ids"]:
            processed["graph_pos_user.ids"] = torch.tensor(
                processed["graph_pos_user.ids"], dtype=torch.long
            )
            processed["graph_pos_item.ids"] = torch.tensor(
                processed["graph_pos_item.ids"], dtype=torch.long
            )
        else:
            processed["graph_pos_user.ids"] = torch.empty(0, dtype=torch.long)
            processed["graph_pos_item.ids"] = torch.empty(0, dtype=torch.long)

        processed["labels.ids"] = torch.tensor(processed["labels.ids"], dtype=torch.long)

        # Apply MLM masking for BERT4Rec training
        if self.mode == "train" and self.seq_encoder_type == "bert4rec":
            masked_ids, mlm_labels, mlm_mask = self._mask_bert4rec_input(
                padded_items,
                padded_raw_items,
                attention_mask,
            )
            processed["input_ids"] = masked_ids
            processed["mlm_labels"] = mlm_labels
            processed["mlm_mask"] = mlm_mask
            # Flatten labels for masked positions (BERT4Rec MLM style)
            processed["labels.ids"] = mlm_labels[mlm_mask]
        elif self.seq_encoder_type == "bert4rec":
            batch_size, seq_len = padded_items.shape
            extended_input_ids = torch.full(
                (batch_size, seq_len + 1),
                fill_value=self.pad_id,
                dtype=torch.long,
            )
            extended_input_ids[:, :seq_len] = padded_items
            extended_mask = torch.zeros((batch_size, seq_len + 1), dtype=torch.bool)
            extended_mask[:, :seq_len] = attention_mask

            mask_positions = attention_mask.sum(dim=1)
            extended_input_ids[torch.arange(batch_size), mask_positions] = self.mask_id
            extended_mask[torch.arange(batch_size), mask_positions] = True

            mlm_mask = torch.zeros((batch_size, seq_len + 1), dtype=torch.bool)
            mlm_mask[torch.arange(batch_size), mask_positions] = True

            processed["input_ids"] = extended_input_ids
            processed["mask"] = extended_mask
            processed["mlm_mask"] = mlm_mask

        return processed


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    max_sequence_length: int = 50,
    batch_size: int = 256,
    num_workers: int = 4,
    seq_encoder_type: str = "sasrec",
    num_items: int = 0,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        max_sequence_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        seq_encoder_type: "sasrec" or "bert4rec"
        num_items: Number of items in dataset

    Returns:
        dataloaders: Dictionary with "train", "validation", "test" keys
    """
    splits = ["train", "validation", "test"]
    paths = [train_path, val_path, test_path]

    datasets = {}
    for split, path in zip(splits, paths):
        datasets[split] = SequenceDataset(
            data_path=path,
            max_sequence_length=max_sequence_length,
            mode=split,
        )

    collators = {}
    for split in splits:
        collators[split] = Collator(
            pad_id=0,
            mask_id=num_items + 1,
            num_items=num_items,
            mode=split,
            seq_encoder_type=seq_encoder_type,
        )

    dataloaders = {}
    for split in splits:
        dataloaders[split] = DataLoader(
            dataset=datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collators[split],
        )

    return dataloaders


def get_dataset_stats(data_path: str) -> Dict:
    """
    Get statistics for a dataset.

    Args:
        data_path: Path to CSV file

    Returns:
        stats: Dictionary with dataset statistics
    """
    df = pd.read_csv(data_path)

    user_col = "user_id" if "user_id" in df.columns else df.columns[0]
    item_col = "item_id" if "item_id" in df.columns else df.columns[1]

    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_interactions = len(df)

    user_counts = df.groupby(user_col).size()
    item_counts = df.groupby(item_col).size()

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "sparsity": 1 - n_interactions / (n_users * n_items),
        "avg_user_interactions": user_counts.mean(),
        "avg_item_interactions": item_counts.mean(),
        "min_user_interactions": user_counts.min(),
        "max_user_interactions": user_counts.max(),
        "min_item_interactions": item_counts.min(),
        "max_item_interactions": item_counts.max(),
    }

    return stats
