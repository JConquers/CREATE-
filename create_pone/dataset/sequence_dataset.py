"""Sequence dataset and collator for next-item prediction."""

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class UserSequenceDataset(Dataset):
    """Simple user-level sequence dataset for sequential training."""

    def __init__(self, user_sequences: Dict[int, List[int]], min_sequence_length: int = 2):
        self._samples: List[dict] = []

        for user_id, sequence in user_sequences.items():
            if len(sequence) < min_sequence_length:
                continue
            self._samples.append(
                {
                    "user_id": int(user_id),
                    "item_ids": [int(item_id) for item_id in sequence],
                }
            )

        self._samples.sort(key=lambda sample: sample["user_id"])

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict:
        return self._samples[index]


class NextItemCollator:
    """Pad and batch user sequences for autoregressive next-item training."""

    def __init__(self, max_sequence_length: int, pad_id: int):
        self.max_sequence_length = max_sequence_length
        self.pad_id = pad_id

    def __call__(self, batch: List[dict]) -> dict:
        user_ids: List[int] = []
        contexts: List[List[int]] = []
        targets: List[List[int]] = []

        for sample in batch:
            sequence = sample["item_ids"][-(self.max_sequence_length + 1) :]
            context = sequence[:-1]
            target = sequence[1:]

            if not context:
                continue

            user_ids.append(sample["user_id"])
            contexts.append(context)
            targets.append(target)

        if not contexts:
            raise RuntimeError("Batch has no valid sequences with at least two events.")

        batch_size = len(contexts)
        max_len = max(len(context) for context in contexts)

        input_ids = torch.full(
            (batch_size, max_len),
            fill_value=self.pad_id,
            dtype=torch.long,
        )
        target_ids = torch.full(
            (batch_size, max_len),
            fill_value=-100,
            dtype=torch.long,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for idx, (context, target) in enumerate(zip(contexts, targets)):
            length = len(context)
            input_ids[idx, :length] = torch.tensor(context, dtype=torch.long)
            target_ids[idx, :length] = torch.tensor(target, dtype=torch.long)
            attention_mask[idx, :length] = True

        return {
            "user_ids": torch.tensor(user_ids, dtype=torch.long),
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "sequence_lengths": attention_mask.long().sum(dim=1),
        }
