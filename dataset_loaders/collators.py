"""Collators and dataset classes for sequential recommendation models."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """PyTorch Dataset for sequential recommendations."""

    def __init__(self, user_sequences: dict, mode: str = 'train'):
        """
        Args:
            user_sequences: Dict mapping user_id to list of item_ids
            mode: One of 'train', 'validation', 'test'
        """
        self.mode = mode
        # Pre-compute flattened arrays for faster batching
        self.user_ids = []
        self.item_sequences = []
        for user_id, item_seq in sorted(user_sequences.items(), key=lambda x: x[0]):
            self.user_ids.append(user_id)
            self.item_sequences.append(item_seq)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        return {
            'user.ids': self.user_ids[index],
            'item.ids': self.item_sequences[index],
        }


class SASRecCollator:
    """Optimized collator for SASRec model batches."""

    def __init__(self, pad_id: int = 0, mode: str = 'train'):
        self.pad_id = pad_id
        self.mode = mode

    def __call__(self, batch):
        batch_size = len(batch)

        # Pre-allocate lists with known sizes
        user_ids = [sample['user.ids'] for sample in batch]

        if self.mode == 'train':
            # Training: context = all but last, label = all but first
            context_seqs = [sample['item.ids'][:-1] for sample in batch]
            label_seqs = [sample['item.ids'][1:] for sample in batch]
        else:
            # Validation/test: context = all but last, label = last item
            context_seqs = [sample['item.ids'][:-1] for sample in batch]
            label_seqs = [sample['item.ids'][-1:] for sample in batch]

        # Get sequence lengths
        seq_lengths = [len(seq) for seq in context_seqs]
        max_len = max(seq_lengths) if seq_lengths else 0

        # Pre-allocate tensors
        padded_seq = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        labels = []

        for i, (ctx_seq, lbl_seq, length) in enumerate(zip(context_seqs, label_seqs, seq_lengths)):
            if length > 0:
                padded_seq[i, :length] = torch.tensor(ctx_seq, dtype=torch.long)
                mask[i, :length] = True
            labels.extend(lbl_seq)

        return {
            'user.ids': torch.tensor(user_ids, dtype=torch.long),
            'padded_sequence_ids': padded_seq,
            'mask': mask,
            'labels.ids': torch.tensor(labels, dtype=torch.long),
        }
