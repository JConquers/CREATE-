"""Collators and dataset classes for sequential recommendation models."""

import torch
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
        self._index = []
        for user_id, item_seq in sorted(user_sequences.items(), key=lambda x: x[0]):
            self._index.append({
                'user.ids': user_id,
                'item.ids': item_seq,
            })

    def __len__(self):
        return len(self._index)

    def __getitem__(self, index):
        return self._index[index]


class SASRecCollator:
    """Collator for SASRec model batches."""

    def __init__(self, pad_id: int = 0, mode: str = 'train'):
        self.pad_id = pad_id
        self.mode = mode

    def __call__(self, batch):
        processed = {
            'user.ids': [],
            'item.ids': [],
            'item.length': [],
            'labels.ids': [],
        }

        for sample in batch:
            processed['user.ids'].append(sample['user.ids'])
            if self.mode == 'train':
                context_seq = sample['item.ids']
                label_seq = sample['item.ids'][1:]
            else:
                context_seq = sample['item.ids'][:-1]
                label_seq = [sample['item.ids'][-1]]

            processed['item.ids'].extend(context_seq)
            processed['item.length'].append(len(context_seq))
            processed['labels.ids'].extend(label_seq)

        for key in processed:
            processed[key] = torch.tensor(processed[key], dtype=torch.long)

        # Pad sequences
        max_len = max(processed['item.length'])
        batch_size = len(processed['item.length'])
        padded_seq = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, length in enumerate(processed['item.length']):
            padded_seq[i, :length] = processed['item.ids'][
                sum(processed['item.length'][:i]):sum(processed['item.length'][:i+1])
            ]
            mask[i, :length] = True

        processed['padded_sequence_ids'] = padded_seq
        processed['mask'] = mask

        return processed
