"""Base dataset class for sequential recommendation datasets."""

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data


class BaseDataset(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, data_dir: str, max_sequence_length: int = 50):
        self.data_dir = Path(data_dir)
        self.max_sequence_length = max_sequence_length
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.num_users = 0
        self.num_items = 0
        self.user2items = defaultdict(list)
        self.item2users = defaultdict(list)

    @abstractmethod
    def load_data(self):
        """Load and preprocess the dataset."""
        pass

    def build_user_item_index(self):
        """Build user-to-items and item-to-users mappings."""
        for _, row in self.train_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            self.user2items[user_id].append(item_id)
            self.item2users[item_id].append(user_id)

    def get_statistics(self) -> dict:
        """Return dataset statistics."""
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': len(self.train_df),
            'sparsity': 1 - len(self.train_df) / (self.num_users * self.num_items),
            'avg_sequence_length': np.mean([len(seq) for seq in self.user2items.values()])
        }


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


def build_graph_edges(train_df: pd.DataFrame, num_users: int, num_items: int,
                      offset: float = 3.5, device: str = 'cpu') -> tuple:
    """
    Build positive and negative graph edges for PoneGNN.

    Args:
        train_df: DataFrame with user_id, item_id, rating columns
        num_users: Number of users
        num_items: Number of items
        offset: Rating threshold for positive/negative split
        device: torch device

    Returns:
        data_p, data_n: PyTorch Geometric Data objects for positive/negative edges
    """
    # Positive edges (rating > offset)
    pos_train = train_df[train_df['rating'] > offset]
    edge_user_pos = torch.tensor(pos_train['user_id'].values, dtype=torch.long)
    edge_item_pos = torch.tensor(pos_train['item_id'].values, dtype=torch.long) + num_users

    edge_p = torch.stack([
        torch.cat([edge_user_pos, edge_item_pos]),
        torch.cat([edge_item_pos, edge_user_pos])
    ], dim=0)

    # Negative edges (rating < offset)
    neg_offset = 3.5
    neg_train = train_df[train_df['rating'] < neg_offset]
    edge_user_neg = torch.tensor(neg_train['user_id'].values, dtype=torch.long)
    edge_item_neg = torch.tensor(neg_train['item_id'].values, dtype=torch.long) + num_users

    edge_n = torch.stack([
        torch.cat([edge_user_neg, edge_item_neg]),
        torch.cat([edge_item_neg, edge_user_neg])
    ], dim=0)

    data_p = Data(edge_index=edge_p)
    data_n = Data(edge_index=edge_n)

    return data_p, data_n
