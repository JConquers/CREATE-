"""Base dataset class for sequential recommendation datasets."""

import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BaseDataset:
    """Base class for dataset loaders.

    Provides common functionality:
    - Train/val/test DataFrames with contiguous user/item IDs
    - User-item interaction mappings
    - Pickled cache for fast reloading
    - Graph edge construction for signed graph learning
    """

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
        self.user_sequences = {}

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
            'avg_interactions_per_user': len(self.train_df) / self.num_users,
        }

    def _save_processed_data(self, extra_data: dict = None):
        """Save processed data to pickle file."""
        data = {
            'train_df': self.train_df,
            'val_df': self.val_df,
            'test_df': self.test_df,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'user2items': dict(self.user2items),
            'item2users': dict(self.item2users),
            'user_sequences': self.user_sequences,
        }
        if extra_data:
            data.update(extra_data)

        self.processed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.processed_file, 'wb') as f:
            pickle.dump(data, f)

    def _load_processed_data(self):
        """Load processed data from pickle file."""
        with open(self.processed_file, 'rb') as f:
            data = pickle.load(f)
            self.train_df = data['train_df']
            self.val_df = data['val_df']
            self.test_df = data['test_df']
            self.num_users = data['num_users']
            self.num_items = data['num_items']
            self.user2items = data['user2items']
            self.item2users = data['item2users']
            self.user_sequences = data.get('user_sequences', {})


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
