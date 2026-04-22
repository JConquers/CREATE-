"""
Amazon Books dataset loader for CREATE++.
Downloads and processes the 5-core Books dataset from Amazon Reviews 2023.
"""

import pandas as pd
import torch
from torch_geometric.data import download_url
from pathlib import Path


class BooksDataset:
    """Amazon Books dataset (5-core)."""

    URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz"

    def __init__(self, root="data"):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_file = self.raw_dir / "Books.csv.gz"
        self.processed_file = self.processed_dir / "books_data.pt"

    def download(self):
        """Download the dataset if not present."""
        if self.raw_file.exists():
            print(f"Dataset already exists at {self.raw_file}")
            return

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Books dataset from {self.URL}...")
        download_url(self.URL, self.raw_dir)

    def process(self):
        """Process the raw CSV into train/test splits."""
        if self.processed_file.exists():
            print(f"Processed data already exists at {self.processed_file}")
            return self._load_processed()

        print("Processing dataset...")

        df = pd.read_csv(self.raw_file, compression='gzip')
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample rows:\n{df.head()}")
        df = df.sort_values(['user_id', 'timestamp'])

        user_col = 'user_id'
        item_col = 'parent_asin' if 'parent_asin' in df.columns else 'asin'

        user2idx = {u: i for i, u in enumerate(df[user_col].unique())}
        item2idx = {i: idx for idx, i in enumerate(df[item_col].unique())}

        df['user_idx'] = df[user_col].map(user2idx)
        df['item_idx'] = df[item_col].map(item2idx)

        n_users = len(user2idx)
        n_items = len(item2idx)

        train_user_list, train_item_list, train_time_list = [], [], []
        val_user_list, val_item_list, val_time_list = [], [], []
        test_user_list, test_item_list, test_time_list = [], [], []
        
        for user_id, group in df.groupby('user_idx'):
            group = group.sort_values('timestamp')
            if len(group) >= 3:
                # Last item for test, second last for validation, rest for train
                test_user_list.append(user_id)
                test_item_list.append(group.iloc[-1]['item_idx'])
                test_time_list.append(group.iloc[-1]['timestamp'])
                val_user_list.append(user_id)
                val_item_list.append(group.iloc[-2]['item_idx'])
                val_time_list.append(group.iloc[-2]['timestamp'])
                for _, row in group.iloc[:-2].iterrows():
                    train_user_list.append(user_id)
                    train_item_list.append(row['item_idx'])
                    train_time_list.append(row['timestamp'])
            elif len(group) == 2:
                # Last item for test, first for validation
                test_user_list.append(user_id)
                test_item_list.append(group.iloc[-1]['item_idx'])
                test_time_list.append(group.iloc[-1]['timestamp'])
                val_user_list.append(user_id)
                val_item_list.append(group.iloc[0]['item_idx'])
                val_time_list.append(group.iloc[0]['timestamp'])
            elif len(group) == 1:
                # Only one interaction, put in test
                test_user_list.append(user_id)
                test_item_list.append(group.iloc[0]['item_idx'])
                test_time_list.append(group.iloc[0]['timestamp'])

        edge_index = torch.tensor([train_user_list, train_item_list], dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1])

        data = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'n_users': n_users,
            'n_items': n_items,
            'train_user': torch.tensor(train_user_list, dtype=torch.long),
            'train_item': torch.tensor(train_item_list, dtype=torch.long),
            'train_time': torch.tensor(train_time_list, dtype=torch.float),
            'val_user': torch.tensor(val_user_list, dtype=torch.long),
            'val_item': torch.tensor(val_item_list, dtype=torch.long),
            'val_time': torch.tensor(val_time_list, dtype=torch.float),
            'test_user': torch.tensor(test_user_list, dtype=torch.long),
            'test_item': torch.tensor(test_item_list, dtype=torch.long),
            'test_time': torch.tensor(test_time_list, dtype=torch.float),
            'user2idx': user2idx,
            'item2idx': item2idx,
        }

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, self.processed_file)
        print(f"Saved processed data to {self.processed_file}")
        return data

    def _load_processed(self):
        """Load processed data from disk."""
        return torch.load(self.processed_file)

    def get_edge_index(self):
        """Get the interaction graph edge index."""
        data = self.process()
        return data['edge_index'], data['edge_weight']

    def get_stats(self):
        """Return dataset statistics."""
        data = self.process()
        return {
            'n_users': data['n_users'],
            'n_items': data['n_items'],
            'n_interactions': data['edge_index'].shape[1]
        }

    def load(self):
        """Main entry point to load and process the dataset."""
        self.download()
        return self.process()


if __name__ == "__main__":
    dataset = BooksDataset(root="data/Books")
    stats = dataset.load()
    print(f"Dataset stats: {stats}")
