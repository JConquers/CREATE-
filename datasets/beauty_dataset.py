"""
Amazon Beauty and Personal Care dataset loader for CREATE++.
Downloads and processes the 5-core Beauty dataset from Amazon Reviews 2023.
"""

import os
import gzip
import json
import pandas as pd
import torch
from torch_geometric.data import download_url, extract_gz
from pathlib import Path


class BeautyDataset:
    """Amazon Beauty and Personal Care dataset."""

    URL = "https://datarepo.eng.ucsd.edu/mcsp-group/amazon/reviews/Beauty_and_Personal_Care_5.json.gz"

    def __init__(self, root="data"):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_file = self.raw_dir / "Beauty_and_Personal_Care_5.json.gz"
        self.processed_file = self.processed_dir / "beauty_data.pt"

    def download(self):
        """Download the dataset if not present."""
        if self.raw_file.exists():
            print(f"Dataset already exists at {self.raw_file}")
            return

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Beauty dataset from {self.URL}...")
        download_url(self.URL, self.raw_dir)

    def process(self):
        """Process the raw JSON into train/test splits."""
        if self.processed_file.exists():
            print(f"Processed data already exists at {self.processed_file}")
            return self._load_processed()

        print("Processing dataset...")

        interactions = []
        with gzip.open(self.raw_file, 'rt', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line)
                interactions.append({
                    'user_id': review['user_id'],
                    'item_id': review['item_id'],
                    'rating': review.get('rating', 5.0),
                    'timestamp': review.get('unixReviewTime', 0)
                })

        df = pd.DataFrame(interactions)
        df = df.sort_values(['user_id', 'timestamp'])

        user2idx = {u: i for i, u in enumerate(df['user_id'].unique())}
        item2idx = {i: idx for idx, i in enumerate(df['item_id'].unique())}

        df['user_idx'] = df['user_id'].map(user2idx)
        df['item_idx'] = df['item_id'].map(item2idx)

        n_users = len(user2idx)
        n_items = len(item2idx)

        train_data, test_data = [], []
        for user_id, group in df.groupby('user_idx'):
            group = group.sort_values('timestamp')
            if len(group) < 3:
                test_data.append(group.iloc[-1])
                train_data.extend(group.iloc[:-1].values.tolist())
            else:
                test_data.append(group.iloc[-1])
                train_data.extend(group.iloc[:-1].values.tolist())

        train_df = pd.DataFrame(train_data, columns=df.columns)
        test_df = pd.DataFrame(test_data, columns=df.columns)

        edge_index = torch.tensor(
            list(zip(train_df['user_idx'], train_df['item_idx'])),
            dtype=torch.long
        ).t()

        edge_weight = torch.ones(edge_index.shape[1])

        data = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'n_users': n_users,
            'n_items': n_items,
            'train_user': torch.tensor(train_df['user_idx'].values, dtype=torch.long),
            'train_item': torch.tensor(train_df['item_idx'].values, dtype=torch.long),
            'test_user': torch.tensor(test_df['user_idx'].values, dtype=torch.long),
            'test_item': torch.tensor(test_df['item_idx'].values, dtype=torch.long),
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
    dataset = BeautyDataset(root="data/Beauty_and_Personal_Care")
    stats = dataset.load()
    print(f"Dataset stats: {stats}")
