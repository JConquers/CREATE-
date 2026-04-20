"""
Amazon Beauty and Personal Care dataset loader for CREATE++.
Downloads and processes the 5-core Beauty dataset from Amazon Reviews 2023.
"""

import pandas as pd
import torch
from torch_geometric.data import download_url
from pathlib import Path


class BeautyDataset:
    """Amazon Beauty and Personal Care dataset (5-core)."""

    URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Beauty_and_Personal_Care.csv.gz"

    def __init__(self, root="data"):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_file = self.raw_dir / "Beauty_and_Personal_Care.csv.gz"
        self.processed_file = self.processed_dir / "beauty_data.pt"
        self.sequences_file = self.processed_dir / "beauty_sequences.pt"

    def download(self):
        """Download the dataset if not present."""
        if self.raw_file.exists():
            print(f"Dataset already exists at {self.raw_file}")
            return

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Beauty dataset from {self.URL}...")
        download_url(self.URL, self.raw_dir)

    def _create_sequences(self, df, max_seq_len):
        """Create padded sequences from interaction dataframe."""
        import pandas as pd
        df = df.sort_values('user_idx')
        sequences = []
        for user_id, group in df.groupby('user_idx', sort=False):
            items = group['item_idx'].values
            if len(items) >= 2:
                for i in range(1, len(items)):
                    start = max(0, i - max_seq_len)
                    seq = items[start:i]
                    pad_len = max_seq_len - len(seq)
                    sequences.append([0] * pad_len + seq.tolist())
        return torch.tensor(sequences, dtype=torch.long) if sequences else torch.zeros(1, max_seq_len, dtype=torch.long)

    def process(self, max_seq_len=50, rating_offset=3.5):
        """Process the raw CSV into train/test splits with dual feedback graphs.

        Args:
            max_seq_len: max sequence length for sequences (not used in processing)
            rating_offset: threshold for positive/negative feedback (default 3.5)
        """
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
        item_offset = n_users  # offset for item indices in unified embedding table

        # Rating offset for positive/negative feedback
        df['weight'] = df['rating'] - rating_offset  # positive = likes, negative = dislikes

        train_user_list, train_item_list, train_weight_list = [], [], []
        val_user_list, val_item_list, test_user_list, test_item_list = [], [], [], []

        for user_id, group in df.groupby('user_idx'):
            group = group.sort_values('timestamp')
            if len(group) >= 3:
                test_user_list.append(user_id)
                test_item_list.append(group.iloc[-1]['item_idx'])
                val_user_list.append(user_id)
                val_item_list.append(group.iloc[-2]['item_idx'])
                for _, row in group.iloc[:-2].iterrows():
                    train_user_list.append(user_id)
                    train_item_list.append(row['item_idx'])
                    train_weight_list.append(row['weight'])
            elif len(group) == 2:
                test_user_list.append(user_id)
                test_item_list.append(group.iloc[-1]['item_idx'])
                val_user_list.append(user_id)
                val_item_list.append(group.iloc[0]['item_idx'])
            elif len(group) == 1:
                test_user_list.append(user_id)
                test_item_list.append(group.iloc[0]['item_idx'])

        edge_weight = torch.ones(len(train_user_list))

        # Positive edges: rating > offset
        pos_df = df[df['rating'] > rating_offset]
        pos_edge_user = torch.tensor(pos_df['user_idx'].values, dtype=torch.long)
        pos_edge_item = torch.tensor(pos_df['item_idx'].values, dtype=torch.long) + item_offset
        pos_edge_index = torch.stack([
            torch.cat([pos_edge_user, pos_edge_item], dim=0),
            torch.cat([pos_edge_item, pos_edge_user], dim=0)
        ], dim=0)  # undirected

        # Negative edges: rating < offset
        neg_df = df[df['rating'] < rating_offset]
        neg_edge_user = torch.tensor(neg_df['user_idx'].values, dtype=torch.long)
        neg_edge_item = torch.tensor(neg_df['item_idx'].values, dtype=torch.long) + item_offset
        neg_edge_index = torch.stack([
            torch.cat([neg_edge_user, neg_edge_item], dim=0),
            torch.cat([neg_edge_item, neg_edge_user], dim=0)
        ], dim=0)  # undirected

        data = {
            'edge_index': torch.tensor([train_user_list, train_item_list], dtype=torch.long),
            'edge_weight': edge_weight,
            'pos_edge_index': pos_edge_index,
            'neg_edge_index': neg_edge_index,
            'n_users': n_users,
            'n_items': n_items,
            'train_user': torch.tensor(train_user_list, dtype=torch.long),
            'train_item': torch.tensor(train_item_list, dtype=torch.long),
            'train_weight': torch.tensor(train_weight_list, dtype=torch.float),
            'val_user': torch.tensor(val_user_list, dtype=torch.long),
            'val_item': torch.tensor(val_item_list, dtype=torch.long),
            'test_user': torch.tensor(test_user_list, dtype=torch.long),
            'test_item': torch.tensor(test_item_list, dtype=torch.long),
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

    def get_sequences(self, max_seq_len=50):
        """Get cached sequences indexed by global user ID (0..n_users-1).

        Builds one sequence per user (last seq_len items of their training history).
        Returns a tensor of shape (n_users, max_seq_len).
        """
        if self.sequences_file.exists():
            return torch.load(self.sequences_file)
        data = self.process()
        # Build one sequence per user — last max_seq_len items of their training history
        user_items = {}
        for u, it in zip(data['train_user'].tolist(), data['train_item'].tolist()):
            if u not in user_items:
                user_items[u] = []
            user_items[u].append(it)
        n_users = data['n_users']
        seqs = torch.zeros((n_users, max_seq_len), dtype=torch.long)
        for u, items in user_items.items():
            seq = items[-max_seq_len:]
            seqs[u, max_seq_len - len(seq):] = torch.tensor(seq, dtype=torch.long)
        torch.save(seqs, self.sequences_file)
        print(f"Saved sequences to {self.sequences_file}")
        return seqs

    def load(self):
        """Main entry point to load and process the dataset."""
        self.download()
        return self.process()


if __name__ == "__main__":
    dataset = BeautyDataset(root="data/Beauty_and_Personal_Care")
    stats = dataset.load()
    print(f"Dataset stats: {stats}")
