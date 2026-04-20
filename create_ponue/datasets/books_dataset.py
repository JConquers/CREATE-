"""
Amazon Books dataset loader for CREATE-Pone.
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

    def process(self, rating_threshold=4):
        """
        Process the raw CSV into train/val/test splits with positive/negative graphs.

        Args:
            rating_threshold: Ratings >= threshold are positive, else negative

        Returns:
            data: Dictionary with processed data
        """
        # Check for threshold-specific cached file
        threshold_file = self.processed_dir / f"books_data_thresh{rating_threshold}.pt"
        if threshold_file.exists():
            print(f"Processed data already exists at {threshold_file}")
            return torch.load(threshold_file, weights_only=False)

        print("Processing dataset...")

        df = pd.read_csv(self.raw_file, compression="gzip")
        print(f"Loaded {len(df)} interactions")

        # Sort by user and timestamp
        df = df.sort_values(["user_id", "timestamp"])

        # Map to indices
        user_col = "user_id"
        item_col = "parent_asin" if "parent_asin" in df.columns else "asin"

        user2idx = {u: i for i, u in enumerate(df[user_col].unique())}
        item2idx = {i: idx for idx, i in enumerate(df[item_col].unique())}

        df["user_idx"] = df[user_col].map(user2idx)
        df["item_idx"] = df[item_col].map(item2idx)

        n_users = len(user2idx)
        n_items = len(item2idx)

        print(f"Users: {n_users}, Items: {n_items}")

        # Split by time and build sequences
        train_user_list, train_item_list = [], []
        val_user_list, val_item_list = [], []
        test_user_list, test_item_list = [], []

        # Positive and negative edges
        pos_user_list, pos_item_list = [], []
        neg_user_list, neg_item_list = [], []

        for user_id, group in df.groupby("user_idx"):
            group = group.sort_values("timestamp")
            items = group["item_idx"].tolist()
            ratings = group["rating"].tolist() if "rating" in group.columns else [5] * len(items)

            if len(group) >= 3:
                # Last item for test, second last for validation, rest for train
                test_user_list.append(user_id)
                test_item_list.append(items[-1])
                val_user_list.append(user_id)
                val_item_list.append(items[-2])

                for item, rating in zip(items[:-2], ratings[:-2]):
                    train_user_list.append(user_id)
                    train_item_list.append(item)
                    # Build positive/negative graphs
                    if rating >= rating_threshold:
                        pos_user_list.append(user_id)
                        pos_item_list.append(item)
                    else:
                        neg_user_list.append(user_id)
                        neg_item_list.append(item)

            elif len(group) == 2:
                test_user_list.append(user_id)
                test_item_list.append(items[-1])
                val_user_list.append(user_id)
                val_item_list.append(items[0])

                item, rating = items[0], ratings[0]
                train_user_list.append(user_id)
                train_item_list.append(item)
                if rating >= rating_threshold:
                    pos_user_list.append(user_id)
                    pos_item_list.append(item)
                else:
                    neg_user_list.append(user_id)
                    neg_item_list.append(item)

            elif len(group) == 1:
                test_user_list.append(user_id)
                test_item_list.append(items[0])

        # Build user sequences for sequential model
        user_sequences = {}
        for user_id, group in df.groupby("user_idx"):
            group = group.sort_values("timestamp")
            user_sequences[user_id] = group["item_idx"].tolist()

        data = {
            "edge_index": torch.tensor([train_user_list, train_item_list], dtype=torch.long),
            "pos_edge_index": torch.tensor([pos_user_list, pos_item_list], dtype=torch.long) if pos_user_list else torch.tensor([[0], [0]], dtype=torch.long),
            "neg_edge_index": torch.tensor([neg_user_list, neg_item_list], dtype=torch.long) if neg_user_list else torch.tensor([[0], [0]], dtype=torch.long),
            "n_users": n_users,
            "n_items": n_items,
            "train_user": torch.tensor(train_user_list, dtype=torch.long),
            "train_item": torch.tensor(train_item_list, dtype=torch.long),
            "val_user": torch.tensor(val_user_list, dtype=torch.long),
            "val_item": torch.tensor(val_item_list, dtype=torch.long),
            "test_user": torch.tensor(test_user_list, dtype=torch.long),
            "test_item": torch.tensor(test_item_list, dtype=torch.long),
            "user_sequences": user_sequences,
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, threshold_file)
        print(f"Saved processed data to {threshold_file}")
        return data

    def get_stats(self, rating_threshold=4):
        """Return dataset statistics."""
        data = self.process(rating_threshold=rating_threshold)
        return {
            "n_users": data["n_users"],
            "n_items": data["n_items"],
            "n_interactions": data["edge_index"].shape[1],
            "n_positive": data["pos_edge_index"].shape[1],
            "n_negative": data["neg_edge_index"].shape[1],
        }

    def load(self, rating_threshold=4):
        """Main entry point to load and process the dataset."""
        self.download()
        return self.process(rating_threshold=rating_threshold)
