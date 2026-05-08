"""
MovieLens-1M dataset loader for CREATE-Uni.

Downloads and processes MovieLens-1M ratings into the same tensor format used by
the Amazon Beauty and Office Products loaders.
"""

import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import download_url


class MovieLens1MDataset:
    """MovieLens-1M ratings dataset."""

    URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(self, root="data"):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.raw_archive = self.raw_dir / "ml-1m.zip"
        self.raw_file = self.raw_dir / "ml-1m" / "ratings.dat"
        self.flat_raw_file = self.raw_dir / "ratings.dat"
        self.processed_file = self.processed_dir / "ml1m_data.pt"

    def download(self):
        """Download and extract the dataset if not present."""
        if self.raw_file.exists() or self.flat_raw_file.exists():
            print(f"Dataset already exists at {self._ratings_path()}")
            return

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if not self.raw_archive.exists():
            print(f"Downloading MovieLens-1M dataset from {self.URL}...")
            download_url(self.URL, self.raw_dir)

        print(f"Extracting {self.raw_archive}...")
        with zipfile.ZipFile(self.raw_archive, "r") as archive:
            archive.extractall(self.raw_dir)

        if not self.raw_file.exists():
            raise FileNotFoundError(f"Expected ratings file not found at {self.raw_file}")

    def _ratings_path(self):
        if self.raw_file.exists():
            return self.raw_file
        return self.flat_raw_file

    def process(self):
        """Process ratings.dat into leave-last-two train/val/test splits."""
        if self.processed_file.exists():
            print(f"Processed data already exists at {self.processed_file}")
            return self._load_processed()

        print("Processing MovieLens-1M dataset...")

        df = pd.read_csv(
            self._ratings_path(),
            sep="::",
            engine="python",
            names=["user_id_raw", "item_id_raw", "rating", "timestamp"],
        )
        df["row_order"] = range(len(df))
        df = df.sort_values(["user_id_raw", "timestamp", "row_order"])

        user2idx = {u: idx for idx, u in enumerate(df["user_id_raw"].unique())}
        item2idx = {i: idx for idx, i in enumerate(df["item_id_raw"].unique())}

        df["user_idx"] = df["user_id_raw"].map(user2idx)
        df["item_idx"] = df["item_id_raw"].map(item2idx)

        n_users = len(user2idx)
        n_items = len(item2idx)

        train_user_list, train_item_list, train_time_list, train_rating_list = [], [], [], []
        val_user_list, val_item_list, val_time_list, val_rating_list = [], [], [], []
        test_user_list, test_item_list, test_time_list, test_rating_list = [], [], [], []

        for user_id, group in df.groupby("user_idx", sort=False):
            group = group.sort_values(["timestamp", "row_order"])
            if len(group) >= 3:
                test_row = group.iloc[-1]
                val_row = group.iloc[-2]

                test_user_list.append(user_id)
                test_item_list.append(test_row["item_idx"])
                test_time_list.append(test_row["timestamp"])
                test_rating_list.append(test_row["rating"])

                val_user_list.append(user_id)
                val_item_list.append(val_row["item_idx"])
                val_time_list.append(val_row["timestamp"])
                val_rating_list.append(val_row["rating"])

                for _, row in group.iloc[:-2].iterrows():
                    train_user_list.append(user_id)
                    train_item_list.append(row["item_idx"])
                    train_time_list.append(row["timestamp"])
                    train_rating_list.append(row["rating"])
            elif len(group) == 2:
                test_row = group.iloc[-1]
                val_row = group.iloc[0]

                test_user_list.append(user_id)
                test_item_list.append(test_row["item_idx"])
                test_time_list.append(test_row["timestamp"])
                test_rating_list.append(test_row["rating"])

                val_user_list.append(user_id)
                val_item_list.append(val_row["item_idx"])
                val_time_list.append(val_row["timestamp"])
                val_rating_list.append(val_row["rating"])
            elif len(group) == 1:
                test_row = group.iloc[0]
                test_user_list.append(user_id)
                test_item_list.append(test_row["item_idx"])
                test_time_list.append(test_row["timestamp"])
                test_rating_list.append(test_row["rating"])

        edge_index = torch.tensor([train_user_list, train_item_list], dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1])

        data = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "n_users": n_users,
            "n_items": n_items,
            "train_user": torch.tensor(train_user_list, dtype=torch.long),
            "train_item": torch.tensor(train_item_list, dtype=torch.long),
            "train_time": torch.tensor(train_time_list, dtype=torch.float),
            "train_rating": torch.tensor(train_rating_list, dtype=torch.float),
            "val_user": torch.tensor(val_user_list, dtype=torch.long),
            "val_item": torch.tensor(val_item_list, dtype=torch.long),
            "val_time": torch.tensor(val_time_list, dtype=torch.float),
            "val_rating": torch.tensor(val_rating_list, dtype=torch.float),
            "test_user": torch.tensor(test_user_list, dtype=torch.long),
            "test_item": torch.tensor(test_item_list, dtype=torch.long),
            "test_time": torch.tensor(test_time_list, dtype=torch.float),
            "test_rating": torch.tensor(test_rating_list, dtype=torch.float),
            "user2idx": user2idx,
            "item2idx": item2idx,
        }

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, self.processed_file)
        print(f"Saved processed data to {self.processed_file}")
        return data

    def _load_processed(self):
        """Load processed data from disk."""
        return torch.load(self.processed_file, weights_only=False)

    def get_edge_index(self):
        """Get the interaction graph edge index."""
        data = self.process()
        return data["edge_index"], data["edge_weight"]

    def get_stats(self):
        """Return dataset statistics."""
        data = self.process()
        return {
            "n_users": data["n_users"],
            "n_items": data["n_items"],
            "n_interactions": data["edge_index"].shape[1],
        }

    def load(self):
        """Main entry point to load and process the dataset."""
        self.download()
        return self.process()


if __name__ == "__main__":
    dataset = MovieLens1MDataset(root="data/ml1m")
    stats = dataset.load()
    print(f"Dataset stats: {stats}")
