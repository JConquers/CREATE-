"""Amazon Beauty dataset loader with automatic download and preprocessing."""

import hashlib
import os
import pickle
import gzip
import shutil
from pathlib import Path

import pandas as pd
from .base_dataset import BaseDataset


class AmazonBeautyDataset(BaseDataset):
    """Amazon Beauty and Personal Care dataset (5-core).

    Downloads from:
    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Beauty_and_Personal_Care.csv.gz

    Uses leave-last-out splitting:
    - For each user, the last interaction (by timestamp) goes to test
    - The second-to-last goes to validation
    - All earlier interactions go to training
    """

    URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Beauty_and_Personal_Care.csv.gz"

    @staticmethod
    def _resolve_column(columns: list[str], candidates: list[str], field_name: str) -> str:
        for name in candidates:
            if name in columns:
                return name
        raise KeyError(
            f"Could not find required {field_name} column. "
            f"Tried {candidates}. Available columns: {columns}"
        )

    def _normalize_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = list(df.columns)
        user_col = self._resolve_column(columns, ["user_id", "reviewerID", "reviewer_id"], "user")
        item_col = self._resolve_column(columns, ["item_id", "parent_asin", "asin"], "item")
        rating_col = self._resolve_column(columns, ["rating", "overall"], "rating")
        time_col = self._resolve_column(columns, ["timestamp", "unixReviewTime", "time"], "timestamp")

        normalized = df[[user_col, item_col, rating_col, time_col]].rename(
            columns={
                user_col: "user_id",
                item_col: "item_id",
                rating_col: "rating",
                time_col: "timestamp",
            }
        ).copy()

        normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")
        normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="coerce")

        missing_ts = normalized["timestamp"].isna()
        if missing_ts.any():
            parsed_ts = pd.to_datetime(df.loc[missing_ts, time_col], errors="coerce", utc=True)
            normalized.loc[missing_ts, "timestamp"] = parsed_ts.map(
                lambda x: x.timestamp() if pd.notna(x) else pd.NA
            )
            normalized["timestamp"] = pd.to_numeric(normalized["timestamp"], errors="coerce")

        normalized = normalized.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
        if normalized.empty:
            raise ValueError("No valid interactions after column normalization.")

        normalized["timestamp"] = normalized["timestamp"].astype("int64")
        return normalized

    def __init__(self, data_dir: str, max_sequence_length: int = 50):
        super().__init__(data_dir, max_sequence_length)
        self.dataset_name = 'amazon_beauty'

        # Check multiple possible locations for Kaggle compatibility
        data_dir_path = Path(data_dir)

        # Check if preprocessed file exists in common Kaggle locations
        possible_processed_paths = [
            data_dir_path / 'beauty_processed.pkl',
            Path('./data/beauty_processed.pkl'),
            Path('../input/beauty_processed.pkl'),
            Path('/kaggle/input/beauty_processed.pkl'),
            Path('./beauty_processed.pkl'),
        ]

        self.processed_file = None
        self.raw_file = None

        # Check for preprocessed file first (skip processing if found)
        for p in possible_processed_paths:
            if p.exists():
                self.processed_file = p
                print(f"Found preprocessed data at: {self.processed_file}")
                break

        # If no preprocessed file found, look for raw file
        if self.processed_file is None:
            possible_raw_paths = [
                data_dir_path / 'Beauty_and_Personal_Care.csv.gz',
                Path('./data/Beauty_and_Personal_Care.csv.gz'),
                Path('../input/beauty/Beauty_and_Personal_Care.csv.gz'),
                Path('/kaggle/input/beauty/Beauty_and_Personal_Care.csv.gz'),
            ]
            for p in possible_raw_paths:
                if p.exists():
                    self.raw_file = p
                    self.processed_file = data_dir_path / 'beauty_processed.pkl'
                    print(f"Found raw file at: {self.raw_file}")
                    break

            # Default to download if no raw file found
            if self.raw_file is None:
                self.raw_file = data_dir_path / 'Beauty_and_Personal_Care.csv.gz'
                self.processed_file = data_dir_path / 'beauty_processed.pkl'
        else:
            # Set raw_file for potential re-download if needed
            self.raw_file = data_dir_path / 'Beauty_and_Personal_Care.csv.gz'

    def _download(self):
        """Download the dataset if not already present."""
        if self.raw_file.exists():
            print(f"Raw file already exists: {self.raw_file}")
            return

        print(f"Downloading Amazon Beauty dataset from {self.URL}...")
        import urllib.request

        self.raw_file.parent.mkdir(parents=True, exist_ok=True)

        def reporthook(blocknum, blocksize, totalsize):
            readsofar = blocknum * blocksize
            if totalsize > 0:
                percent = readsofar * 100 / totalsize
                print(f"\rDownload progress: {percent:.1f}%", end='')

        urllib.request.urlretrieve(self.URL, self.raw_file, reporthook)
        print(f"\nDownload complete: {self.raw_file}")

    def _compute_hash(self, filepath: Path) -> str:
        """Compute MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _preprocess(self):
        """Preprocess the raw dataset with leave-last-out splitting."""
        # Check if processed file exists
        if self.processed_file and self.processed_file.exists():
            print(f"Loading preprocessed data from {self.processed_file}")
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
            return

        # Download if needed
        self._download()

        print("Preprocessing dataset...")

        # Read the CSV
        with gzip.open(self.raw_file, 'rt') as f:
            df = pd.read_csv(f)

        # Normalize schema differences across Amazon dumps.
        df = self._normalize_raw_columns(df)

        # Sort by user and timestamp to ensure chronological order
        df = df.sort_values(['user_id', 'timestamp'])

        # Leave-last-out splitting
        train_rows = []
        val_rows = []
        test_rows = []

        for user_id, user_data in df.groupby('user_id'):
            user_interactions = user_data.values.tolist()
            n = len(user_interactions)

            if n >= 3:
                # Train: all except last 2
                train_rows.extend(user_interactions[:-2])
                # Val: second to last
                val_rows.append(user_interactions[-2])
                # Test: last
                test_rows.append(user_interactions[-1])
            elif n == 2:
                # Train: first
                train_rows.append(user_interactions[0])
                # Val: second (no test)
                val_rows.append(user_interactions[1])
            else:
                # Only 1 interaction - put in train
                train_rows.extend(user_interactions)

        self.train_df = pd.DataFrame(train_rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        self.val_df = pd.DataFrame(val_rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        self.test_df = pd.DataFrame(test_rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])

        # Create contiguous IDs
        all_users = set()
        all_items = set()
        for df in [self.train_df, self.val_df, self.test_df]:
            all_users.update(df['user_id'].unique())
            all_items.update(df['item_id'].unique())

        user_map = {old: new for new, old in enumerate(sorted(all_users))}
        item_map = {old: new for new, old in enumerate(sorted(all_items))}

        for df in [self.train_df, self.val_df, self.test_df]:
            df['user_id'] = df['user_id'].map(user_map).astype(int)
            df['item_id'] = df['item_id'].map(item_map).astype(int)

        self.num_users = len(all_users)
        self.num_items = len(all_items)

        # Build mappings
        self.build_user_item_index()

        # Build user sequences for sequential models
        self.user_sequences = {}
        for user_id, group in self.train_df.groupby('user_id'):
            items = group.sort_values('timestamp')['item_id'].tolist()
            self.user_sequences[user_id] = items[-self.max_sequence_length:]

        # Save processed data
        print(f"Saving preprocessed data to {self.processed_file}")
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
        with open(self.processed_file, 'wb') as f:
            pickle.dump(data, f)

        print(f"Preprocessing complete!")
        print(f"  Users: {self.num_users:,}")
        print(f"  Items: {self.num_items:,}")
        print(f"  Train interactions: {len(self.train_df):,}")
        print(f"  Val interactions: {len(self.val_df):,}")
        print(f"  Test interactions: {len(self.test_df):,}")

    def load_data(self):
        """Load or preprocess and load the dataset."""
        self._preprocess()
        return self.train_df, self.val_df, self.test_df

    def get_user_sequences(self) -> dict:
        """Return user-item sequences for sequential models."""
        if not hasattr(self, 'user_sequences'):
            self.user_sequences = {}
            for user_id, group in self.train_df.groupby('user_id'):
                items = group.sort_values('timestamp')['item_id'].tolist()
                self.user_sequences[user_id] = items[-self.max_sequence_length:]
        return self.user_sequences
