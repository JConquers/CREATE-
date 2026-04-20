"""Amazon Beauty dataset loader with automatic download and preprocessing."""

import gzip
import pickle
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

    def __init__(self, data_dir: str, max_sequence_length: int = 50):
        super().__init__(data_dir, max_sequence_length)
        self.dataset_name = 'amazon_beauty'
        self.data_dir = Path(data_dir)
        self.raw_file = self.data_dir / 'Beauty_and_Personal_Care.csv.gz'
        self.processed_file = self.data_dir / 'beauty_processed.pkl'

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

    def _preprocess(self):
        """Preprocess the raw dataset with leave-last-out splitting."""
        if self.processed_file.exists():
            print(f"Loading preprocessed data from {self.processed_file}")
            self._load_processed_data()
            return

        self._download()
        print("Preprocessing dataset...")

        with gzip.open(self.raw_file, 'rt') as f:
            df = pd.read_csv(f)

        # Amazon 2023 dataset column names: user_id, item_id, rating, timestamp
        # Check and rename columns if needed
        print(f"Original columns: {df.columns.tolist()}")

        # Handle different column name formats
        column_mapping = {}
        if 'user_id' not in df.columns:
            if 'User_ID' in df.columns:
                column_mapping['User_ID'] = 'user_id'
            elif 'userId' in df.columns:
                column_mapping['userId'] = 'user_id'
            elif 'reviewerID' in df.columns:
                column_mapping['reviewerID'] = 'user_id'

        if 'item_id' not in df.columns:
            if 'Item_ID' in df.columns:
                column_mapping['Item_ID'] = 'item_id'
            elif 'asin' in df.columns:
                column_mapping['asin'] = 'item_id'
            elif 'product_id' in df.columns:
                column_mapping['product_id'] = 'item_id'

        if 'rating' not in df.columns:
            if 'Rating' in df.columns:
                column_mapping['Rating'] = 'rating'
            elif 'stars' in df.columns:
                column_mapping['stars'] = 'rating'

        if 'timestamp' not in df.columns:
            if 'Timestamp' in df.columns:
                column_mapping['Timestamp'] = 'timestamp'
            elif 'unix_review_time' in df.columns:
                column_mapping['unix_review_time'] = 'timestamp'
            else:
                # Create synthetic timestamp if not available
                print("No timestamp column found, creating synthetic timestamps...")
                df['timestamp'] = range(len(df))

        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"Renamed columns: {column_mapping}")

        # Select and order columns
        df = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()
        df = df.sort_values(['user_id', 'timestamp'])

        train_rows, val_rows, test_rows = [], [], []

        for user_id, user_data in df.groupby('user_id'):
            user_interactions = user_data.values.tolist()
            n = len(user_interactions)

            if n >= 3:
                train_rows.extend(user_interactions[:-2])
                val_rows.append(user_interactions[-2])
                test_rows.append(user_interactions[-1])
            elif n == 2:
                train_rows.append(user_interactions[0])
                val_rows.append(user_interactions[1])
            else:
                train_rows.extend(user_interactions)

        self.train_df = pd.DataFrame(train_rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        self.val_df = pd.DataFrame(val_rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])
        self.test_df = pd.DataFrame(test_rows, columns=['user_id', 'item_id', 'rating', 'timestamp'])

        all_users = set()
        all_items = set()
        for data_df in [self.train_df, self.val_df, self.test_df]:
            all_users.update(data_df['user_id'].unique())
            all_items.update(data_df['item_id'].unique())

        user_map = {old: new for new, old in enumerate(sorted(all_users))}
        item_map = {old: new for new, old in enumerate(sorted(all_items))}

        for data_df in [self.train_df, self.val_df, self.test_df]:
            data_df['user_id'] = data_df['user_id'].map(user_map).astype(int)
            data_df['item_id'] = data_df['item_id'].map(item_map).astype(int)

        self.num_users = len(all_users)
        self.num_items = len(all_items)

        self.build_user_item_index()
        self.user_sequences = {}
        for user_id, group in self.train_df.groupby('user_id'):
            items = group.sort_values('timestamp')['item_id'].tolist()
            self.user_sequences[user_id] = items[-self.max_sequence_length:]

        self._save_processed_data()

        print("Preprocessing complete!")
        print(f"  Users: {self.num_users:,}")
        print(f"  Items: {self.num_items:,}")
        print(f"  Train interactions: {len(self.train_df):,}")
        print(f"  Val interactions: {len(self.val_df):,}")
        print(f"  Test interactions: {len(self.test_df):,}")

    def load_data(self):
        """Load or preprocess and load the dataset."""
        self._preprocess()
        return self.train_df, self.val_df, self.test_df
