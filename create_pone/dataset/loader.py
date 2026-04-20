"""Dataset loading helpers for CREATE-Pone."""

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from dataset_loaders import get_dataset


@dataclass
class DatasetBundle:
    """Container for processed dataset artifacts used by training."""

    name: str
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    user_sequences: Dict[int, List[int]]
    num_users: int
    num_items: int


def load_dataset_bundle(
    dataset_name: str,
    data_dir: str,
    max_sequence_length: int,
) -> DatasetBundle:
    """Load Beauty/Books data using the existing dataset_loaders package."""

    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        max_sequence_length=max_sequence_length,
    )
    train_df, val_df, test_df = dataset.load_data()

    cleaned_sequences: Dict[int, List[int]] = {}
    for user_id, item_ids in dataset.get_user_sequences().items():
        seq = [int(item_id) for item_id in item_ids]
        if len(seq) >= 2:
            cleaned_sequences[int(user_id)] = seq

    return DatasetBundle(
        name=dataset_name.lower(),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        user_sequences=cleaned_sequences,
        num_users=int(dataset.num_users),
        num_items=int(dataset.num_items),
    )
