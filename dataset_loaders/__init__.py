"""Dataset loaders package for sequential recommendation models."""

from .base_dataset import BaseDataset, build_graph_edges
from .collators import SequenceDataset, SASRecCollator
from .books_dataset import AmazonBooksDataset
from .beauty_dataset import AmazonBeautyDataset

__all__ = [
    'BaseDataset',
    'SequenceDataset',
    'SASRecCollator',
    'build_graph_edges',
    'AmazonBooksDataset',
    'AmazonBeautyDataset',
]


def get_dataset(dataset_name: str, data_dir: str, max_sequence_length: int = 50):
    """Factory function to get dataset by name.

    Args:
        dataset_name: Name of the dataset ('beauty' or 'books')
        data_dir: Path to data directory
        max_sequence_length: Maximum sequence length for sequential models

    Returns:
        Dataset instance
    """
    datasets = {
        'beauty': AmazonBeautyDataset,
        'books': AmazonBooksDataset,
    }

    if dataset_name.lower() not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(datasets.keys())}")

    return datasets[dataset_name.lower()](data_dir, max_sequence_length)
