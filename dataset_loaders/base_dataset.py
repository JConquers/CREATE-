"""
Base dataset class for CREATE++.
"""

from abc import ABC, abstractmethod
import torch
from pathlib import Path


class BaseDataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self, root="data"):
        self.root = Path(root)
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"

    @abstractmethod
    def download(self):
        """Download the dataset."""
        pass

    @abstractmethod
    def process(self):
        """Process the raw data."""
        pass

    def load(self):
        """Main entry point to load the dataset."""
        self.download()
        return self.process()

    def get_edge_index(self):
        """Get the interaction graph edge index."""
        data = self.process()
        return data['edge_index'], data.get('edge_weight', torch.ones(data['edge_index'].shape[1]))

    def get_stats(self):
        """Return dataset statistics."""
        data = self.process()
        return {
            'n_users': data['n_users'],
            'n_items': data['n_items'],
            'n_interactions': data['edge_index'].shape[1]
        }
