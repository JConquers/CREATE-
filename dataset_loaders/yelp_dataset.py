"""
Yelp dataset loader for CREATE++.
Uses PyTorch Geometric's Yelp dataset.
"""

import os
import torch
from torch_geometric.datasets import Yelp
from pathlib import Path


class YelpDataset:
    """Yelp dataset from PyTorch Geometric."""

    def __init__(self, root="data"):
        self.root = Path(root)
        self.processed_file = self.root / "yelp_data.pt"

    def load(self):
        """Load the Yelp dataset and process it."""
        if self.processed_file.exists():
            print(f"Loading processed data from {self.processed_file}")
            return torch.load(self.processed_file)

        print("Downloading Yelp dataset from PyTorch Geometric...")
        dataset = Yelp(root=self.root)

        data = dataset[0]

        n_users = int(data['user'].x.shape[0])
        n_items = int(data['item'].x.shape[0])

        edge_index = data['edge_index']
        edge_weight = torch.ones(edge_index.shape[1])

        train_user = []
        train_item = []
        test_user = []
        test_item = []

        user_items = {}
        for i in range(edge_index.shape[1]):
            u = edge_index[0, i].item()
            it = edge_index[1, i].item()
            if u not in user_items:
                user_items[u] = []
            user_items[u].append(it)

        for u, items in user_items.items():
            if len(items) >= 2:
                test_item.append(items[-1])
                test_user.append(u)
                for it in items[:-1]:
                    train_item.append(it)
                    train_user.append(u)
            elif len(items) == 1:
                test_item.append(items[0])
                test_user.append(u)

        result = {
            'edge_index': edge_index,
            'edge_weight': edge_weight,
            'n_users': n_users,
            'n_items': n_items,
            'train_user': torch.tensor(train_user, dtype=torch.long),
            'train_item': torch.tensor(train_item, dtype=torch.long),
            'test_user': torch.tensor(test_user, dtype=torch.long),
            'test_item': torch.tensor(test_item, dtype=torch.long),
        }

        torch.save(result, self.processed_file)
        print(f"Saved processed data to {self.processed_file}")
        return result

    def get_edge_index(self):
        """Get the interaction graph edge index."""
        data = self.load()
        return data['edge_index'], data.get('edge_weight', torch.ones(data['edge_index'].shape[1]))

    def get_stats(self):
        """Return dataset statistics."""
        data = self.load()
        return {
            'n_users': data['n_users'],
            'n_items': data['n_items'],
            'n_interactions': data['edge_index'].shape[1]
        }


if __name__ == "__main__":
    dataset = YelpDataset(root="data/Yelp")
    stats = dataset.load()
    print(f"Dataset stats: {stats}")
