"""
CREATE-Pone: Cross-Representation Alignment with Signed Graph Encoder

A variant of CREATE++ that combines:
- CREATE: Cross-representation knowledge transfer with alignment
- Pone-GNN: Signed graph learning with dual-branch message passing
"""

from .models import SignedGraphEncoder, SequentialEncoder, CREATEPone
from .datasets import BeautyDataset, BooksDataset

__version__ = "1.0.0"
__all__ = [
    "SignedGraphEncoder",
    "SequentialEncoder",
    "CREATEPone",
    "BeautyDataset",
    "BooksDataset",
]
