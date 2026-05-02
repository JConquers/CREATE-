"""Dataset loading helpers for CREATE-Pone."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

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


def _resolve_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in columns:
            return name
    return None


def _normalize_pone_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)

    user_col = _resolve_existing_column(
        columns,
        ["user_id", "userId", "uid", "user"],
    )
    item_col = _resolve_existing_column(
        columns,
        ["item_id", "movieId", "iid", "item"],
    )
    if user_col is None or item_col is None:
        raise KeyError(
            "Pone split must contain user/item columns. "
            f"Found columns: {columns}"
        )

    rating_col = _resolve_existing_column(columns, ["rating", "overall", "score"])
    timestamp_col = _resolve_existing_column(
        columns,
        ["timestamp", "time", "unixReviewTime", "timestemp", "ts"],
    )

    normalized = df[[user_col, item_col]].rename(
        columns={
            user_col: "user_id",
            item_col: "item_id",
        }
    )

    if rating_col is not None:
        normalized["rating"] = pd.to_numeric(df[rating_col], errors="coerce")
    else:
        normalized["rating"] = 1.0

    if timestamp_col is not None:
        normalized["timestamp"] = pd.to_numeric(df[timestamp_col], errors="coerce")
    else:
        normalized["timestamp"] = np.arange(len(normalized), dtype=np.int64)

    fallback_ts = pd.Series(np.arange(len(normalized), dtype=np.int64), index=normalized.index)
    normalized["timestamp"] = normalized["timestamp"].fillna(fallback_ts)

    normalized["user_id"] = pd.to_numeric(normalized["user_id"], errors="coerce")
    normalized["item_id"] = pd.to_numeric(normalized["item_id"], errors="coerce")
    normalized = normalized.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    normalized["user_id"] = normalized["user_id"].astype(int)
    normalized["item_id"] = normalized["item_id"].astype(int)
    normalized["timestamp"] = normalized["timestamp"].astype(np.int64)

    return normalized[["user_id", "item_id", "rating", "timestamp"]]


def _read_pone_split(path: Path) -> pd.DataFrame:
    try:
        raw_df = pd.read_csv(path, index_col=0)
    except Exception:
        raw_df = pd.read_csv(path)

    if raw_df.empty:
        raise ValueError(f"Pone split file is empty: {path}")

    return _normalize_pone_split_frame(raw_df)


def _resolve_pone_split_dir(
    data_dir: str,
    split_dir: str | None,
    version: int,
) -> Path:
    candidates: list[Path] = []
    if split_dir:
        candidates.append(Path(split_dir))

    data_path = Path(data_dir)
    candidates.extend(
        [
            data_path / "amazon-book",
            data_path,
            Path("./amazon-book"),
            Path("./Pone-GNN/amazon-book"),
            Path("/kaggle/input/amazon-book"),
            Path("/kaggle/working/CREATE-/Pone-GNN/amazon-book"),
        ]
    )

    for candidate in candidates:
        train_file = candidate / f"train_amazon{version}.dat"
        test_file = candidate / f"test_amazon{version}.dat"
        if train_file.exists() and test_file.exists():
            return candidate

    searched = [str(path) for path in candidates]
    raise FileNotFoundError(
        "Could not locate Pone Amazon split files. "
        f"Expected train_amazon{version}.dat and test_amazon{version}.dat. "
        f"Searched: {searched}"
    )


def _build_user_sequences(train_df: pd.DataFrame, max_sequence_length: int) -> Dict[int, List[int]]:
    ordered = train_df
    if "timestamp" in ordered.columns:
        ordered = ordered.sort_values(["user_id", "timestamp"])

    user_sequences: Dict[int, List[int]] = {}
    for user_id, group in ordered.groupby("user_id", sort=False):
        item_ids = [int(item_id) for item_id in group["item_id"].tolist()]
        if item_ids:
            user_sequences[int(user_id)] = item_ids[-max_sequence_length:]

    return user_sequences


def _load_pone_books_bundle(
    data_dir: str,
    max_sequence_length: int,
    version: int,
    split_dir: str | None,
) -> DatasetBundle:
    resolved_dir = _resolve_pone_split_dir(
        data_dir=data_dir,
        split_dir=split_dir,
        version=version,
    )
    train_path = resolved_dir / f"train_amazon{version}.dat"
    test_path = resolved_dir / f"test_amazon{version}.dat"

    train_df = _read_pone_split(train_path)
    test_df = _read_pone_split(test_path)

    all_users = sorted(set(train_df["user_id"]).union(set(test_df["user_id"])))
    all_items = sorted(set(train_df["item_id"]).union(set(test_df["item_id"])))

    user_map = {old_id: new_id for new_id, old_id in enumerate(all_users)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(all_items)}

    for frame in [train_df, test_df]:
        frame["user_id"] = frame["user_id"].map(user_map).astype(int)
        frame["item_id"] = frame["item_id"].map(item_map).astype(int)

    val_df = pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"])
    user_sequences = _build_user_sequences(
        train_df=train_df,
        max_sequence_length=max_sequence_length,
    )

    return DatasetBundle(
        name=f"books_pone_v{version}",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        user_sequences=user_sequences,
        num_users=len(all_users),
        num_items=len(all_items),
    )


def load_dataset_bundle(
    dataset_name: str,
    data_dir: str,
    max_sequence_length: int,
    books_protocol: str = "default",
    books_pone_version: int = 1,
    books_pone_split_dir: str | None = None,
) -> DatasetBundle:
    """Load Beauty/Books data using the existing dataset_loaders package."""

    if dataset_name.lower() == "books" and books_protocol.lower() == "pone":
        return _load_pone_books_bundle(
            data_dir=data_dir,
            max_sequence_length=max_sequence_length,
            version=books_pone_version,
            split_dir=books_pone_split_dir,
        )

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
