"""
Shared data loading / preprocessing utilities.
Used by notebooks (training) and backend (inference) so field encodings
stay consistent across both.
"""
import json
from pathlib import Path

import pandas as pd


def load_ratings(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_movies(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_id_mappings(df: pd.DataFrame, col: str) -> dict[int, int]:
    """Map raw MovieLens IDs -> contiguous 0..N-1 indices for embedding tables."""
    unique_ids = sorted(df[col].unique())
    return {raw_id: idx for idx, raw_id in enumerate(unique_ids)}


def save_mappings(mappings: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mappings, f, indent=2)


def load_mappings(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def train_test_split_by_time(df: pd.DataFrame, timestamp_col: str = "timestamp", test_frac: float = 0.2):
    """Time-aware split — avoids leaking future interactions into train set."""
    df = df.sort_values(timestamp_col)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]
