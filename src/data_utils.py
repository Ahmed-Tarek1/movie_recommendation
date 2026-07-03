"""
Shared data loading / preprocessing utilities.
Used by training notebooks and backend inference so field encodings
stay consistent across both.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Raw loaders ────────────────────────────────────────────────────────────────

def load_ratings(path: str) -> pd.DataFrame:
    """Load ratings.csv → userId, movieId, rating, timestamp."""
    return pd.read_csv(path)


def load_movies(path: str) -> pd.DataFrame:
    """Load movies.csv → movieId, title, genres (pipe-separated string)."""
    return pd.read_csv(path)


def load_tags(path: str) -> pd.DataFrame:
    """Load tags.csv → userId, movieId, tag, timestamp."""
    return pd.read_csv(path)


# ── Genre encoding ─────────────────────────────────────────────────────────────

def get_all_genres(movies: pd.DataFrame) -> list[str]:
    """Return sorted list of all genres, excluding '(no genres listed)'."""
    return sorted(
        g for g in movies.genres.str.split('|').explode().unique()
        if g != '(no genres listed)'
    )


def encode_genres(movies: pd.DataFrame, genres: list[str]) -> pd.DataFrame:
    """Add a binary column per genre to the movies DataFrame."""
    movies = movies.copy()
    for g in genres:
        movies[g] = movies.genres.str.contains(g, regex=False).astype(int)
    return movies


def merge_genre_flags(ratings: pd.DataFrame, movies: pd.DataFrame, genre_cols: list[str]) -> pd.DataFrame:
    """Left-join genre binary columns from movies into ratings."""
    return ratings.merge(movies[['movieId'] + genre_cols], on='movieId', how='left')


# ── ID mappings ────────────────────────────────────────────────────────────────

def build_id_mappings(df: pd.DataFrame, col: str) -> dict:
    """Map raw IDs → contiguous 0..N-1 indices for embedding tables."""
    unique_ids = sorted(df[col].unique())
    return {raw_id: idx for idx, raw_id in enumerate(unique_ids)}


def save_mappings(mappings: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in mappings.items()}, f, indent=2)


def load_mappings(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    # try to restore original key types (int if possible)
    try:
        return {int(k): v for k, v in raw.items()}
    except (ValueError, TypeError):
        return raw


# ── Train / test split ─────────────────────────────────────────────────────────

def train_test_split_by_time(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    test_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split — avoids leaking future interactions into train set.
    Sorts by timestamp globally, then cuts at (1 - test_frac).
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ── Item metadata helper ───────────────────────────────────────────────────────

def build_item_metadata(movies: pd.DataFrame, genre_cols: list[str]) -> pd.DataFrame:
    """
    Returns a clean item metadata DataFrame with:
      movieId, title, year (extracted from title), genres (pipe string), genre binary cols
    Used by backend /profile/item endpoint and dashboard item profile view.
    """
    meta = movies.copy()
    meta['year'] = meta['title'].str.extract(r'\((\d{4})\)$').astype('Int64')
    meta['title_clean'] = meta['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.strip()
    return meta[['movieId', 'title', 'title_clean', 'year', 'genres'] + genre_cols]


def get_user_history(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    user_id: int,
) -> pd.DataFrame:
    """Return a user's interaction history sorted by timestamp descending."""
    user_ratings = ratings[ratings.userId == user_id].copy()
    user_ratings = user_ratings.merge(movies[['movieId', 'title', 'genres']], on='movieId', how='left')
    return user_ratings.sort_values('timestamp', ascending=False).reset_index(drop=True)
