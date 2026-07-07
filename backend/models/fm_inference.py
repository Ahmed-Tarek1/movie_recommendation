"""
Loads the trained FM model + mappings and serves top-N predictions for a user.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for src/ import
from src.fm_model import DeepFM, FactorizationMachine  # noqa: E402


class FMRecommender:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        device: str = "cpu",
    ):
        with open(config_path) as f:
            self.config = json.load(f)

        self.device = device

        model_type = self.config.get("model_type", "fm")
        if model_type == "deepfm":
            self.model = DeepFM(
                field_dims=self.config["field_dims"],
                embed_dim=self.config["embed_dim"],
                mlp_dims=self.config.get("mlp_dims", [64, 32]),
                dropout=self.config.get("dropout", 0.4),
            )
        else:
            self.model = FactorizationMachine(
                field_dims=self.config["field_dims"],
                embed_dim=self.config["embed_dim"],
            )

        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.user_mapping: dict = self.config["user_mapping"]
        self.item_mapping: dict = self.config["item_mapping"]
        self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}

        self.genre_cols: list[str] = self.config["genre_cols"]
        self.genre_matrix = self._build_genre_matrix(movies_df)

        # precompute per-user seen item sets for filtering at inference time
        self.user_seen_items: dict[int, set[int]] = self._build_seen_items(ratings_df)

    def _build_genre_matrix(self, movies_df: pd.DataFrame) -> torch.Tensor:
        """
        Precompute a (n_items, n_genres) long tensor of 0/1 genre flags,
        row-aligned with item_mapping order. Built once at startup.
        Items not in movies_df get all-zero genre row.
        """
        n_items  = len(self.item_mapping)
        n_genres = len(self.genre_cols)
        genre_to_col = {g: i for i, g in enumerate(self.genre_cols)}
        matrix = torch.zeros((n_items, n_genres), dtype=torch.long)
        movies_by_id = movies_df.set_index("movieId")

        for raw_id_str, idx in self.item_mapping.items():
            raw_id = int(raw_id_str)
            if raw_id not in movies_by_id.index:
                continue
            genres_str = movies_by_id.loc[raw_id, "genres"]
            if pd.isna(genres_str):
                continue
            for genre in genres_str.split("|"):
                col = genre_to_col.get(genre)
                if col is not None:
                    matrix[idx, col] = 1

        return matrix

    def _build_seen_items(self, ratings_df: pd.DataFrame) -> dict[int, set[int]]:
        """
        Precompute {user_id -> set of movie_ids they have already rated}.
        Built once at startup so filtering per-request is a fast set lookup.
        """
        seen: dict[int, set[int]] = {}
        for user_id, movie_id in zip(ratings_df["userId"], ratings_df["movieId"]):
            seen.setdefault(int(user_id), set()).add(int(movie_id))
        return seen

    @torch.no_grad()
    def recommend_for_user(
        self,
        raw_user_id: int,
        top_n: int = 10,
        filter_seen: bool = True,
    ) -> list[tuple[int, float]]:
        """
        Returns list of (raw_item_id, score) sorted descending by score.

        filter_seen=True (default): excludes movies the user has already rated,
        so recommendations are genuinely new items for that user.
        """
        if str(raw_user_id) not in self.user_mapping:
            raise ValueError(f"Unknown user_id: {raw_user_id}")

        user_idx = self.user_mapping[str(raw_user_id)]
        n_items  = len(self.item_mapping)

        # build (n_items, 2 + n_genres) batch: [user_idx, item_idx, *genre_flags]
        user_col = torch.full((n_items, 1), user_idx, dtype=torch.long)
        item_col = torch.arange(n_items, dtype=torch.long).unsqueeze(1)
        batch    = torch.cat([user_col, item_col, self.genre_matrix], dim=1).to(self.device)

        scores = self.model(batch).cpu().numpy()

        seen = self.user_seen_items.get(raw_user_id, set()) if filter_seen else set()

        ranked = sorted(
            (
                (idx, float(score))
                for idx, score in enumerate(scores)
                if int(self.reverse_item_mapping[idx]) not in seen
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [(int(self.reverse_item_mapping[idx]), float(score)) for idx, score in ranked]


# Module-level singleton, populated by main.py on startup
recommender: FMRecommender | None = None


def get_recommender() -> FMRecommender:
    if recommender is None:
        raise RuntimeError("FMRecommender not initialized — check startup in main.py")
    return recommender