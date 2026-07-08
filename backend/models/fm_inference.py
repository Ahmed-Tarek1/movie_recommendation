"""
Loads the trained FM/DeepFM model + mappings and serves top-N predictions per user.
Features used: user_id, movie_id, 19 genre binary fields, 30 tag binary fields.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.fm_model import DeepFM, FactorizationMachine  # noqa: E402


class FMRecommender:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        movies_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        tags_df: pd.DataFrame,
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

        self.user_mapping: dict           = self.config["user_mapping"]
        self.item_mapping: dict           = self.config["item_mapping"]
        self.reverse_item_mapping         = {v: k for k, v in self.item_mapping.items()}
        self.feature_cols: list[str]      = self.config["feature_cols"]   # genres + tags in order

        self.feature_matrix = self._build_feature_matrix(movies_df, tags_df)
        self.user_seen_items = self._build_seen_items(ratings_df)

    def _build_feature_matrix(
        self, movies_df: pd.DataFrame, tags_df: pd.DataFrame
    ) -> torch.Tensor:
        """
        Precompute (n_items, n_features) long tensor of 0/1 flags for all
        genre and tag fields, row-aligned with item_mapping. Built once at startup.
        """
        # reconstruct per-movie tag sets from tags_df
        movie_tag_sets: dict[int, set[str]] = (
            tags_df.groupby("movieId")["tag"]
            .apply(lambda x: set(t.lower() for t in x))
            .to_dict()
        )

        n_items   = len(self.item_mapping)
        n_features = len(self.feature_cols)
        matrix    = torch.zeros((n_items, n_features), dtype=torch.long)

        movies_by_id = movies_df.set_index("movieId")
        col_index    = {col: i for i, col in enumerate(self.feature_cols)}

        for raw_id_str, idx in self.item_mapping.items():
            raw_id = int(raw_id_str)
            if raw_id not in movies_by_id.index:
                continue

            row = movies_by_id.loc[raw_id]

            # genre flags
            genres_str = row.get("genres", "")
            if pd.notna(genres_str):
                for genre in genres_str.split("|"):
                    col_name = genre  # genre cols are stored as plain genre names
                    if col_name in col_index:
                        matrix[idx, col_index[col_name]] = 1

            # tag flags  (stored as "tag_<tagname>" in feature_cols)
            movie_tags = movie_tag_sets.get(raw_id, set())
            for col_name, col_i in col_index.items():
                if col_name.startswith("tag_"):
                    tag_value = col_name[4:]   # strip "tag_" prefix
                    if tag_value in movie_tags:
                        matrix[idx, col_i] = 1

        return matrix

    def _build_seen_items(self, ratings_df: pd.DataFrame) -> dict[int, set[int]]:
        """Precompute {user_id -> set of rated movie_ids} for seen-item filtering."""
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
        Returns list of (raw_item_id, score) sorted descending.
        filter_seen=True excludes movies the user has already rated.
        """
        if str(raw_user_id) not in self.user_mapping:
            raise ValueError(f"Unknown user_id: {raw_user_id}")

        user_idx = self.user_mapping[str(raw_user_id)]
        n_items  = len(self.item_mapping)

        user_col = torch.full((n_items, 1), user_idx, dtype=torch.long)
        item_col = torch.arange(n_items, dtype=torch.long).unsqueeze(1)
        batch    = torch.cat([user_col, item_col, self.feature_matrix], dim=1).to(self.device)

        scores = self.model(batch).cpu().numpy()
        seen   = self.user_seen_items.get(raw_user_id, set()) if filter_seen else set()

        ranked = sorted(
            (
                (idx, float(scores[idx]))
                for idx in range(n_items)
                if int(self.reverse_item_mapping[idx]) not in seen
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [(int(self.reverse_item_mapping[idx]), float(score)) for idx, score in ranked]


recommender: FMRecommender | None = None


def get_recommender() -> FMRecommender:
    if recommender is None:
        raise RuntimeError("FMRecommender not initialized — check startup in main.py")
    return recommender
