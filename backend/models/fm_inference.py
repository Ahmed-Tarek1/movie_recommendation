"""
Loads the trained FM model + mappings and serves top-N predictions for a user.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))  # repo root, for src/ import
from src.fm_model import DeepFM, FactorizationMachine  # noqa: E402


class FMRecommender:
    def __init__(self, model_path: str, config_path: str, device: str = "cpu"):
        with open(config_path) as f:
            self.config = json.load(f)

        self.device = device

        # config["model_type"]: "fm" or "deepfm" — set this when saving fm_config.json
        model_type = self.config.get("model_type", "fm")
        if model_type == "deepfm":
            self.model = DeepFM(
                field_dims=self.config["field_dims"],
                embed_dim=self.config["embed_dim"],
                mlp_dims=self.config.get("mlp_dims", [128, 64, 32]),
                dropout=self.config.get("dropout", 0.2),
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

    @torch.no_grad()
    def recommend_for_user(self, raw_user_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        """Returns list of (raw_item_id, score) sorted descending by score."""
        if str(raw_user_id) not in self.user_mapping:
            raise ValueError(f"Unknown user_id: {raw_user_id}")

        user_idx = self.user_mapping[str(raw_user_id)]
        n_items = len(self.item_mapping)

        # build (n_items, 2) batch of [user_idx, item_idx] pairs
        user_col = torch.full((n_items,), user_idx, dtype=torch.long)
        item_col = torch.arange(n_items, dtype=torch.long)
        batch = torch.stack([user_col, item_col], dim=1).to(self.device)

        scores = self.model(batch).cpu().numpy()
        ranked = sorted(
            zip(range(n_items), scores), key=lambda x: x[1], reverse=True
        )[:top_n]

        return [(self.reverse_item_mapping[idx], float(score)) for idx, score in ranked]


# Module-level singleton, populated by main.py on startup
recommender: FMRecommender | None = None


def get_recommender() -> FMRecommender:
    if recommender is None:
        raise RuntimeError("FMRecommender not initialized — check startup in main.py")
    return recommender
