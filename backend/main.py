import sys
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from backend.models import fm_inference, similarity
from backend.models.fm_inference import FMRecommender
from backend.models.similarity import ItemSimilarity
from backend.routers import item as item_router
from backend.routers import user as user_router

CONFIG_PATH = Path(__file__).parent / "config.yaml"
REPO_ROOT   = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()

app = FastAPI(title=config["api"]["title"], version=config["api"]["version"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models_and_data():
    # --- data first: FMRecommender needs movies_df to build genre features ---
    ratings_df = pd.read_csv(REPO_ROOT / config["data"]["ratings_path"])
    movies_df  = pd.read_csv(REPO_ROOT / config["data"]["movies_path"])

    # --- models ---
    fm_inference.recommender = FMRecommender(
        model_path=str(REPO_ROOT / config["model"]["fm_model_path"]),
        config_path=str(REPO_ROOT / config["model"]["fm_config_path"]),
        movies_df=movies_df,
        device=config["model"]["device"],
    )
    similarity.item_similarity = ItemSimilarity(
        item_embeddings_path=str(REPO_ROOT / config["model"]["item_embeddings_path"]),
        item_mapping=fm_inference.recommender.item_mapping,
        method=config["similarity"]["method"],
    )

    user_router.init_data(ratings_df, movies_df)
    item_router.init_data(movies_df)


app.include_router(user_router.router)
app.include_router(item_router.router)


@app.get("/health")
def health():
    return {"status": "ok"}