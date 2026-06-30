import sys
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root

from backend.models import fm_inference, similarity  # noqa: E402
from backend.models.fm_inference import FMRecommender  # noqa: E402
from backend.models.similarity import ItemSimilarity  # noqa: E402
from backend.routers import item, user  # noqa: E402

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


config = load_config()

app = FastAPI(title=config["api"]["title"], version=config["api"]["version"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_models():
    repo_root = Path(__file__).resolve().parents[1]

    fm_inference.recommender = FMRecommender(
        model_path=str(repo_root / config["model"]["fm_model_path"]),
        config_path=str(repo_root / config["model"]["fm_config_path"]),
        device=config["model"]["device"],
    )

    similarity.item_similarity = ItemSimilarity(
        item_embeddings_path=str(repo_root / config["model"]["item_embeddings_path"]),
        item_mapping=fm_inference.recommender.item_mapping,
        method=config["similarity"]["method"],
    )


app.include_router(user.router)
app.include_router(item.router)


@app.get("/health")
def health():
    return {"status": "ok"}
