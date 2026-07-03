import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.models.similarity import get_item_similarity
from backend.schemas import (
    ItemSimilarityResponse,
    ItemProfileResponse,
    SimilarItem,
)

router = APIRouter(prefix="", tags=["item"])

_movies_df: pd.DataFrame | None = None


def init_data(movies_df: pd.DataFrame):
    global _movies_df
    _movies_df = movies_df


def _get_movie_row(item_id: int):
    if _movies_df is None:
        return None
    rows = _movies_df[_movies_df.movieId == item_id]
    return rows.iloc[0] if len(rows) else None


# ── Similar items ──────────────────────────────────────────────────────────────

@router.get("/similar/item/{item_id}", response_model=ItemSimilarityResponse)
def similar_items(item_id: int, page: int = 1, page_size: int = 10):
    if page < 1 or page_size < 1:
        raise HTTPException(400, "page and page_size must be >= 1")

    sim = get_item_similarity()
    try:
        all_ranked = sim.similar_to(item_id, top_n=page * page_size)
    except ValueError as e:
        raise HTTPException(404, str(e))

    start = (page - 1) * page_size
    page_items = all_ranked[start:start + page_size]

    def get_title(iid):
        row = _get_movie_row(iid)
        return row['title'] if row is not None else f"Item {iid}"

    return ItemSimilarityResponse(
        item_id=item_id,
        page=page,
        page_size=page_size,
        total_available=len(all_ranked),
        items=[
            SimilarItem(item_id=iid, title=get_title(iid), similarity=round(score, 4))
            for iid, score in page_items
        ],
    )


# ── Item profile ───────────────────────────────────────────────────────────────

@router.get("/profile/item/{item_id}", response_model=ItemProfileResponse)
def item_profile(item_id: int):
    row = _get_movie_row(item_id)
    if row is None:
        raise HTTPException(404, f"item_id={item_id} not found")

    genres = (
        [g for g in row['genres'].split('|') if g != '(no genres listed)']
        if pd.notna(row.get('genres')) else []
    )
    return ItemProfileResponse(
        item_id=item_id,
        title=str(row['title']),
        genres=genres,
    )


# ── Items list (for dropdown) ──────────────────────────────────────────────────

@router.get("/items")
def list_items():
    if _movies_df is None:
        raise HTTPException(503, "Data not loaded yet")
    return {
        "items": [
            {"item_id": int(row.movieId), "title": str(row.title)}
            for row in _movies_df[['movieId', 'title']].itertuples()
        ]
    }
