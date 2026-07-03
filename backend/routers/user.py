import pandas as pd
from fastapi import APIRouter, HTTPException

from backend.models.fm_inference import get_recommender
from backend.schemas import (
    RecommendedItem,
    UserRecommendationResponse,
    UserHistoryEntry,
    UserHistoryResponse,
)

router = APIRouter(prefix="", tags=["user"])

# loaded once at startup by main.py
_ratings_df: pd.DataFrame | None = None
_movies_df:  pd.DataFrame | None = None


def init_data(ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
    global _ratings_df, _movies_df
    _ratings_df = ratings_df
    _movies_df  = movies_df


def _get_title(item_id: int) -> str:
    if _movies_df is None:
        return f"Item {item_id}"
    row = _movies_df[_movies_df.movieId == item_id]
    return row.iloc[0]['title'] if len(row) else f"Item {item_id}"


# ── Recommendations ────────────────────────────────────────────────────────────

@router.get("/recommend/user/{user_id}", response_model=UserRecommendationResponse)
def recommend_for_user(user_id: int, page: int = 1, page_size: int = 10):
    if page < 1 or page_size < 1:
        raise HTTPException(400, "page and page_size must be >= 1")

    recommender = get_recommender()
    try:
        all_ranked = recommender.recommend_for_user(user_id, top_n=page * page_size)
    except ValueError as e:
        raise HTTPException(404, str(e))

    start = (page - 1) * page_size
    page_items = all_ranked[start:start + page_size]

    return UserRecommendationResponse(
        user_id=user_id,
        page=page,
        page_size=page_size,
        total_available=len(all_ranked),
        items=[
            RecommendedItem(item_id=iid, title=_get_title(iid), score=round(score, 4))
            for iid, score in page_items
        ],
    )


# ── User history ───────────────────────────────────────────────────────────────

@router.get("/history/user/{user_id}", response_model=UserHistoryResponse)
def user_history(user_id: int):
    if _ratings_df is None or _movies_df is None:
        raise HTTPException(503, "Data not loaded yet")

    user_rows = _ratings_df[_ratings_df.userId == user_id]
    if user_rows.empty:
        raise HTTPException(404, f"No history found for user_id={user_id}")

    merged = user_rows.merge(_movies_df[['movieId', 'title']], on='movieId', how='left')
    merged = merged.sort_values('timestamp', ascending=False)

    return UserHistoryResponse(
        user_id=user_id,
        history=[
            UserHistoryEntry(
                item_id=int(row.movieId),
                title=str(row.title),
                rating=float(row.rating),
                timestamp=int(row.timestamp),
            )
            for row in merged.itertuples()
        ],
    )


# ── Users list (for dropdown) ──────────────────────────────────────────────────

@router.get("/users")
def list_users():
    if _ratings_df is None:
        raise HTTPException(503, "Data not loaded yet")
    return {"user_ids": sorted(_ratings_df.userId.unique().tolist())}
