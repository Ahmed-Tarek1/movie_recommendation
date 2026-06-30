from fastapi import APIRouter, HTTPException

from backend.models.fm_inference import get_recommender
from backend.schemas import RecommendedItem, UserRecommendationResponse

router = APIRouter(prefix="/recommend/user", tags=["user"])

# TODO: replace with real titles lookup from data/processed/items.csv
ITEM_TITLES: dict[int, str] = {}


@router.get("/{user_id}", response_model=UserRecommendationResponse)
def recommend_for_user(user_id: int, page: int = 1, page_size: int = 10):
    if page < 1 or page_size < 1:
        raise HTTPException(400, "page and page_size must be >= 1")

    recommender = get_recommender()
    try:
        # over-fetch enough to paginate; for small MovieLens this is cheap
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
            RecommendedItem(item_id=iid, title=ITEM_TITLES.get(iid, f"Item {iid}"), score=score)
            for iid, score in page_items
        ],
    )
