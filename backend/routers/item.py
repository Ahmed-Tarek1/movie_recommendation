from fastapi import APIRouter, HTTPException

from backend.models.similarity import get_item_similarity
from backend.schemas import ItemSimilarityResponse, SimilarItem

router = APIRouter(prefix="/similar/item", tags=["item"])

# TODO: replace with real titles lookup from data/processed/items.csv
ITEM_TITLES: dict[int, str] = {}


@router.get("/{item_id}", response_model=ItemSimilarityResponse)
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

    return ItemSimilarityResponse(
        item_id=item_id,
        page=page,
        page_size=page_size,
        total_available=len(all_ranked),
        items=[
            SimilarItem(item_id=iid, title=ITEM_TITLES.get(iid, f"Item {iid}"), similarity=score)
            for iid, score in page_items
        ],
    )
