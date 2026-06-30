from pydantic import BaseModel


class RecommendedItem(BaseModel):
    item_id: int
    title: str
    score: float


class UserRecommendationResponse(BaseModel):
    user_id: int
    page: int
    page_size: int
    total_available: int
    items: list[RecommendedItem]


class SimilarItem(BaseModel):
    item_id: int
    title: str
    similarity: float


class ItemSimilarityResponse(BaseModel):
    item_id: int
    page: int
    page_size: int
    total_available: int
    items: list[SimilarItem]


class UserHistoryEntry(BaseModel):
    item_id: int
    title: str
    rating: float
    timestamp: int


class UserHistoryResponse(BaseModel):
    user_id: int
    history: list[UserHistoryEntry]


class ItemProfileResponse(BaseModel):
    item_id: int
    title: str
    genres: list[str]
