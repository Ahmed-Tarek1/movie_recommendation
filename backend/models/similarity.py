"""
Item-item similarity — embedding-based (from FM item vectors) or
content-based (genre/tag TF-IDF), selectable via config.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ItemSimilarity:
    def __init__(self, item_embeddings_path: str, item_mapping: dict, method: str = "embedding"):
        self.method = method
        self.item_mapping = item_mapping
        self.reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        self.embeddings = np.load(item_embeddings_path)  # (n_items, dim), index-aligned with item_mapping

    def similar_to(self, raw_item_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        if str(raw_item_id) not in self.item_mapping:
            raise ValueError(f"Unknown item_id: {raw_item_id}")

        idx = self.item_mapping[str(raw_item_id)]
        target_vec = self.embeddings[idx].reshape(1, -1)
        sims = cosine_similarity(target_vec, self.embeddings)[0]

        ranked = sorted(
            ((i, s) for i, s in enumerate(sims) if i != idx),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        return [(self.reverse_item_mapping[i], float(s)) for i, s in ranked]


# Module-level singleton, populated by main.py on startup
item_similarity: ItemSimilarity | None = None


def get_item_similarity() -> ItemSimilarity:
    if item_similarity is None:
        raise RuntimeError("ItemSimilarity not initialized — check startup in main.py")
    return item_similarity
