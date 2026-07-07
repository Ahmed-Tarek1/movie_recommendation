"""
Item-item similarity — two methods selectable via config.yaml similarity.method:
  "embedding" : cosine similarity on DeepFM item latent vectors (default)
  "content"   : TF-IDF on genres + tags per movie (cold-start fallback)
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingSimilarity:
    """Cosine similarity on item embedding vectors from the trained DeepFM model."""

    def __init__(self, item_embeddings_path: str, item_mapping: dict):
        self.item_mapping = item_mapping
        self.reverse_item_mapping = {v: k for k, v in item_mapping.items()}
        self.embeddings = np.load(item_embeddings_path)  # (n_items, embed_dim)

    def similar_to(self, raw_item_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        if raw_item_id not in self.item_mapping:
            raise ValueError(f'Unknown item_id: {raw_item_id}')
        idx = self.item_mapping[raw_item_id]
        sims = cosine_similarity(self.embeddings[idx].reshape(1, -1), self.embeddings)[0]
        ranked = sorted(
            ((i, s) for i, s in enumerate(sims) if i != idx),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        return [(self.reverse_item_mapping[i], float(s)) for i, s in ranked]


class ContentSimilarity:
    """TF-IDF cosine similarity on genres + tags text document per movie."""

    def __init__(self, movies_path: str, tags_path: str, item_mapping: dict):
        movies = pd.read_csv(movies_path)
        tags   = pd.read_csv(tags_path)

        tag_docs = tags.groupby('movieId')['tag'].apply(
            lambda x: ' '.join(x.str.lower())
        ).reset_index()
        tag_docs.columns = ['movieId', 'tag_doc']

        movies = movies.merge(tag_docs, on='movieId', how='left')
        movies['doc'] = (
            movies['genres'].str.replace('|', ' ', regex=False)
                            .str.replace('(no genres listed)', '', regex=False)
            + ' ' + movies['tag_doc'].fillna('')
        ).str.strip()

        # restrict to items in item_mapping (training set)
        movies = movies[movies.movieId.isin(item_mapping.keys())].reset_index(drop=True)

        self.movie_ids = movies['movieId'].tolist()
        tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        self.tfidf_matrix = tfidf.fit_transform(movies['doc'])

    def similar_to(self, raw_item_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        if raw_item_id not in self.movie_ids:
            raise ValueError(f'Unknown item_id: {raw_item_id}')
        idx = self.movie_ids.index(raw_item_id)
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]
        ranked = sorted(
            ((i, s) for i, s in enumerate(sims) if i != idx),
            key=lambda x: x[1], reverse=True
        )[:top_n]
        return [(self.movie_ids[i], float(s)) for i, s in ranked]


class ItemSimilarity:
    """
    Unified similarity interface — delegates to EmbeddingSimilarity or ContentSimilarity
    based on config.yaml similarity.method.
    """

    def __init__(
        self,
        item_embeddings_path: str,
        item_mapping: dict,
        method: str = 'embedding',
        movies_path: str | None = None,
        tags_path:   str | None = None,
    ):
        self.method = method
        if method == 'content':
            if not movies_path or not tags_path:
                raise ValueError('movies_path and tags_path required for content similarity')
            self._backend = ContentSimilarity(movies_path, tags_path, item_mapping)
        else:
            self._backend = EmbeddingSimilarity(item_embeddings_path, item_mapping)

    def similar_to(self, raw_item_id: int, top_n: int = 10) -> list[tuple[int, float]]:
        return self._backend.similar_to(raw_item_id, top_n)


# module-level singleton populated by backend/main.py on startup
item_similarity: ItemSimilarity | None = None


def get_item_similarity() -> ItemSimilarity:
    if item_similarity is None:
        raise RuntimeError('ItemSimilarity not initialized — check startup in main.py')
    return item_similarity
