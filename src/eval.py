"""
Evaluation metrics for the recommender system.
Used in training notebooks for model comparison.
"""
import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(relevant: set, recommended: list, k: int) -> float:
    """Fraction of top-k recommended items that are relevant."""
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / k if k > 0 else 0.0


def recall_at_k(relevant: set, recommended: list, k: int) -> float:
    """Fraction of relevant items that appear in top-k."""
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(relevant: set, recommended: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE and MAE together — used in training loop comparison table."""
    return {
        'RMSE': round(rmse(y_true, y_pred), 4),
        'MAE' : round(mae(y_true, y_pred), 4),
    }


def ranking_metrics_for_user(
    user_id: int,
    ratings_df: pd.DataFrame,
    recommended_item_ids: list[int],
    threshold: float = 4.0,
    k: int = 10,
) -> dict:
    """
    Compute precision@k, recall@k, NDCG@k for a single user.
    relevant = items the user rated >= threshold in the test set.
    """
    relevant = set(
        ratings_df.loc[
            (ratings_df.userId == user_id) & (ratings_df.rating >= threshold),
            'movieId'
        ]
    )
    return {
        'user_id'    : user_id,
        f'P@{k}'     : round(precision_at_k(relevant, recommended_item_ids, k), 4),
        f'R@{k}'     : round(recall_at_k(relevant, recommended_item_ids, k), 4),
        f'NDCG@{k}'  : round(ndcg_at_k(relevant, recommended_item_ids, k), 4),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a side-by-side comparison DataFrame.
    results: {'FM': {'RMSE': 0.87, 'MAE': 0.67, ...}, 'DeepFM': {...}}
    """
    return pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
