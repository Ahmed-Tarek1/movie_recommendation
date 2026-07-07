"""
Evaluation metrics for Recommender Systems.

This module is split into two groups:

  Rating-prediction metrics
  ─────────────────────────
  rmse         Root Mean Squared Error
  mae          Mean Absolute Error

  Ranking / Top-N metrics  (all per-user, then averaged via mean_metric)
  ─────────────────────────────────────────────────────────────────────
  hit_rate_at_k     1 if ≥1 relevant item appears in the top-K list, else 0
  precision_at_k    fraction of top-K recommendations that are relevant
  recall_at_k       fraction of relevant items retrieved in top-K
  ndcg_at_k         Normalised Discounted Cumulative Gain (graded relevance)
  mrr               Mean Reciprocal Rank

  Utilities
  ─────────
  mean_metric       Average any per-user metric over a list of users

All ranking metrics treat a predicted item as relevant if it appears in the
user's *actual* collection (binary relevance), **except** ndcg_at_k which
accepts a {item: relevance_score} dict for graded relevance.
"""
from typing import Any, List, Set, Union, Dict
import numpy as np

try:
    import torch
except ImportError:
    torch = None


def _to_numpy(x: Any) -> np.ndarray:
    """Helper to convert array-like input to a numpy array, handling PyTorch tensors."""
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =====================================================================
# 1. Rating Prediction (Regression) Metrics
# =====================================================================

def rmse(y_true: Any, y_pred: Any) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    Args:
        y_true: Ground truth ratings.
        y_pred: Predicted ratings.
    """
    true_arr = _to_numpy(y_true)
    pred_arr = _to_numpy(y_pred)
    if true_arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))

def hit_rate_at_k(actual: Union[List[Any], Set[Any]], predicted: List[Any], k: int) -> int:
    """
    Calculate Hit Rate at K (HR@K) for a single user.

    HR@K is a binary metric: returns 1 if **at least one** item from the
    top-K recommendations appears in the user's ground-truth set, and 0
    otherwise.

    Args:
        actual:    Ground-truth items the user interacted with / liked.
                   Can be a list or a set of item identifiers.
        predicted: Ranked list of recommended item identifiers, ordered from
                   most to least relevant (model output).
        k:         Number of top recommendations to consider.  Must be ≥ 1.

    Returns:
        1 if any item in predicted[:k] is in *actual*, else 0.

    Raises:
        TypeError:  If *predicted* is not a list or *actual* is not a list/set.
        ValueError: If *k* is not a positive integer.

    Example:
        >>> hit_rate_at_k({1, 3, 5}, [2, 3, 7, 9], k=2)
        1   # item 3 is in the top-2 list
        >>> hit_rate_at_k({1, 3, 5}, [2, 4, 7, 9], k=2)
        0   # neither 2 nor 4 is in the ground-truth set
    """
    if not isinstance(predicted, list):
        raise TypeError(f"predicted must be a list, got {type(predicted).__name__!r}")
    if not isinstance(actual, (list, set)):
        raise TypeError(f"actual must be a list or set, got {type(actual).__name__!r}")
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")

    if not actual or not predicted:
        return 0

    actual_set = set(actual)
    for item in predicted[:k]:
        if item in actual_set:
            return 1
    return 0


def mae(y_true: Any, y_pred: Any) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    Args:
        y_true: Ground truth ratings.
        y_pred: Predicted ratings.
    """
    true_arr = _to_numpy(y_true)
    pred_arr = _to_numpy(y_pred)
    if true_arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(true_arr - pred_arr)))


# =====================================================================
# 2. Ranking-based (Top-N Recommendation) Metrics (Per-User)
# =====================================================================

def precision_at_k(actual: Union[List[Any], Set[Any]], predicted: List[Any], k: int) -> float:
    """
    Calculate Precision at K for a single user.

    Args:
        actual: Collection of ground truth items the user interacted with/liked.
        predicted: Ordered list of recommended items.
        k: The number of top recommendations to consider.
    """
    if not actual or not predicted or k <= 0:
        return 0.0

    predicted_k = predicted[:k]
    actual_set = set(actual)
    intersection = actual_set.intersection(predicted_k)
    return len(intersection) / k


def recall_at_k(actual: Union[List[Any], Set[Any]], predicted: List[Any], k: int) -> float:
    """
    Calculate Recall at K for a single user.

    Args:
        actual: Collection of ground truth items the user interacted with/liked.
        predicted: Ordered list of recommended items.
        k: The number of top recommendations to consider.
    """
    if not actual or not predicted or k <= 0:
        return 0.0

    predicted_k = predicted[:k]
    actual_set = set(actual)
    intersection = actual_set.intersection(predicted_k)
    return len(intersection) / len(actual)


def ndcg_at_k(actual: Dict[Any, float], predicted: List[Any], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at K for a single user,
    assuming dense (graded) relevance — e.g. ratings.

    Args:
        actual: Mapping of item -> relevance (e.g. {movie_id: rating}) for items
                this user has ground-truth relevance for.
        predicted: Ordered list of recommended item ids (model's ranking).
        k: The number of top recommendations to consider.
    """
    if not actual or not predicted or k <= 0:
        return 0.0

    predicted_k = predicted[:k]

    # Discount factors per rank position
    discounts = [1 / np.log2(idx + 1) for idx, _ in enumerate(predicted_k, 1)]

    # DCG: look up each predicted item's true relevance (0 if user has no rating for it)
    dcg = 0.0
    for item, disc in zip(predicted_k, discounts):
        rel = actual.get(item, 0.0)
        dcg += (2**rel - 1) * disc

    # IDCG: best possible ordering — sort the user's known relevances, take top-k
    ideal_relevances = sorted(actual.values(), reverse=True)[:k]

    idcg = 0.0
    for rel, disc in zip(ideal_relevances, discounts):
        idcg += (2**rel - 1) * disc

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr(actual: List[Any], predicted: List[Any], k: int = None) -> float:
    """
    Calculate Reciprocal Rank (RR) for a single user (the inverse rank of the
    first relevant recommended item).

    Args:
        actual: Collection of ground truth items the user interacted with/liked.
        predicted: Ordered list of recommended items.
        k: Optional cutoff — only the first k recommendations are considered.
           If None, the full predicted list is searched.
    """
    if not actual or not predicted:
        return 0.0

    predicted_k = predicted[:k] if k is not None else predicted
    actual_set = set(actual)
    for i, item in enumerate(predicted_k):
        if item in actual_set:
            return 1.0 / (i + 1)
    return 0.0


# =====================================================================
# 3. Batch/Mean Evaluation Helpers
# =====================================================================

def mean_metric(
    actuals: List[Union[List[Any], Set[Any], Dict[Any, float]]],
    predictions: List[List[Any]],
    metric_fn: Any,
    **kwargs
) -> float:
    """
    Compute the mean value of a ranking metric over all users.

    Args:
        actuals: A list where each element is the ground truth for a user
                 (set/list of items for precision/recall/mrr, or
                 {item: relevance} dict for ndcg_at_k).
        predictions: A list where each element is the ordered list of predictions for a user.
        metric_fn: The metric function (e.g. precision_at_k, recall_at_k, ndcg_at_k, mrr).
        **kwargs: Additional parameters passed to the metric function (e.g., k=10).
    """
    if not actuals or not predictions:
        return 0.0
    if len(actuals) != len(predictions):
        raise ValueError("The lengths of actuals and predictions must match.")

    scores = [metric_fn(act, pred, **kwargs) for act, pred in zip(actuals, predictions)]
    return float(np.mean(scores))

# =====================================================================
# 4. Catalog/System-Level Metrics (Coverage, Diversity, Novelty, Serendipity)
# =====================================================================

def item_coverage(predictions: List[List[Any]], catalog: Union[List[Any], Set[Any]]) -> float:
    """
    Calculate catalog (item) coverage: the fraction of the total item catalog
    that appears across all users' recommendation lists.

    Args:
        predictions: A list where each element is the ordered list of recommended
                      items for a user.
        catalog: The full set of items available in the system.
    """
    if not predictions or not catalog:
        return 0.0

    recommended_items = set()
    for pred in predictions:
        recommended_items.update(pred)

    return len(recommended_items) / len(set(catalog))

def diversity_at_k(predicted: List[Any], item_features: Dict[Any, np.ndarray], k: int) -> float:
    """
    Calculate intra-list diversity for a single user: average pairwise dissimilarity
    (1 - cosine similarity) between items in the top-k recommended list.

    Args:
        predicted: Ordered list of recommended items.
        item_features: Mapping of item -> feature vector (e.g. genre one-hot,
                        or learned embedding) used to measure similarity.
        k: The number of top recommendations to consider.
    """
    predicted_k = [item for item in predicted[:k] if item in item_features]
    if len(predicted_k) < 2:
        return 0.0

    vectors = np.array([item_features[item] for item in predicted_k])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # avoid division by zero
    normalized = vectors / norms

    sim_matrix = normalized @ normalized.T
    n = len(predicted_k)
    # Sum upper triangle (excluding diagonal) of similarity matrix, average, then invert
    upper_tri_sum = np.sum(np.triu(sim_matrix, k=1))
    num_pairs = n * (n - 1) / 2
    avg_similarity = upper_tri_sum / num_pairs

    return 1.0 - float(avg_similarity)

def novelty_at_k(predicted: List[Any], item_popularity: Dict[Any, int], total_interactions: int, k: int) -> float:
    """
    Calculate novelty for a single user: average self-information of the
    top-k recommended items, based on how rare they are in the training data.
    Higher = more novel (less popular/obvious) recommendations.

    Args:
        predicted: Ordered list of recommended items.
        item_popularity: Mapping of item -> number of interactions in training data.
        total_interactions: Total number of interactions across the training data,
                             used to normalize popularity into a probability.
        k: The number of top recommendations to consider.
    """
    predicted_k = predicted[:k]
    if not predicted_k or total_interactions <= 0:
        return 0.0

    self_information_scores = []
    for item in predicted_k:
        pop = item_popularity.get(item, 0)
        if pop <= 0:
            continue  # item unseen in training; undefined probability, skip rather than divide by zero
        prob = pop / total_interactions
        self_information_scores.append(-np.log2(prob))

    if not self_information_scores:
        return 0.0

    return float(np.mean(self_information_scores))

def serendipity_at_k(
    actual: Union[List[Any], Set[Any]],
    predicted: List[Any],
    expected_items: Union[List[Any], Set[Any]],
    k: int
) -> float:
    """
    Calculate serendipity for a single user: fraction of top-k recommended items
    that are both relevant (in actual) AND unexpected (not in expected_items,
    e.g. not what a popularity or content-based baseline would have suggested).

    Args:
        actual: Collection of ground truth items the user interacted with/liked.
        predicted: Ordered list of recommended items.
        expected_items: Items considered "obvious"/unsurprising for this user,
                         e.g. top-N from a popularity baseline, or items similar
                         to the user's historical interactions.
        k: The number of top recommendations to consider.
    """
    if not actual or not predicted or k <= 0:
        return 0.0

    predicted_k = predicted[:k]
    actual_set = set(actual)
    expected_set = set(expected_items)

    unexpected_and_relevant = [
        item for item in predicted_k
        if item in actual_set and item not in expected_set
    ]

    return len(unexpected_and_relevant) / k