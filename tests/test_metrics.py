import numpy as np
import pytest
from src.metrics.eval_metrics import (
    rmse, mae,
    hit_rate_at_k,
    precision_at_k, recall_at_k, ndcg_at_k, mrr,
    mean_metric,
)

def test_rating_metrics():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 1.9, 3.2]
    
    assert abs(rmse(y_true, y_pred) - 0.141421) < 1e-4
    assert abs(mae(y_true, y_pred) - 0.133333) < 1e-4

def test_ranking_metrics_single_user():
    actual = {1, 2, 3}
    predicted = [2, 4, 1, 5]
    
    # Precision@3 = 2/3 (2 and 1 are in actual)
    assert precision_at_k(actual, predicted, k=3) == 2 / 3
    # Recall@3 = 2/3 (2 and 1 are in actual)
    assert recall_at_k(actual, predicted, k=3) == 2 / 3
    
    # NDCG@3:
    # dcg@3 = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.0 + 0.0 + 0.5 = 1.5
    # idcg@3 = 1/log2(2) + 1/log2(3) + 1/log2(4) = 1.0 + 0.6309297 + 0.5 = 2.1309297
    # ndcg@3 = 1.5 / 2.1309297 = 0.703918
    assert abs(ndcg_at_k(actual, predicted, k=3) - 0.7039180) < 1e-6
    
    # MRR: first relevant is at index 0 (item 2), so MRR = 1/1 = 1.0
    assert mrr(actual, predicted) == 1.0

def test_mean_metric():
    actuals = [{1, 2}, {3, 4}]
    predictions = [[2, 3], [4, 5]]
    
    # User 1: actual={1, 2}, pred=[2, 3], K=2. Precision@2 = 1/2 = 0.5
    # User 2: actual={3, 4}, pred=[4, 5], K=2. Precision@2 = 1/2 = 0.5
    assert mean_metric(actuals, predictions, precision_at_k, k=2) == 0.5


class TestHitRateAtK:
    """Validate hit_rate_at_k correctness and guard clauses."""

    def test_hit_when_relevant_item_in_top_k(self):
        """A relevant item present within the top-K window  -> 1."""
        assert hit_rate_at_k({1, 3, 5}, [2, 3, 7, 9], k=2) == 1

    def test_miss_when_relevant_item_beyond_top_k(self):
        """A relevant item exists in the list but OUTSIDE top-K window -> 0."""
        # item 3 is at index 2 (0-based), so top-2 = [2, 7] -> no hit
        assert hit_rate_at_k({3}, [2, 7, 3, 9], k=2) == 0

    def test_full_miss_no_relevant_items_in_list(self):
        """None of the predicted items are relevant -> 0."""
        assert hit_rate_at_k({1, 3, 5}, [2, 4, 6, 8], k=4) == 0

    def test_hit_on_first_position(self):
        """Relevant item is at position 0 -> 1 for any k >= 1."""
        assert hit_rate_at_k({7}, [7, 1, 2, 3], k=1) == 1

    def test_empty_actual_returns_zero(self):
        """Empty ground-truth set -> no possible hit -> 0."""
        assert hit_rate_at_k(set(), [1, 2, 3], k=3) == 0

    def test_empty_predicted_returns_zero(self):
        """Empty predicted list -> nothing to check -> 0."""
        assert hit_rate_at_k({1, 2}, [], k=3) == 0

    def test_invalid_k_zero_raises(self):
        """k=0 is not a valid cutoff -> ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            hit_rate_at_k({1}, [1, 2], k=0)

    def test_invalid_k_negative_raises(self):
        """Negative k -> ValueError."""
        with pytest.raises(ValueError, match="k must be a positive integer"):
            hit_rate_at_k({1}, [1, 2], k=-5)

    def test_invalid_predicted_type_raises(self):
        """Passing a tuple instead of a list -> TypeError."""
        with pytest.raises(TypeError, match="predicted must be a list"):
            hit_rate_at_k({1}, (1, 2, 3), k=2)   # tuple, not list

    def test_invalid_actual_type_raises(self):
        """Passing a dict as actual -> TypeError."""
        with pytest.raises(TypeError, match="actual must be a list or set"):
            hit_rate_at_k({1: 5.0}, [1, 2], k=2)   # dict, not list/set

    def test_k_larger_than_list_clips_gracefully(self):
        """k > len(predicted) should not raise — just check all predictions."""
        # top-10 of a 3-item list: item 1 is relevant -> hit
        assert hit_rate_at_k({1}, [2, 3, 1], k=10) == 1
