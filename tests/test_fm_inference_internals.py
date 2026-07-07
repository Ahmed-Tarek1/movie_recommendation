"""
Regression tests for the exact bug that broke /recommend/user/{id}:
fm_inference.py's feature construction must stay in sync with how the
model was actually trained.

Verified against notebooks/03_train_fm_model.ipynb and
04_train_deepfm_model.ipynb: input fields are laid out as
[user_idx, item_idx, *genre_flags], with genre_flags one-hot encoded
in ALL_GENRES order (alphabetically sorted), matching field_dims =
[n_users, n_items] + [2] * n_genres.

These tests inspect the loaded FMRecommender directly (in-process, no
HTTP) so a training/inference mismatch is caught with a clear assertion
message immediately, rather than surfacing later as an opaque PyTorch
RuntimeError about mismatched tensor sizes.
"""
import torch

from backend.models import fm_inference


def test_field_dims_matches_user_item_genre_layout(client):
    recommender = fm_inference.get_recommender()
    field_dims = recommender.config["field_dims"]
    genre_cols = recommender.genre_cols

    n_users = len(recommender.user_mapping)
    n_items = len(recommender.item_mapping)

    assert field_dims[0] == n_users
    assert field_dims[1] == n_items
    assert field_dims[2:] == [2] * len(genre_cols), (
        "field_dims' genre section no longer matches genre_cols — check "
        "that the training notebook still encodes genres as one binary "
        "field per genre, in the same order as genre_cols in fm_config.json"
    )


def test_genre_matrix_shape_and_values(client):
    recommender = fm_inference.get_recommender()
    matrix = recommender.genre_matrix

    n_items = len(recommender.item_mapping)
    n_genres = len(recommender.genre_cols)

    assert matrix.shape == (n_items, n_genres)
    assert not matrix.dtype.is_floating_point, "genre matrix must be a long tensor of 0/1 flags"

    unique_values = set(matrix.unique().tolist())
    assert unique_values <= {0, 1}, "genre matrix must be strictly binary (one-hot per genre)"


def test_recommend_input_batch_width_matches_field_dims(client, valid_user_id):
    """
    The most direct regression check for the original bug: the tensor built
    for inference must have exactly len(field_dims) columns — the same
    width the model's embedding table was trained on.
    """
    recommender = fm_inference.get_recommender()
    expected_width = len(recommender.config["field_dims"])

    n_items = len(recommender.item_mapping)
    user_idx = recommender.user_mapping[str(valid_user_id)]

    user_col = torch.full((n_items, 1), user_idx, dtype=torch.long)
    item_col = torch.arange(n_items, dtype=torch.long).unsqueeze(1)
    batch = torch.cat([user_col, item_col, recommender.genre_matrix], dim=1)

    assert batch.shape[1] == expected_width, (
        f"inference batch has {batch.shape[1]} columns but the model expects "
        f"{expected_width} (field_dims length) — this is the exact shape "
        f"mismatch that previously broke /recommend/user/{{id}}"
    )