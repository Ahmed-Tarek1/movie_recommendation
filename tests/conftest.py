"""
Shared fixtures for end-to-end endpoint tests.

These tests hit the real FastAPI app with real model artifacts, as configured
in backend/config.yaml (no mocking). The point is to exercise the actual
trained FM model, the similarity index, and the real ratings/movies data.

We don't hardcode user_id / item_id values because they depend on whatever
dataset/model is currently trained. Instead:

- valid_user_id / valid_item_id are discovered at runtime and cross-checked
  against the model's own mappings directly (in-process dict lookups),
  because the /users and /items list endpoints read from ratings_df/
  movies_df, which is NOT guaranteed to be the same population as the FM
  model's user_mapping / item_mapping (e.g. if the model was trained on a
  subset/split). If there's a mismatch, the fixture fails loudly instead of
  silently testing the wrong thing.

- unknown_user_id / unknown_item_id are ids guaranteed not to exist in either
  source, for the "not in training set" edge cases.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.models import fm_inference, similarity


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def valid_user_id(client):
    resp = client.get("/users")
    assert resp.status_code == 200
    user_ids = resp.json()["user_ids"]
    assert user_ids, "No users returned by /users — check ratings data is loaded"

    mapping = fm_inference.recommender.user_mapping
    for uid in user_ids:
        if str(uid) in mapping:
            return uid

    pytest.fail(
        "No user_id from /users is present in the FM model's user_mapping. "
        "ratings_df and the trained model appear to use different user populations."
    )


@pytest.fixture(scope="session")
def unknown_user_id(client):
    resp = client.get("/users")
    user_ids = resp.json()["user_ids"]
    return max(user_ids) + 10_000_000


@pytest.fixture(scope="session")
def valid_item_id(client):
    resp = client.get("/items")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert items, "No items returned by /items — check movies data is loaded"

    mapping = similarity.item_similarity._backend.item_mapping
    for entry in items:
        iid = entry["item_id"]
        if str(iid) in mapping or iid in mapping:
            return iid

    pytest.fail(
        "No item_id from /items is present in the similarity index's item_mapping. "
        "movies_df and the trained embeddings appear to use different item populations."
    )


@pytest.fixture(scope="session")
def unknown_item_id(client):
    resp = client.get("/items")
    items = resp.json()["items"]
    max_id = max(entry["item_id"] for entry in items)
    return max_id + 10_000_000
