"""
End-to-end tests for GET /similar/item/{item_id}
"""


def test_similar_items_happy_path(client, valid_item_id):
    resp = client.get(f"/similar/item/{valid_item_id}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["item_id"] == valid_item_id
    assert body["page"] == 1
    assert body["page_size"] == 10
    assert len(body["items"]) <= 10

    for item in body["items"]:
        assert isinstance(item["item_id"], int)
        assert item["item_id"] != valid_item_id, "an item should never be similar to itself"
        assert isinstance(item["title"], str) and item["title"]
        assert -1.0 <= item["similarity"] <= 1.0

    sims = [item["similarity"] for item in body["items"]]
    assert sims == sorted(sims, reverse=True), "results must be ranked by similarity descending"


def test_similar_items_excludes_self_across_larger_page(client, valid_item_id):
    resp = client.get(f"/similar/item/{valid_item_id}", params={"page_size": 50})
    assert resp.status_code == 200
    ids = [i["item_id"] for i in resp.json()["items"]]
    assert valid_item_id not in ids


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_similar_items_unknown_item_returns_404(client, unknown_item_id):
    resp = client.get(f"/similar/item/{unknown_item_id}")
    assert resp.status_code == 404
    assert "Unknown item_id" in resp.json()["detail"]


def test_similar_items_page_out_of_range_returns_empty(client, valid_item_id):
    resp = client.get(f"/similar/item/{valid_item_id}", params={"page": 999999, "page_size": 10})
    assert resp.status_code == 200
    assert resp.json()["items"] == []


def test_similar_items_page_zero_is_rejected(client, valid_item_id):
    resp = client.get(f"/similar/item/{valid_item_id}", params={"page": 0})
    assert resp.status_code == 400


def test_similar_items_negative_page_size_is_rejected(client, valid_item_id):
    resp = client.get(f"/similar/item/{valid_item_id}", params={"page_size": -5})
    assert resp.status_code == 400


def test_similar_items_non_integer_item_id_returns_422(client):
    resp = client.get("/similar/item/not-an-int")
    assert resp.status_code == 422