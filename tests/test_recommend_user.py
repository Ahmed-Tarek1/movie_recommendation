"""
End-to-end tests for GET /recommend/user/{user_id}
"""


def test_recommend_happy_path(client, valid_user_id):
    resp = client.get(f"/recommend/user/{valid_user_id}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["user_id"] == valid_user_id
    assert body["page"] == 1
    assert body["page_size"] == 10
    assert body["total_available"] >= 0
    assert len(body["items"]) == min(10, body["total_available"])

    for item in body["items"]:
        assert isinstance(item["item_id"], int)
        assert isinstance(item["title"], str) and item["title"]
        assert isinstance(item["score"], float)

    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True), "items must be ranked by score descending"


def test_recommend_custom_page_size(client, valid_user_id):
    resp = client.get(f"/recommend/user/{valid_user_id}", params={"page": 1, "page_size": 3})
    assert resp.status_code == 200
    assert len(resp.json()["items"]) <= 3


def test_recommend_pagination_is_consistent(client, valid_user_id):
    """Page 2 should continue the ranking, not repeat page 1's items."""
    page1 = client.get(f"/recommend/user/{valid_user_id}", params={"page": 1, "page_size": 5}).json()
    page2 = client.get(f"/recommend/user/{valid_user_id}", params={"page": 2, "page_size": 5}).json()

    ids_page1 = {i["item_id"] for i in page1["items"]}
    ids_page2 = {i["item_id"] for i in page2["items"]}
    assert ids_page1.isdisjoint(ids_page2)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_recommend_unknown_user_returns_404(client, unknown_user_id):
    resp = client.get(f"/recommend/user/{unknown_user_id}")
    assert resp.status_code == 404
    assert "Unknown user_id" in resp.json()["detail"]


def test_recommend_page_out_of_range_returns_empty_items(client, valid_user_id):
    resp = client.get(f"/recommend/user/{valid_user_id}", params={"page": 999999, "page_size": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["total_available"] >= 0  # request still succeeds, just no items on this page


def test_recommend_page_zero_is_rejected(client, valid_user_id):
    resp = client.get(f"/recommend/user/{valid_user_id}", params={"page": 0})
    assert resp.status_code == 400


def test_recommend_negative_page_is_rejected(client, valid_user_id):
    resp = client.get(f"/recommend/user/{valid_user_id}", params={"page": -1})
    assert resp.status_code == 400


def test_recommend_page_size_zero_is_rejected(client, valid_user_id):
    resp = client.get(f"/recommend/user/{valid_user_id}", params={"page_size": 0})
    assert resp.status_code == 400


def test_recommend_non_integer_user_id_returns_422(client):
    resp = client.get("/recommend/user/not-an-int")
    assert resp.status_code == 422