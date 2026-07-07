"""
End-to-end tests for GET /profile/item/{item_id}
"""


def test_item_profile_happy_path(client, valid_item_id):
    resp = client.get(f"/profile/item/{valid_item_id}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["item_id"] == valid_item_id
    assert isinstance(body["title"], str) and body["title"]
    assert isinstance(body["genres"], list)
    assert "(no genres listed)" not in body["genres"], "sentinel genre value should be filtered out"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_item_profile_unknown_item_returns_404(client, unknown_item_id):
    resp = client.get(f"/profile/item/{unknown_item_id}")
    assert resp.status_code == 404


def test_item_profile_non_integer_item_id_returns_422(client):
    resp = client.get("/profile/item/xyz")
    assert resp.status_code == 422