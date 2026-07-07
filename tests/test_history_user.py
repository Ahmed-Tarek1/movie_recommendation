"""
End-to-end tests for GET /history/user/{user_id}
"""


def test_history_happy_path(client, valid_user_id):
    resp = client.get(f"/history/user/{valid_user_id}")
    assert resp.status_code == 200
    body = resp.json()

    assert body["user_id"] == valid_user_id
    assert isinstance(body["history"], list)
    assert len(body["history"]) > 0

    for entry in body["history"]:
        assert isinstance(entry["item_id"], int)
        assert isinstance(entry["title"], str) and entry["title"]
        assert isinstance(entry["rating"], float)
        assert isinstance(entry["timestamp"], int)

    timestamps = [e["timestamp"] for e in body["history"]]
    assert timestamps == sorted(timestamps, reverse=True), "history must be most-recent-first"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_history_unknown_user_returns_404(client, unknown_user_id):
    resp = client.get(f"/history/user/{unknown_user_id}")
    assert resp.status_code == 404


def test_history_non_integer_user_id_returns_422(client):
    resp = client.get("/history/user/abc")
    assert resp.status_code == 422