"""
End-to-end tests for GET /users and GET /items
(dropdown / listing endpoints used by the dashboard)
"""


def test_list_users(client):
    resp = client.get("/users")
    assert resp.status_code == 200
    body = resp.json()

    assert "user_ids" in body
    assert isinstance(body["user_ids"], list)
    assert len(body["user_ids"]) > 0
    assert all(isinstance(uid, int) for uid in body["user_ids"])
    assert body["user_ids"] == sorted(body["user_ids"])


def test_list_items(client):
    resp = client.get("/items")
    assert resp.status_code == 200
    body = resp.json()

    assert "items" in body
    assert isinstance(body["items"], list)
    assert len(body["items"]) > 0

    for entry in body["items"][:20]:
        assert isinstance(entry["item_id"], int)
        assert isinstance(entry["title"], str) and entry["title"]