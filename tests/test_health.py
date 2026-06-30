"""
Basic smoke test. Run: pytest tests/
Note: requires trained model artifacts in models/ to pass startup —
expand with mocks once the team agrees on a fixture strategy.
"""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
