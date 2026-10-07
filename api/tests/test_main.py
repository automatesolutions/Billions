"""
Tests for app-level endpoints.
"""


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BILLIONS API"


def test_unknown_path_is_404(client):
    assert client.get("/invalid-endpoint").status_code == 404


def test_no_trading_routes_exist(client):
    paths = client.get("/openapi.json").json()["paths"].keys()
    forbidden = ("trade", "order", "portfolio", "hft", "alpaca", "user", "login", "auth")
    assert not [p for p in paths if any(word in p.lower() for word in forbidden)]
