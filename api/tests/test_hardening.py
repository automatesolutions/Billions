"""
Rate limiting and error responses.
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.limits import limiter
from api.main import app
from api.services import analysis, prices


@pytest.fixture
def limited_client(test_db, monkeypatch):
    """A client with rate limiting switched on and a tiny analysis limit."""
    monkeypatch.setattr(limiter, "enabled", True)
    limiter.reset()
    monkeypatch.setattr(prices, "daily_closes", lambda t: pd.Series(dtype=float))
    with TestClient(app) as client:
        yield client
    limiter.reset()


def test_analysis_is_rate_limited(limited_client):
    codes = [limited_client.get("/api/v1/analysis/ZZZZ").status_code for _ in range(25)]
    assert codes[0] == 404
    assert 429 in codes
    blocked = limited_client.get("/api/v1/analysis/ZZZZ")
    assert blocked.status_code == 429
    assert blocked.json()["detail"].startswith("Too many requests")
    assert blocked.headers["retry-after"] == "60"


def test_health_is_never_rate_limited(limited_client):
    assert all(limited_client.get("/health").status_code == 200 for _ in range(200))


def test_unhandled_errors_return_plain_json(test_db, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(analysis, "get_analysis", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/outliers/strategies")  # sanity: unaffected route still works
        assert response.status_code == 200
        response = client.get("/api/v1/analysis/AAPL")
    # The analysis router turns unexpected failures into a 502 with a plain message.
    assert response.status_code == 502
    assert "secret" not in response.text


def test_outlier_read_model_is_cached_until_data_changes(client, db_session, monkeypatch):
    from api.models import PerfMetric
    from api.services import outliers as service

    calls = []
    original = service._build_outliers
    monkeypatch.setattr(service, "_build_outliers", lambda db, name: calls.append(name) or original(db, name))

    client.get("/api/v1/outliers/swing")
    client.get("/api/v1/outliers/swing")
    assert calls == ["swing"]

    db_session.add(PerfMetric(strategy="swing", symbol="NEW", metric_x=1, metric_y=1, z_x=0, z_y=0, is_outlier=False))
    db_session.commit()
    assert client.get("/api/v1/outliers/swing").json()["universe_count"] == 1
    assert calls == ["swing", "swing"]


def test_global_handler_hides_internal_errors(test_db):
    def explode():
        raise RuntimeError("secret internal detail")

    app.add_api_route("/__test_explode", explode)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/__test_explode")
    finally:
        app.router.routes.pop()
    assert response.status_code == 500
    assert response.json() == {"detail": "Something went wrong on our side. Try again later."}
    assert "secret" not in response.text
