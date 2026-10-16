"""
Tests for live quotes and GET /api/v1/quotes/{ticker}. The upstream API is faked; no network.
"""

from datetime import datetime, timezone

import pytest

from api.config import settings
from api.services import quotes

RAW = {"bp": 99.98, "bs": 300, "ap": 100.02, "as": 100, "t": "2026-10-15T14:30:00.123456789Z"}
NOW = datetime(2026, 10, 15, 14, 30, 2, 123456, tzinfo=timezone.utc)  # a Thursday, market open


@pytest.fixture(autouse=True)
def clear_cache():
    quotes._cache.clear()
    yield


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "secret")


def test_measures():
    m = quotes.measures(99.98, 100.02, 300, 100)
    assert m["mid"] == pytest.approx(100.0)
    assert m["spread"] == pytest.approx(0.04)
    assert m["spread_bps"] == pytest.approx(4.0)
    assert m["imbalance"] == pytest.approx(0.5)  # more size on the bid
    assert m["microprice"] == pytest.approx(100.01)  # leans toward the thin ask


def test_measures_without_size():
    m = quotes.measures(10, 10.1, 0, 0)
    assert m["imbalance"] is None and m["microprice"] is None


def test_build_quote():
    q = quotes.build_quote("AAPL", RAW, now=NOW)
    assert q["available"] is True and q["market_open"] is True
    assert q["bid"] == 99.98 and q["ask_size"] == 100
    assert q["quoted_at"].startswith("2026-10-15T14:30:00.123456")
    assert q["age_seconds"] == pytest.approx(2.0)


@pytest.mark.parametrize("raw", [{}, {"bp": 0, "ap": 10}, {"bp": 10.5, "ap": 10.0}])
def test_build_quote_rejects_one_sided_or_crossed(raw):
    q = quotes.build_quote("AAPL", raw, now=NOW)
    assert q["available"] is False and "AAPL" in q["reason"]


def test_not_configured_reports_unavailable(client, monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "")
    data = client.get("/api/v1/quotes/aapl").json()
    assert data == {"ticker": "AAPL", "available": False, "reason": quotes.NOT_CONFIGURED}


def test_endpoint_caches_upstream(client, keys, monkeypatch):
    calls = []

    def fetch(t):
        calls.append(t)
        return RAW

    monkeypatch.setattr(quotes, "fetch_alpaca_quote", fetch)
    for _ in range(3):
        response = client.get("/api/v1/quotes/AAPL")
        assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.headers["cache-control"] == "public, max-age=10"
    assert calls == ["AAPL"]


def test_endpoint_upstream_failure_is_502(client, keys, monkeypatch):
    def boom(t):
        raise ConnectionError("alpaca down")

    monkeypatch.setattr(quotes, "fetch_alpaca_quote", boom)
    response = client.get("/api/v1/quotes/AAPL")
    assert response.status_code == 502
    assert "alpaca" not in response.json()["detail"].lower()


def test_endpoint_invalid_ticker_is_422(client):
    assert client.get("/api/v1/quotes/$$$").status_code == 422
