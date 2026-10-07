"""
Tests for GET /api/v1/analysis/{ticker}. Prices are faked; no network.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from api.services import analysis, prices


def _closes(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, n))), index=idx)


@pytest.fixture(autouse=True)
def clear_cache():
    analysis._cache.clear()
    analysis._price_cache.clear()
    yield


@pytest.fixture
def fake_prices(monkeypatch):
    def install(fn):
        monkeypatch.setattr(prices, "daily_closes", fn)

    return install


def test_invalid_ticker_is_422(client):
    for bad in ["$$$", "1ABC", "TOO-LONG-TICKER", "A B"]:
        assert client.get(f"/api/v1/analysis/{bad}").status_code == 422


def test_unknown_ticker_is_404(client, fake_prices):
    fake_prices(lambda t: pd.Series(dtype=float))
    response = client.get("/api/v1/analysis/ZZZZ")
    assert response.status_code == 404
    assert "ZZZZ" in response.json()["detail"]


def test_upstream_failure_is_502(client, fake_prices):
    def boom(t):
        raise ConnectionError("yahoo down")

    fake_prices(boom)
    response = client.get("/api/v1/analysis/AAPL")
    assert response.status_code == 502
    assert "yahoo" not in response.json()["detail"].lower()  # no internals leak to the client


def test_short_history_returns_partial_payload(client, fake_prices):
    fake_prices(lambda t: _closes(60))
    data = client.get("/api/v1/analysis/new").json()  # lower case is normalised
    assert data["ticker"] == "NEW"
    assert data["status"] == "insufficient_history"
    assert data["history_days"] == 59 and data["needed_days"] == 250
    assert data["risk"]["days"] == 59
    assert len(data["history"]) == 60


def test_full_analysis_payload(client, fake_prices):
    calls = []

    def fetch(t):
        calls.append(t)
        return _closes(520, seed=3)

    fake_prices(fetch)
    response = client.get("/api/v1/analysis/AAPL")
    assert response.status_code == 200
    assert response.headers["cache-control"].startswith("public")
    data = response.json()
    assert data["status"] == "ok"
    for key in ("signal", "forecast", "edge", "risk", "models", "validation", "cost_check", "microstructure", "caveats"):
        assert key in data
    assert data["horizon_days"] == 1
    assert data["default_cost_bps"] == 10.0
    assert data["signal"]["forecast_for"] > data["as_of"]
    json.dumps(data, allow_nan=False)  # strict JSON: no NaN or Infinity

    # Second call is served from the cache (prices are fetched again only after the TTL).
    client.get("/api/v1/analysis/AAPL")
    assert calls == ["AAPL"]


def test_payload_never_contains_order_language(client, fake_prices):
    fake_prices(lambda t: _closes(520, seed=4))
    text = client.get("/api/v1/analysis/AAPL").text.lower()
    for pattern in (r"\bbuy", r"\bsell", r"\border(?!-book)", r"position size", r"leverage", r"\bentry\b", r"\bexit\b"):
        assert not re.search(pattern, text), pattern


def test_completed_sessions_drops_todays_partial_bar():
    from datetime import datetime

    from api.services.market_calendar import NEW_YORK

    close = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-10-06", "2026-10-07"]))
    during = datetime(2026, 10, 7, 11, 0, tzinfo=NEW_YORK)
    after = datetime(2026, 10, 7, 16, 30, tzinfo=NEW_YORK)
    assert list(prices.completed_sessions(close, during)) == [1.0]
    assert list(prices.completed_sessions(close, after)) == [1.0, 2.0]
