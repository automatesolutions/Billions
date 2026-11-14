"""
Lead-lag research: candidates, scan estimates, scans and storage, the daily job, and the /lead-lag endpoints.
A fake loader returns synthetic prices with a planted lead; no network.
"""

from datetime import date, datetime, timezone

import pytest

from api.config import settings
from api.main import app
from api.models import LeadLagResult, PerfMetric
from api.services import lead_lag_research as research
from api.services import polygon
from api.services.lead_lag import EngineParams
from api.tests.test_lead_lag_engine import planted

LEADER, FOLLOWER = planted(n_days=12)
SERIES = {"NVDA": LEADER, "AMD": FOLLOWER, "SPY": LEADER, "QQQ": FOLLOWER}
PARAMS = EngineParams(0.3, 60, 300)


def fake_loader(ticker, source, start, end, max_wait=None):
    if ticker not in SERIES:
        raise polygon.PolygonError(f"Polygon has no bars for {ticker}.")
    return SERIES[ticker]


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "k")
    monkeypatch.setattr(settings, "LEAD_LAG_SOURCE", "polygon")
    monkeypatch.setattr(settings, "POLYGON_SECOND_BARS", False)
    polygon.clear_cache()


@pytest.fixture
def scans(test_db):
    manager = research.ScanManager(session_factory=test_db, loader=fake_loader)
    app.dependency_overrides[research.get_manager] = lambda: manager
    return manager


# ---------------------------------------------------------------- candidates and estimates


def test_candidates_cover_peers_etfs_and_index_proxies(db_session):
    pairs = research.candidates(db_session)
    keys = {research.pair_key(p["a"], p["b"]) for p in pairs}
    assert "AMD|NVDA" in keys and "AAPL|QQQ" in keys and "QQQ|SPY" in keys
    assert len(keys) == len(pairs)  # no duplicates


def test_candidates_drop_nasdaq_names_missing_from_the_universe(db_session):
    db_session.add(PerfMetric(strategy="swing", symbol="NVDA", metric_x=0, metric_y=0, z_x=0, z_y=0, is_outlier=False))
    db_session.commit()
    keys = {research.pair_key(p["a"], p["b"]) for p in research.candidates(db_session)}
    assert "AMD|NVDA" not in keys  # AMD isn't in this (tiny) universe
    assert "QQQ|SPY" in keys and "BAC|JPM" in keys  # ETFs and NYSE names aren't filtered


def test_estimate_reflects_the_polygon_rate_limit():
    pairs = [{"a": "NVDA", "b": "AMD"}, {"a": "SPY", "b": "QQQ"}, {"a": "XOM", "b": "CVX"}]
    now = datetime(2026, 10, 15, 22, 0, tzinfo=timezone.utc)
    est = research.estimate(pairs, "polygon", 60, now)
    assert est["tickers"] == 6 and est["requests"] == 12  # 60 days of minute bars: 2 pages a ticker
    assert est["seconds"] >= 60  # 12 requests at 5 a minute need at least two more windows
    assert est["resolution"] == "minute" and est["end"] == "2026-10-15"


def test_second_bars_when_the_plan_allows(monkeypatch):
    monkeypatch.setattr(settings, "POLYGON_SECOND_BARS", True)
    assert research.resolution("polygon") == (1, "second")
    assert research.resolution("alpaca") == (60, "minute")


def test_check_source(monkeypatch):
    assert research.check_source("polygon") is None
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "")
    assert "POLYGON_API_KEY" in research.check_source("polygon")
    assert "Unknown" in research.check_source("yahoo")


def test_date_range_uses_completed_sessions():
    during = datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc)  # Thursday 11:00 ET
    start, end = research.date_range(5, during)
    assert end == date(2026, 10, 14) and start == date(2026, 10, 8)


# ---------------------------------------------------------------- scans and storage


def test_scan_stores_results_and_reports_errors(test_db, scans):
    pairs = [{"a": "AMD", "b": "NVDA"}, {"a": "NVDA", "b": "MISSING"}]
    job = scans.start(pairs, "polygon", 12, PARAMS, run_date=date(2026, 10, 15), background=False)
    status = scans.status()
    assert status["state"] == "done" and status["done_pairs"] == 2
    assert status["errors"] == ["MISSING: Polygon has no bars for MISSING."]
    assert job["estimate"]["pairs"] == 2
    with test_db() as db:
        row = db.query(LeadLagResult).one()
        assert (row.leader, row.follower, row.run_date) == ("NVDA", "AMD", date(2026, 10, 15))
        assert row.accuracy > 0.75 and row.payload["details"]["xcorr"]


def test_rescanning_the_same_day_updates_instead_of_duplicating(test_db, scans):
    for _ in range(2):
        scans.start([{"a": "AMD", "b": "NVDA"}], "polygon", 12, PARAMS, run_date=date(2026, 10, 15), background=False)
    with test_db() as db:
        assert db.query(LeadLagResult).count() == 1


def test_one_scan_at_a_time(scans):
    scans.job = {"state": "running"}
    with pytest.raises(research.ScanRunning):
        scans.start([{"a": "AMD", "b": "NVDA"}], "polygon", 12, PARAMS)


# ---------------------------------------------------------------- daily job


@pytest.mark.parametrize(
    "now, last, due",
    [
        (datetime(2026, 10, 15, 20, 10, tzinfo=timezone.utc), None, None),  # 16:10 ET, too soon after the close
        (datetime(2026, 10, 15, 20, 25, tzinfo=timezone.utc), None, date(2026, 10, 15)),
        (datetime(2026, 10, 15, 20, 25, tzinfo=timezone.utc), date(2026, 10, 15), None),  # already done today
        (datetime(2026, 10, 15, 15, 0, tzinfo=timezone.utc), None, None),  # market still open
        (datetime(2026, 10, 17, 20, 25, tzinfo=timezone.utc), None, None),  # Saturday
        (datetime(2026, 11, 27, 18, 30, tzinfo=timezone.utc), None, date(2026, 11, 27)),  # early close at 13:00 ET
    ],
)
def test_research_due(now, last, due):
    assert research.research_due(now, last) == due


def test_daily_job_runs_once_per_trading_day(test_db, scans, monkeypatch):
    monkeypatch.setattr(research, "candidates", lambda db: [{"a": "NVDA", "b": "AMD"}])
    monkeypatch.setattr(settings, "LEAD_LAG_DAYS", 12)
    job = research.DailyResearch(scans=scans, session_factory=test_db)
    after_close = datetime(2026, 10, 15, 20, 30, tzinfo=timezone.utc)
    first = job.run_if_due(after_close)
    assert first["trigger"] == "daily" and scans.status()["state"] == "done"
    assert job.run_if_due(after_close) is None
    with test_db() as db:
        row = db.query(LeadLagResult).one()
        assert row.trigger == "daily" and row.run_date == date(2026, 10, 15)
    # A fresh job object (API restart) reads the stored date and doesn't rerun.
    assert research.DailyResearch(scans=scans, session_factory=test_db).run_if_due(after_close) is None


def test_daily_job_skips_without_a_data_source(test_db, scans, monkeypatch, caplog):
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "")
    job = research.DailyResearch(scans=scans, session_factory=test_db)
    assert job.run_if_due(datetime(2026, 10, 15, 20, 30, tzinfo=timezone.utc)) is None
    assert "POLYGON_API_KEY" in caplog.text


# ---------------------------------------------------------------- endpoints


def test_analyze_endpoint(client, monkeypatch):
    monkeypatch.setattr(research, "load_closes", fake_loader)
    body = {"a": "amd", "b": "nvda", "threshold_pct": 0.3, "lookback_seconds": 60, "max_wait_seconds": 300, "days": 12}
    data = client.post("/api/v1/lead-lag/analyze", json=body).json()
    assert data["leader"] == "NVDA" and data["follower"] == "AMD" and data["lag_seconds"] == 120
    assert data["events"] > 100 and data["source"] == "polygon" and data["details"]["event_path"]


def test_analyze_validates_input(client):
    assert client.post("/api/v1/lead-lag/analyze", json={"a": "AMD", "b": "amd"}).status_code == 422
    assert client.post("/api/v1/lead-lag/analyze", json={"a": "$$", "b": "AMD"}).status_code == 422
    assert client.post("/api/v1/lead-lag/analyze", json={"a": "AMD", "b": "NVDA", "days": 2}).status_code == 422


def test_analyze_rate_limit_is_429_with_retry_after(client, monkeypatch):
    def limited(*a, **k):
        raise polygon.PolygonError("Polygon allows 5 requests a minute. Try again in 40 seconds.", retry_after=40)

    monkeypatch.setattr(research, "load_closes", limited)
    response = client.post("/api/v1/lead-lag/analyze", json={"a": "AMD", "b": "NVDA"})
    assert response.status_code == 429 and response.headers["retry-after"] == "40"


def test_analyze_without_a_key_says_what_to_set(client, monkeypatch):
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "")
    response = client.post("/api/v1/lead-lag/analyze", json={"a": "AMD", "b": "NVDA"})
    assert response.status_code == 400 and "POLYGON_API_KEY" in response.json()["detail"]


def test_watchlist_crud(client, monkeypatch):
    monkeypatch.setattr(settings, "LEAD_LAG_MAX_WATCHLIST", 1)
    assert client.post("/api/v1/lead-lag/watchlist", json={"a": "nvda", "b": "amd"}).json()["pairs"] == [
        {"a": "NVDA", "b": "AMD"}
    ]
    assert len(client.post("/api/v1/lead-lag/watchlist", json={"a": "AMD", "b": "NVDA"}).json()["pairs"]) == 1  # same pair
    assert client.post("/api/v1/lead-lag/watchlist", json={"a": "SPY", "b": "QQQ"}).status_code == 409
    assert client.delete("/api/v1/lead-lag/watchlist/AMD/NVDA").json()["pairs"] == []


def test_scan_results_and_history_endpoints(client, scans, monkeypatch):
    monkeypatch.setattr(research, "candidates", lambda db: [])
    monkeypatch.setattr(settings, "LEAD_LAG_DAYS", 12)
    assert client.post("/api/v1/lead-lag/scan", json={"watchlist": True, "candidates": False}).status_code == 422
    client.post("/api/v1/lead-lag/watchlist", json={"a": "NVDA", "b": "AMD"})
    est = client.post("/api/v1/lead-lag/scan/estimate", json={"watchlist": True, "candidates": False}).json()
    assert est["pairs"] == 1 and est["tickers"] == 2
    started = client.post("/api/v1/lead-lag/scan", json={"watchlist": True, "candidates": False})
    assert started.status_code == 200
    scans._thread.join(10)
    assert client.get("/api/v1/lead-lag/scan").json()["scan"]["state"] == "done"

    results = client.get("/api/v1/lead-lag/results").json()
    assert len(results["results"]) == 1 and "details" not in results["results"][0]
    assert results["results"][0]["leader"] == "NVDA"
    history = client.get("/api/v1/lead-lag/pairs/AMD/NVDA").json()
    assert len(history["history"]) == 1 and history["latest"]["details"]["lag_histogram"]


def test_results_empty_before_any_scan(client, scans):
    assert client.get("/api/v1/lead-lag/results").json() == {"run_date": None, "results": [], "scan": None}
