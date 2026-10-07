"""
Tests for outlier endpoints.
"""

from datetime import date, datetime, timezone

from api.models import PerfMetric


def test_get_strategies(client):
    response = client.get("/api/v1/outliers/strategies")
    assert response.status_code == 200
    names = [s["strategy"] for s in response.json()["strategies"]]
    assert names == ["scalp", "swing", "longterm"]


def test_strategy_info(client):
    for strategy in ["scalp", "swing", "longterm"]:
        data = client.get(f"/api/v1/outliers/{strategy}/info").json()
        assert data["strategy"] == strategy
        assert data["x_days"] > data["y_days"]


def test_unknown_strategy_is_404(client):
    assert client.get("/api/v1/outliers/invalid/info").status_code == 404
    assert client.get("/api/v1/outliers/invalid").status_code == 404


def test_refresh_cannot_be_triggered_publicly(client):
    assert client.post("/api/v1/outliers/refresh").status_code in (404, 405)
    assert client.get("/api/v1/outliers/status").status_code == 200


def test_outliers_empty(client):
    data = client.get("/api/v1/outliers/scalp").json()
    assert data["outlier_count"] == 0
    assert data["as_of"] is None
    assert "state" in data["market"]


def test_outliers_ranked_with_reason(client, db_session):
    kw = dict(strategy="swing", price_date=date(2026, 10, 6), inserted=datetime(2026, 10, 6, 20, 30, tzinfo=timezone.utc))
    db_session.add_all(
        [
            PerfMetric(symbol="AAA", metric_x=40.0, metric_y=5.0, z_x=2.5, z_y=0.4, is_outlier=True, **kw),
            PerfMetric(symbol="BBB", metric_x=-30.0, metric_y=-20.0, z_x=-2.2, z_y=-3.1, is_outlier=True, **kw),
            PerfMetric(symbol="CCC", metric_x=2.0, metric_y=1.0, z_x=0.1, z_y=0.2, is_outlier=False, **kw),
        ]
    )
    db_session.commit()

    response = client.get("/api/v1/outliers/swing")
    assert response.headers["cache-control"].startswith("public")
    data = response.json()
    assert data["as_of"] == "2026-10-06"
    assert data["computed_at"].startswith("2026-10-06T20:30")
    assert data["universe_count"] == 3
    assert [o["symbol"] for o in data["outliers"]] == ["BBB", "AAA"]
    top = data["outliers"][0]
    assert top["rank"] == 1 and top["direction"] == "down"
    assert top["reason"] == "Down 30.0% over 3 months and down 20.0% over 1 month, 3.1σ from the group"
