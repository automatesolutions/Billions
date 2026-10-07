"""
Tests for outlier endpoints.
"""

from api.models import PerfMetric


def test_get_strategies(client):
    response = client.get("/api/v1/outliers/strategies")
    assert response.status_code == 200
    names = [s["strategy"] for s in response.json()["strategies"]]
    assert names == ["scalp", "swing", "longterm"]


def test_strategy_info(client):
    for strategy in ["scalp", "swing", "longterm"]:
        response = client.get(f"/api/v1/outliers/{strategy}/info")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == strategy
        assert data["lookback_x_days"] > data["lookback_y_days"]


def test_unknown_strategy_is_404(client):
    assert client.get("/api/v1/outliers/invalid/info").status_code == 404
    assert client.get("/api/v1/outliers/invalid").status_code == 404


def test_outliers_empty(client):
    response = client.get("/api/v1/outliers/scalp")
    assert response.status_code == 200
    assert response.json()["count"] == 0


def test_outliers_with_data(client, db_session):
    db_session.add_all(
        [
            PerfMetric(strategy="swing", symbol="AAA", metric_x=40.0, metric_y=12.0, z_x=2.5, z_y=1.1, is_outlier=True),
            PerfMetric(strategy="swing", symbol="BBB", metric_x=2.0, metric_y=1.0, z_x=0.1, z_y=0.2, is_outlier=False),
            PerfMetric(strategy="scalp", symbol="CCC", metric_x=1.0, metric_y=1.0, z_x=0.0, z_y=0.0, is_outlier=False),
        ]
    )
    db_session.commit()

    data = client.get("/api/v1/outliers/swing").json()
    assert data["count"] == 2
    by_symbol = {m["symbol"]: m for m in data["metrics"]}
    assert by_symbol["AAA"]["is_outlier"] is True
    assert by_symbol["BBB"]["is_outlier"] is False
