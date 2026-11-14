"""
/api/v1/bot endpoints: the on/off switch, the token, start/stop/kill, and status. The runner's broker is faked.
"""

import pytest

from api.config import settings
from api.main import app
from api.services.bot import runner as bot
from api.services.paper_broker import LiveTradingRefused
from api.tests.test_bot_runner import FakeBroker, FakeStream

MA = {"strategy": "ma_crossover", "ma_crossover": {"tickers": ["AAPL"], "short_window": 5, "long_window": 20}}


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "BOT_ENABLED", True)
    monkeypatch.setattr(settings, "BOT_ADMIN_TOKEN", "")


@pytest.fixture
def fake_runner(test_db):
    broker = FakeBroker()
    runner = bot.BotRunner(
        broker_factory=lambda: broker,
        bars_loader=lambda *a: __import__("pandas").Series(dtype=float),
        stream_factory=FakeStream,
        session_factory=test_db,
        market_open=lambda now: False,  # the loop thread only logs "market closed"
    )
    app.dependency_overrides[bot.get_runner] = lambda: runner
    yield runner, broker
    runner.stop()


def test_info_is_open_and_has_no_secrets(client, monkeypatch):
    monkeypatch.setattr(settings, "BOT_ADMIN_TOKEN", "hunter2")
    data = client.get("/api/v1/bot/info").json()
    assert data["enabled"] is False and data["token_required"] is True and data["paper_url_ok"] is True
    assert "hunter2" not in str(data)


def test_everything_else_is_403_while_disabled(client):
    for method, path in [("get", "/status"), ("get", "/strategies"), ("post", "/start"), ("post", "/stop"), ("post", "/kill")]:
        response = getattr(client, method)(f"/api/v1/bot{path}")
        assert response.status_code == 403
        assert "BOT_ENABLED" in response.json()["detail"]


def test_token_is_required_when_set(client, enabled, fake_runner, monkeypatch):
    monkeypatch.setattr(settings, "BOT_ADMIN_TOKEN", "hunter2")
    assert client.get("/api/v1/bot/status").status_code == 401
    assert client.get("/api/v1/bot/status", headers={"X-Bot-Token": "wrong"}).status_code == 401
    assert client.get("/api/v1/bot/status", headers={"X-Bot-Token": "hunter2"}).status_code == 200


def test_start_status_stop(client, enabled, fake_runner):
    response = client.post("/api/v1/bot/start", json=MA)
    assert response.status_code == 200, response.text
    status = client.get("/api/v1/bot/status").json()
    assert status["running"] is True and status["strategy"] == "ma_crossover"
    assert status["account"]["equity"] == 100500.0
    assert client.post("/api/v1/bot/start", json=MA).status_code == 409
    assert client.post("/api/v1/bot/stop").json() == {"running": False, "was_running": True}


def test_start_validates_settings(client, enabled, fake_runner):
    bad = {"strategy": "ma_crossover", "ma_crossover": {"tickers": ["AAPL"], "short_window": 30, "long_window": 10}}
    assert client.post("/api/v1/bot/start", json=bad).status_code == 422
    assert client.post("/api/v1/bot/start", json={"strategy": "lead_lag"}).status_code == 422


def test_start_refuses_live_url(client, enabled, test_db):
    def refuse():
        raise LiveTradingRefused("ALPACA_TRADING_URL must be https://paper-api.alpaca.markets.")

    app.dependency_overrides[bot.get_runner] = lambda: bot.BotRunner(broker_factory=refuse, session_factory=test_db)
    response = client.post("/api/v1/bot/start", json=MA)
    assert response.status_code == 400 and "paper-api" in response.json()["detail"]


def test_start_refuses_delayed_polygon_stream(client, enabled, test_db, monkeypatch):
    monkeypatch.setattr(settings, "POLYGON_REALTIME", False)
    runner = bot.BotRunner(broker_factory=FakeBroker, session_factory=test_db)  # the real make_stream
    app.dependency_overrides[bot.get_runner] = lambda: runner
    body = {"strategy": "lead_lag", "lead_lag": {"leader": "NVDA", "followers": ["AMD"], "source": "polygon"}}
    response = client.post("/api/v1/bot/start", json=body)
    assert response.status_code == 400 and "delayed" in response.json()["detail"]


def test_kill(client, enabled, fake_runner):
    runner, broker = fake_runner
    client.post("/api/v1/bot/start", json=MA)
    data = client.post("/api/v1/bot/kill", json={"close_positions": True}).json()
    assert data["running"] is False and data["cancelled_orders"] == 2 and data["closed_positions"] == 1
    assert runner.risk.killed and broker.closed_all == 1


def test_strategies_catalog(client, enabled):
    names = [s["name"] for s in client.get("/api/v1/bot/strategies").json()["strategies"]]
    assert names == ["ma_crossover", "lead_lag"]


def test_cors_allows_the_token_header(client):
    response = client.options(
        "/api/v1/bot/status",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-bot-token",
        },
    )
    assert "x-bot-token" in response.headers.get("access-control-allow-headers", "").lower()
