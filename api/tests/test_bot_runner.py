"""
The bot runner: market-hours gate, risk checks, orders, fills, latency timings, and the kill switch.
A fake broker and simulated clock stand in for Alpaca; no network, no real threads for trading.
"""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from api.models import BotDecision, BotTrade
from api.services import polygon
from api.services.bot.runner import AlreadyRunning, BotConfig, BotRunner, polygon_closes, trade_dict
from api.services.bot.streams import Tick
from api.services.paper_broker import BrokerError, LiveTradingRefused

T0 = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)  # Thursday, 10:00 New York


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)
        return self.now


class FakeBroker:
    def __init__(self, positions=None, fill=True, fill_price=None):
        self.orders = []
        self.positions = positions or []
        self.fill = fill
        self.fill_price = fill_price
        self.cancelled_all = 0
        self.closed_all = 0
        self.fail_orders = False

    def get_account(self):
        return {"cash": "100000", "equity": "100500", "buying_power": "200000", "status": "ACTIVE"}

    def get_positions(self):
        return self.positions

    def get_orders(self, status="all", limit=50):
        return [
            {"id": o["id"], "symbol": o["symbol"], "side": o["side"], "qty": o["qty"], "status": "new"} for o in self.orders
        ]

    def place_market_order(self, symbol, side, qty, client_order_id=None, time_in_force="day"):
        if self.fail_orders:
            raise BrokerError("Alpaca refused the request (403): insufficient buying power", 403)
        order = {"id": f"00000000-0000-0000-0000-{len(self.orders):012d}", "symbol": symbol, "side": side, "qty": qty}
        self.orders.append(order)
        return order

    def get_order(self, order_id):
        if self.fill is True:
            return {"id": order_id, "status": "filled", "filled_avg_price": str(self.fill_price), "filled_at": self.filled_at}
        return {"id": order_id, "status": self.fill or "new"}

    def cancel_order(self, order_id):
        pass

    def cancel_all_orders(self):
        self.cancelled_all += 1
        return 2

    def close_all_positions(self, cancel_orders=True):
        self.closed_all += 1
        return 1


class FakeStream:
    def __init__(self, source, symbols, on_tick):
        self.source, self.symbols, self.on_tick = source, symbols, on_tick
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def status(self):
        return {"source": self.source, "connected": self.started}


@pytest.fixture
def clock():
    return Clock()


def make_runner(test_db, clock, broker=None, market_open=True, bars=None, **kw):
    broker = broker or FakeBroker()
    runner = BotRunner(
        broker_factory=lambda: broker,
        bars_loader=bars or (lambda symbol, params, now: pd.Series(dtype=float)),
        stream_factory=FakeStream,
        session_factory=test_db,
        now=clock,
        market_open=lambda now: market_open if not callable(market_open) else market_open(now),
        **kw,
    )
    return runner, broker


def decisions(test_db):
    with test_db() as db:
        return [(d.action, d.status, d.symbol, d.reason) for d in db.query(BotDecision).order_by(BotDecision.id)]


MA_CONFIG = {"strategy": "ma_crossover", "ma_crossover": {"tickers": ["AAPL"], "short_window": 2, "long_window": 4}}
LL_CONFIG = {
    "strategy": "lead_lag",
    "lead_lag": {"leader": "NVDA", "followers": ["AMD"], "threshold_pct": 0.3, "lookback_seconds": 5, "max_wait_seconds": 30},
    "risk": {"max_position_dollars": 1000, "max_orders_per_day": 5},
}


def series(values):
    return pd.Series(values, index=pd.date_range("2026-10-01", periods=len(values), tz="UTC"), dtype=float)


# ---------------------------------------------------------------- config and lifecycle


def test_config_needs_params_for_the_chosen_strategy():
    with pytest.raises(ValueError):
        BotConfig(strategy="lead_lag", ma_crossover=MA_CONFIG["ma_crossover"])


def test_live_url_stops_start_before_anything_runs(test_db, clock):
    def refuse():
        raise LiveTradingRefused("paper only")

    runner = BotRunner(broker_factory=refuse, session_factory=test_db, now=clock)
    with pytest.raises(LiveTradingRefused):
        runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    assert not runner.running and decisions(test_db) == []


def test_start_twice_is_refused_and_stop_logs(test_db, clock):
    runner, _ = make_runner(test_db, clock)
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    with pytest.raises(AlreadyRunning):
        runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    assert runner.stop() is True
    assert [d[0] for d in decisions(test_db)] == ["start", "stop"]


def test_thread_starts_and_stops(test_db, clock):
    runner, _ = make_runner(test_db, clock, market_open=False)
    runner.start(BotConfig(**LL_CONFIG))
    time.sleep(0.3)
    assert runner.running and runner.stream.started
    runner.stop()
    assert not runner.running and runner.stream.stopped


# ---------------------------------------------------------------- strategy 1 through the runner


def test_market_closed_places_nothing(test_db, clock):
    runner, broker = make_runner(test_db, clock, market_open=False, bars=lambda *a: series([10, 9, 8, 7, 6, 12]))
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.run_bars_once()
    runner.run_bars_once()
    assert broker.orders == []
    assert [d[3] for d in decisions(test_db)].count(
        "Market is closed. The bot waits for the next session and places no orders."
    ) == 1


def test_crossover_buys_sized_by_risk(test_db, clock):
    runner, broker = make_runner(test_db, clock, bars=lambda *a: series([10, 9, 8, 7, 6, 12]))
    runner.start(BotConfig(**MA_CONFIG, risk={"max_position_dollars": 100, "max_orders_per_day": 5}), run_thread=False)
    runner.run_bars_once()
    assert broker.orders == [{"id": broker.orders[0]["id"], "symbol": "AAPL", "side": "buy", "qty": 8}]
    placed = [d for d in decisions(test_db) if d[1] == "placed"]
    assert len(placed) == 1 and "crossed above" in placed[0][3] and "Sized to 8 shares" in placed[0][3]
    assert runner.risk.orders_today == 1


def test_crossover_sells_whole_position(test_db, clock):
    broker = FakeBroker(positions=[{"symbol": "AAPL", "qty": "5", "market_value": "50"}])
    runner, _ = make_runner(test_db, clock, broker=broker, bars=lambda *a: series([6, 7, 8, 9, 10, 4]))
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.run_bars_once()
    assert broker.orders[0]["side"] == "sell" and broker.orders[0]["qty"] == 5


def test_holds_are_logged_once_per_reason(test_db, clock):
    prices = {"values": [1, 2, 3, 4, 5, 6, 7]}
    runner, broker = make_runner(test_db, clock, bars=lambda *a: series(prices["values"]))
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.run_bars_once()
    prices["values"] = [1, 2, 3, 4, 5, 6, 8]  # numbers change, the reason doesn't
    runner.run_bars_once()
    holds = [d for d in decisions(test_db) if d[0] == "hold"]
    assert len(holds) == 1 and broker.orders == []


def test_daily_limit_blocks_and_is_logged(test_db, clock):
    runner, broker = make_runner(test_db, clock, bars=lambda *a: series([10, 9, 8, 7, 6, 12]))
    runner.start(BotConfig(**MA_CONFIG, risk={"max_position_dollars": 1000, "max_orders_per_day": 1}), run_thread=False)
    runner.risk.record_order()
    runner.run_bars_once()
    assert broker.orders == []
    assert any(d[1] == "blocked" and "Daily order limit" in d[3] for d in decisions(test_db))


def test_orders_today_survive_a_restart(test_db, clock):
    runner, _ = make_runner(test_db, clock, bars=lambda *a: series([10, 9, 8, 7, 6, 12]))
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.run_bars_once()
    runner.stop()
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    assert runner.risk.orders_today == 1


def test_polygon_error_is_a_logged_skip(test_db, clock):
    def no_bars(*a):
        raise polygon.PolygonError("Polygon allows 5 requests a minute. Try again in 30 seconds.")

    runner, broker = make_runner(test_db, clock, bars=no_bars)
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.run_bars_once()
    assert any(d[1] == "skipped" and "5 requests a minute" in d[3] for d in decisions(test_db))


def test_broker_order_failure_is_logged_not_raised(test_db, clock):
    broker = FakeBroker()
    broker.fail_orders = True
    runner, _ = make_runner(test_db, clock, broker=broker, bars=lambda *a: series([10, 9, 8, 7, 6, 12]))
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.run_bars_once()
    assert any(d[1] == "failed" and "insufficient buying power" in d[3] for d in decisions(test_db))
    assert runner.risk.orders_today == 0


def test_polygon_closes_drops_bars_in_progress(monkeypatch):
    idx = pd.DatetimeIndex(["2026-10-14 04:00", "2026-10-15 04:00"], tz="UTC")  # Polygon daily bars start at midnight ET
    monkeypatch.setattr(polygon, "get_daily_bars", lambda *a, **k: pd.DataFrame({"close": [1.0, 2.0]}, index=idx))
    params = BotConfig(**MA_CONFIG).params
    assert list(polygon_closes("AAPL", params, T0)) == [1.0]  # 10:00 ET: today's bar is partial


# ---------------------------------------------------------------- strategy 2 through the runner


def tick(symbol, price, at):
    return Tick(symbol, price, at, at)


def lead_lag_entry(runner, clock):
    runner.process(tick("AMD", 100.0, clock.now))
    runner.process(tick("NVDA", 200.0, clock.now))
    runner.process(tick("NVDA", 201.0, clock.advance(2)))  # +0.5% in 2 s


def test_lead_lag_trade_records_latency_and_slippage(test_db, clock):
    broker = FakeBroker(fill_price=100.08)
    broker.filled_at = (T0 + timedelta(seconds=2.8)).isoformat().replace("+00:00", "Z")
    runner, _ = make_runner(test_db, clock, broker=broker)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    lead_lag_entry(runner, clock)
    assert broker.orders[0]["symbol"] == "AMD" and broker.orders[0]["side"] == "buy" and broker.orders[0]["qty"] == 10

    runner.process(None, clock.advance(0.5))  # fill poll
    with test_db() as db:
        row = db.query(BotTrade).one()
        data = trade_dict(row)
    assert data["status"] == "open" and data["entry_price"] == 100.08
    assert data["leader_move_pct"] == pytest.approx(0.5)
    assert data["follower_price_at_signal"] == 100.0
    t = data["timings"]
    assert t["leader_to_signal_ms"] == 0.0  # the simulated clock didn't move between tick and signal
    assert t["leader_to_fill_ms"] == pytest.approx(800.0)
    assert t["slippage_pct"] == pytest.approx(0.08)
    assert any("Leader move to fill took 800 ms" in d[3] for d in decisions(test_db))


def test_lead_lag_take_profit_closes_the_trade(test_db, clock):
    broker = FakeBroker(fill_price=100.0)
    broker.filled_at = T0.isoformat()
    runner, _ = make_runner(test_db, clock, broker=broker)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    lead_lag_entry(runner, clock)
    runner.process(None, clock.advance(0.5))
    broker.fill_price = 100.4
    runner.process(tick("AMD", 100.4, clock.advance(1)))
    assert [o["side"] for o in broker.orders] == ["buy", "sell"]
    runner.process(None, clock.advance(1.1))
    with test_db() as db:
        row = db.query(BotTrade).one()
    assert row.status == "closed" and row.exit_reason.startswith("Take profit") and row.pnl_pct == pytest.approx(0.4)


def test_lead_lag_time_exit(test_db, clock):
    broker = FakeBroker(fill_price=100.0)
    broker.filled_at = (T0 + timedelta(seconds=2)).isoformat()
    runner, _ = make_runner(test_db, clock, broker=broker)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    lead_lag_entry(runner, clock)
    runner.process(None, clock.advance(0.5))
    runner.process(None, clock.advance(31))
    assert [o["side"] for o in broker.orders] == ["buy", "sell"]
    assert any(d[1] == "placed" and d[3].startswith("Time exit") for d in decisions(test_db))


def test_unfilled_entry_is_canceled_and_released(test_db, clock):
    broker = FakeBroker(fill="new")
    runner, _ = make_runner(test_db, clock, broker=broker, fill_timeout_seconds=5)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    lead_lag_entry(runner, clock)
    runner.process(None, clock.advance(6))
    assert "AMD" not in runner.strategy.trades
    with test_db() as db:
        assert db.query(BotTrade).one().status == "failed"


def test_lead_lag_ticks_ignored_while_market_closed(test_db, clock):
    runner, broker = make_runner(test_db, clock, market_open=False)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    lead_lag_entry(runner, clock)
    assert broker.orders == []


# ---------------------------------------------------------------- kill switch and status


def test_kill_switch_stops_cancels_and_blocks(test_db, clock):
    broker = FakeBroker(fill="new")
    runner, _ = make_runner(test_db, clock, broker=broker)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    lead_lag_entry(runner, clock)
    result = runner.kill(close_positions=True)
    assert result == {"cancelled_orders": 2, "closed_positions": 1, "message": result["message"]}
    assert broker.cancelled_all == 1 and broker.closed_all == 1
    assert not runner.running and runner.risk.killed
    assert not runner.risk.size_entry(10).allowed
    with test_db() as db:
        assert db.query(BotTrade).one().status == "killed"


def test_kill_without_closing_positions(test_db, clock):
    runner, broker = make_runner(test_db, clock)
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    runner.kill()
    assert broker.cancelled_all == 1 and broker.closed_all == 0


def test_status_shape(test_db, clock):
    runner, _ = make_runner(test_db, clock)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    status = runner.status()
    assert status["running"] is True and status["strategy"] == "lead_lag"
    assert status["account"]["cash"] == 100000.0
    assert status["risk"]["max_orders_per_day"] == 5
    assert status["decisions"][0]["action"] == "start"
    assert any("IEX" in c for c in status["caveats"])


def test_status_chart_shows_bars_crosses_and_the_order(test_db, clock):
    bars = series([10, 9, 8, 7, 6, 12])
    runner, _ = make_runner(test_db, clock, bars=lambda *a: bars)
    runner.start(BotConfig(**MA_CONFIG), run_thread=False)
    assert runner.status()["chart"] == []  # nothing loaded yet
    runner.run_bars_once()
    (chart,) = runner.status()["chart"]
    last = bars.index[-1].isoformat()
    assert chart["symbol"] == "AAPL" and chart["short_window"] == 2 and chart["long_window"] == 4
    assert len(chart["bars"]) == 6 and chart["bars"][-1]["t"] == last
    assert chart["crosses"] == [{"t": last, "direction": "up", "price": 12.0}]
    assert [(o["t"], o["side"], o["qty"]) for o in chart["orders"]] == [(last, "buy", 83.0)]


def test_status_chart_is_empty_for_lead_lag(test_db, clock):
    runner, _ = make_runner(test_db, clock)
    runner.start(BotConfig(**LL_CONFIG), run_thread=False)
    assert runner.status()["chart"] == []
