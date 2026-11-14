"""
Lead-lag strategy (entries, skips, exits, cooldown) on simulated price streams, and stream parsing.
No sockets are opened: a fake WebSocket replays messages.
"""

import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from api.config import settings
from api.services.bot import streams
from api.services.bot.lead_lag_strategy import LeadLagParams, LeadLagStrategy, RollingPrices
from api.services.bot.streams import Tick

T0 = datetime(2026, 10, 15, 14, 0, 0, tzinfo=timezone.utc)


def at(seconds):
    return T0 + timedelta(seconds=seconds)


def tick(symbol, price, seconds):
    return Tick(symbol, price, at(seconds), at(seconds))


class SimClock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


def make(**overrides):
    params = LeadLagParams(
        leader="NVDA",
        followers=["AMD"],
        threshold_pct=0.3,
        lookback_seconds=5,
        max_wait_seconds=30,
        take_profit_pct=0.3,
        stop_loss_pct=0.2,
        cooldown_seconds=60,
        **overrides,
    )
    return LeadLagStrategy(params, clock=SimClock())


def feed(strategy, ticks):
    signals = []
    for t in ticks:
        signals.extend(strategy.on_tick(t))
    return signals


def leader_jump(start_s=0.0, pct=0.5, follower_price=100.0):
    """AMD sits still while NVDA jumps `pct` within 3 seconds."""
    return [
        tick("AMD", follower_price, start_s),
        tick("NVDA", 200.0, start_s),
        tick("NVDA", 200.0 * (1 + pct / 200), start_s + 1.5),
        tick("NVDA", 200.0 * (1 + pct / 100), start_s + 3),
    ]


# ---------------------------------------------------------------- params and windows


def test_params_validate():
    with pytest.raises(ValidationError):
        LeadLagParams(leader="NVDA", followers=["NVDA"])
    with pytest.raises(ValidationError):
        LeadLagParams(leader="NVDA", followers=[])
    with pytest.raises(ValidationError):
        LeadLagParams(leader="NVDA", followers=["AMD"], threshold_pct=0)
    assert LeadLagParams(leader="nvda", followers=["amd", "AMD"]).followers == ["AMD"]


def test_rolling_window_measures_move_over_lookback_only():
    w = RollingPrices(keep_seconds=60)
    w.add(at(0), 100)
    w.add(at(10), 101)
    w.add(at(12), 102)
    move, started = w.move_pct(5)
    assert move == pytest.approx((102 / 101 - 1) * 100)
    assert started == at(10)
    w.add(at(100), 103)  # older items fall out of the window
    assert w.move_pct(5) == (0.0, at(100))


# ---------------------------------------------------------------- entries


def test_leader_jump_buys_the_still_follower():
    s = make()
    signals = feed(s, leader_jump())
    assert len(signals) == 1
    sig = signals[0]
    assert (sig.action, sig.symbol, sig.kind) == ("buy", "AMD", "entry")
    assert sig.details["leader_move_pct"] == pytest.approx(0.5)
    assert sig.details["follower_price"] == 100.0
    assert "NVDA moved +0.50% in 3.0s" in sig.reason
    assert s.trades["AMD"].leader_move_at == at(3)  # the tick that crossed the threshold


def test_small_leader_move_does_nothing():
    assert feed(make(), leader_jump(pct=0.2)) == []


def test_follower_that_already_moved_is_skipped():
    s = make()
    ticks = [tick("AMD", 100.0, 0), tick("NVDA", 200, 0), tick("AMD", 100.5, 1), tick("NVDA", 201, 3)]
    (sig,) = feed(s, ticks)
    assert sig.action == "skip" and "already moved +0.50%" in sig.reason
    assert "AMD" not in s.trades


def test_down_move_skips_when_shorting_is_off():
    (sig,) = feed(make(), leader_jump(pct=-0.5))
    assert sig.action == "skip" and "Shorting is off" in sig.reason


def test_down_move_shorts_when_allowed():
    (sig,) = feed(make(allow_short=True), leader_jump(pct=-0.5))
    assert sig.action == "sell" and sig.kind == "entry" and "short AMD" in sig.reason


def test_stale_follower_is_skipped():
    s = make(stale_seconds=10)
    ticks = [tick("AMD", 100, 0), tick("NVDA", 200, 20), tick("NVDA", 201, 22)]
    (sig,) = feed(s, ticks)
    assert sig.action == "skip" and "22s old" in sig.reason


def test_no_follower_price_is_skipped():
    (sig,) = feed(make(), [tick("NVDA", 200, 0), tick("NVDA", 201, 2)])
    assert sig.action == "skip" and "no AMD price" in sig.reason


def test_several_followers_each_get_a_decision():
    s = LeadLagStrategy(LeadLagParams(leader="SPY", followers=["QQQ", "IWM"]), clock=SimClock())
    ticks = [tick("QQQ", 500, 0), tick("IWM", 200, 0), tick("SPY", 600, 0), tick("SPY", 602, 2)]
    assert sorted(sig.symbol for sig in feed(s, ticks)) == ["IWM", "QQQ"]


# ---------------------------------------------------------------- cooldown


def test_cooldown_stops_repeat_signals_on_the_same_move():
    s = make()
    first = feed(s, leader_jump())
    s.on_entry_failed("AMD")  # pretend the order was rejected so "already open" can't hide a repeat
    # The leader keeps climbing for a few more seconds: still the same move.
    again = feed(s, [tick("NVDA", 201.5, 4), tick("NVDA", 202.5, 6)])
    assert len(first) == 1 and again == []


def test_cooldown_ends_and_a_new_move_can_trigger():
    s = make()
    feed(s, leader_jump())
    s.on_entry_failed("AMD")
    later = feed(s, leader_jump(start_s=70))
    assert [sig.action for sig in later] == ["buy"]


def test_skips_also_start_the_cooldown():
    s = make()
    (first,) = feed(s, leader_jump(pct=-0.5))  # skipped: shorting off
    assert first.action == "skip"
    assert feed(s, [tick("NVDA", 198, 4), tick("NVDA", 197, 6)]) == []


# ---------------------------------------------------------------- exits


def opened(s, fill_price=100.0, fill_s=3.5):
    feed(s, leader_jump())
    s.on_entry_filled("AMD", fill_price, at(fill_s))
    return s.trades["AMD"]


def test_take_profit():
    s = make()
    opened(s)
    assert feed(s, [tick("AMD", 100.2, 5)]) == []
    (sig,) = feed(s, [tick("AMD", 100.35, 6)])
    assert (sig.action, sig.kind) == ("sell", "exit") and sig.reason.startswith("Take profit")
    assert feed(s, [tick("AMD", 100.5, 7)]) == []  # no second exit while one is pending


def test_stop_loss():
    s = make()
    opened(s)
    (sig,) = feed(s, [tick("AMD", 99.75, 6)])
    assert sig.reason.startswith("Stop loss") and sig.details["pnl_pct"] == pytest.approx(-0.25)


def test_time_exit_when_the_follower_does_not_follow():
    s = make()
    opened(s, fill_s=3.5)
    feed(s, [tick("AMD", 100.05, 10)])
    assert s.check_time_exits(at(30)) == []
    (sig,) = s.check_time_exits(at(34))
    assert sig.action == "sell" and sig.reason.startswith("Time exit") and "+0.05%" in sig.reason


def test_short_trade_exits_by_buying():
    s = make(allow_short=True)
    feed(s, leader_jump(pct=-0.5))
    s.on_entry_filled("AMD", 100.0, at(3.5))
    (sig,) = feed(s, [tick("AMD", 99.6, 5)])
    assert sig.action == "buy" and sig.reason.startswith("Take profit")


def test_pending_entry_has_no_exits():
    s = make()
    feed(s, leader_jump())
    assert feed(s, [tick("AMD", 90, 5)]) == [] and s.check_time_exits(at(500)) == []


def test_exit_fill_and_failure_callbacks():
    s = make()
    opened(s)
    feed(s, [tick("AMD", 100.5, 6)])
    s.on_exit_failed("AMD")
    assert s.trades["AMD"].exit_pending is False
    s.on_exit_filled("AMD")
    assert "AMD" not in s.trades


# ---------------------------------------------------------------- streams


RECEIVED = T0


def test_alpaca_parse_trades_and_errors():
    stream = streams.AlpacaStream(["NVDA"], on_tick=lambda t: None)
    raw = json.dumps(
        [
            {"T": "success", "msg": "authenticated"},
            {"T": "t", "S": "NVDA", "p": 201.5, "t": "2026-10-15T14:00:01.123456789Z"},
            {"T": "error", "code": 402, "msg": "auth failed"},
        ]
    )
    ticks, notes = stream.parse(raw, RECEIVED)
    assert ticks == [Tick("NVDA", 201.5, datetime(2026, 10, 15, 14, 0, 1, 123456, tzinfo=timezone.utc), RECEIVED)]
    assert ("fatal", "Alpaca error 402: auth failed") in notes


def test_polygon_parse_trades():
    stream = streams.PolygonStream(["AMD"], on_tick=lambda t: None)
    ticks, notes = stream.parse(
        json.dumps([{"ev": "T", "sym": "AMD", "p": 150.25, "t": int(at(1).timestamp() * 1000)}]), RECEIVED
    )
    assert ticks[0].symbol == "AMD" and ticks[0].ts == datetime(2026, 10, 15, 14, 0, 1, tzinfo=timezone.utc)


def test_subscribe_messages():
    assert streams.AlpacaStream(["AMD", "NVDA"], print).subscribe_message() == {
        "action": "subscribe",
        "trades": ["AMD", "NVDA"],
    }
    assert streams.PolygonStream(["AMD", "NVDA"], print).subscribe_message() == {
        "action": "subscribe",
        "params": "T.AMD,T.NVDA",
    }


def test_polygon_stream_refused_without_realtime_plan(monkeypatch, caplog):
    monkeypatch.setattr(settings, "POLYGON_REALTIME", False)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(streams.StreamNotAvailable):
            streams.make_stream("polygon", ["AMD"], print)
    assert "delayed by 15 minutes" in caplog.text


def test_polygon_stream_allowed_with_realtime_plan(monkeypatch):
    monkeypatch.setattr(settings, "POLYGON_REALTIME", True)
    monkeypatch.setattr(settings, "POLYGON_API_KEY", "k")
    assert isinstance(streams.make_stream("polygon", ["AMD"], print), streams.PolygonStream)


def test_alpaca_stream_needs_keys(monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "")
    with pytest.raises(streams.StreamNotAvailable):
        streams.make_stream("alpaca", ["AMD"], print)


class FakeSocket:
    def __init__(self, messages):
        self.sent = []
        self._messages = list(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, text):
        self.sent.append(json.loads(text))

    async def recv(self):
        import asyncio

        if self._messages:
            return self._messages.pop(0)
        await asyncio.sleep(0.05)
        raise ConnectionError("closed")


def test_stream_thread_authenticates_subscribes_and_delivers_ticks(monkeypatch):
    monkeypatch.setattr(settings, "ALPACA_API_KEY", "k")
    monkeypatch.setattr(settings, "ALPACA_SECRET_KEY", "s")
    sockets = []

    def connect(url, **kwargs):
        sock = FakeSocket([json.dumps([{"T": "t", "S": "AMD", "p": 150, "t": "2026-10-15T14:00:00Z"}])])
        sockets.append((url, sock))
        return sock

    received = []
    stream = streams.make_stream("alpaca", ["AMD"], received.append, connect=connect)
    stream.start()
    deadline = time.time() + 3
    while not received and time.time() < deadline:
        time.sleep(0.02)
    stream.stop()
    assert received and received[0].price == 150
    url, sock = sockets[0]
    assert url.endswith(f"/v2/{settings.ALPACA_FEED}")
    assert sock.sent[0] == {"action": "auth", "key": "k", "secret": "s"}
    assert sock.sent[1] == {"action": "subscribe", "trades": ["AMD"]}
