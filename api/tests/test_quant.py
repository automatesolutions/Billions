"""
Tests for the Quant page: bar calendar, strategy math, live record and endpoints. No network.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from api.models import QuantSignal
from api.services import bars, quant
from api.services import quant_strategy as qs

H, M, FOUR, D = (bars.INTERVALS[k] for k in ("1h", "1m", "4h", "1d"))


@pytest.fixture(autouse=True)
def clear_caches():
    quant._bars_cache.clear()
    quant._fit_cache.clear()
    yield


def _daily(n, seed=0, phi=0.3):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0, 0.01)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


# --------------------------------------------------------------------------- bar calendar


def test_next_bar_start_skips_nights_and_weekends():
    assert bars.next_bar_start(H, pd.Timestamp("2026-10-15 14:30")) == pd.Timestamp("2026-10-15 15:30")
    assert bars.next_bar_start(H, pd.Timestamp("2026-10-15 15:30")) == pd.Timestamp("2026-10-16 09:30")
    assert bars.next_bar_start(H, pd.Timestamp("2026-10-16 15:30")) == pd.Timestamp("2026-10-19 09:30")  # Fri -> Mon
    assert bars.next_bar_start(FOUR, pd.Timestamp("2026-10-15 09:30")) == pd.Timestamp("2026-10-15 13:30")
    assert bars.next_bar_start(FOUR, pd.Timestamp("2026-10-15 13:30")) == pd.Timestamp("2026-10-16 09:30")
    assert bars.next_bar_start(M, pd.Timestamp("2026-10-15 15:59")) == pd.Timestamp("2026-10-16 09:30")
    assert bars.next_bar_start(D, pd.Timestamp("2026-10-16")) == pd.Timestamp("2026-10-19")


def test_bar_end_is_capped_at_the_close():
    assert bars.bar_end(H, pd.Timestamp("2026-10-15 15:30")) == datetime(2026, 10, 15, 16, 0)
    assert bars.bar_end(FOUR, pd.Timestamp("2026-10-15 09:30")) == datetime(2026, 10, 15, 13, 30)
    assert bars.bar_end(H, pd.Timestamp("2026-11-27 12:30")) == datetime(2026, 11, 27, 13, 0)  # day after Thanksgiving


def test_completed_drops_the_forming_bar():
    idx = pd.DatetimeIndex(["2026-10-15 09:30", "2026-10-15 10:30", "2026-10-15 11:30"])
    close = pd.Series([1.0, 2.0, 3.0], index=idx)
    now = datetime(2026, 10, 15, 11, 33, tzinfo=bars.NY)  # 10:30 bar ended 11:30, +2 min grace
    assert list(bars.completed(H, close, now)) == [1.0, 2.0]


def test_four_hour_buckets():
    idx = pd.date_range("2026-10-15 09:30", periods=7, freq="h")
    hourly = pd.Series(np.arange(1.0, 8.0), index=idx)
    four = bars.to_four_hour(hourly)
    assert list(four.index) == [pd.Timestamp("2026-10-15 09:30"), pd.Timestamp("2026-10-15 13:30")]
    assert list(four) == [4.0, 7.0]  # last close in each bucket


# --------------------------------------------------------------------------- strategy math


def test_net_returns_charge_half_a_round_trip_per_unit_traded():
    pos, y = np.array([1.0, 1.0, -1.0, 0.0]), np.array([0.01, 0.01, -0.01, 0.02])
    net = qs.net_returns(pos, y, cost_bps=10)
    half = 5 / 1e4
    assert net == pytest.approx([0.01 - half, 0.01, 0.01 - 2 * half, -half])


def test_net_returns_flat_overnight_closes_and_reopens():
    pos, y = np.array([1.0, 1.0]), np.array([0.0, 0.0])
    net = qs.net_returns(pos, y, cost_bps=10, breaks=np.array([True, True]))
    assert net == pytest.approx([-5e-4, -10e-4])  # open; then close yesterday's and open today's


def test_size_position_is_bounded():
    p = qs.size_position(np.array([-1.0, 0.0, 1.0]), 0.01)
    assert p[0] == pytest.approx(-1) and p[1] == 0 and p[2] == pytest.approx(1)


def _fit_stub(hit=0.55, n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 0.01, n)
    pred = np.where(rng.random(n) < hit, np.sign(y), -np.sign(y)) * 0.001
    return {"hit_rate": qs.qa.hit_rate(pred, y), "gross": qs.qa.edge_metrics(qs.qa.signal_returns(pred, y)), "oos_stacked": pred, "y_test": y}


def _signals(n, hit, seed=1):
    rng = np.random.default_rng(seed)
    out, t0 = [], pd.Timestamp("2026-10-01 09:30")
    for i in range(n):
        realized = float(rng.normal(0, 0.01))
        forecast = 0.001 * (np.sign(realized) if rng.random() < hit else -np.sign(realized))
        out.append({"bar_time": t0 + pd.Timedelta(hours=i), "target_time": t0 + pd.Timedelta(hours=i + 1),
                    "forecast": forecast, "position": 0.5, "cost_bps": 2.0, "realized": realized, "void": False})
    return out


def test_live_summary_verdicts():
    fit = _fit_stub(hit=0.6)
    early = qs.live_summary(_signals(5, 0.6), fit, H)
    assert early["verdict"] == "too_early"
    assert 0 <= early["backtest"]["hit_rate_band"][0] and early["backtest"]["hit_rate_band"][1] <= 1
    assert qs.live_summary(_signals(200, 0.6), fit, H)["verdict"] == "in_line"
    bad = qs.live_summary(_signals(200, 0.35), fit, H)
    assert bad["verdict"] == "underperforming" and bad["model"]["hit_rate_z"] < -2
    assert bad["random_baseline"]["draws"] == qs.qa.RANDOM_BASELINE_DRAWS


def test_live_action_record_carries_positions_between_back_to_back_signals():
    sigs = _signals(3, 1.0)
    for s in sigs:
        s["realized"], s["position"] = 0.0, 1.0
    record = qs.live_summary(sigs, _fit_stub(), H)["action"]
    # Opened once (1 bps), then carried: total cost is one half round trip.
    assert record["total_log_return"] == pytest.approx(-1e-4)


# --------------------------------------------------------------------------- service and endpoints


def _fetch(close):
    return lambda ticker, iv: close


def test_records_then_scores_signals(db_session):
    close = _daily(520)
    cost = {"bps": 2.0, "source": "default"}
    first = quant.get_quant(db_session, "TEST", "1d", fetch=_fetch(close.iloc[:-1]), cost=cost)
    assert first["status"] == "ok"
    assert first["live"]["recorded"] == 1 and first["live"]["scored"] == 0
    assert first["action"]["action"] in ("long", "short", "flat")
    assert {g["id"] for g in first["action"]["gates"]} >= {"beats_random", "after_costs", "live_record"}

    second = quant.get_quant(db_session, "TEST", "1d", fetch=_fetch(close), cost=cost)
    assert second["live"]["recorded"] == 2 and second["live"]["scored"] == 1
    scored = db_session.query(QuantSignal).filter(QuantSignal.realized.isnot(None)).one()
    assert scored.realized == pytest.approx(math.log(close.iloc[-1] / close.iloc[-2]))
    assert second["live"]["recent"][1]["status"] == "scored"


def test_signal_with_missing_target_bar_is_void(db_session):
    close = _daily(520)
    cost = {"bps": 2.0, "source": "default"}
    quant.get_quant(db_session, "GAP", "1d", fetch=_fetch(close.iloc[:-2]), cost=cost)
    quant.get_quant(db_session, "GAP", "1d", fetch=_fetch(close.drop(close.index[-2])), cost=cost)
    assert db_session.query(QuantSignal).filter(QuantSignal.void.is_(True)).count() == 1


def test_short_history_is_reported(db_session):
    payload = quant.get_quant(db_session, "NEW", "1d", fetch=_fetch(_daily(100)), cost={"bps": 2.0, "source": "default"})
    assert payload["status"] == "insufficient_history" and payload["needed_bars"] == qs.MIN_BARS


@pytest.fixture
def fake_quant(monkeypatch):
    close = _daily(520)
    monkeypatch.setattr(bars, "load", lambda ticker, iv: close)
    monkeypatch.setattr(quant, "round_trip_cost", lambda ticker: {"bps": 2.0, "source": "default"})


def test_quant_endpoint(client, fake_quant):
    response = client.get("/api/v1/quant/aapl?interval=1d")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    data = response.json()
    assert data["ticker"] == "AAPL" and data["interval"] == "1d" and data["status"] == "ok"
    for key in ("action", "explain", "backtest", "live", "caveats", "cost"):
        assert key in data


def test_quant_endpoint_validates(client):
    assert client.get("/api/v1/quant/AAPL?interval=5m").status_code == 422
    assert client.get("/api/v1/quant/$$$").status_code == 422


def test_watch_list(client, monkeypatch):
    from api.config import settings

    monkeypatch.setattr(settings, "QUANT_MAX_WATCHES", 2)
    assert client.post("/api/v1/quant/AAPL/watch?interval=1h").json()["watching"] is True
    assert client.post("/api/v1/quant/AAPL/watch?interval=1h").status_code == 200  # idempotent
    assert client.post("/api/v1/quant/MSFT/watch?interval=1m").status_code == 200
    assert client.post("/api/v1/quant/TSLA/watch?interval=1d").status_code == 409
    assert client.get("/api/v1/quant/watching").json()["watching"] == [
        {"ticker": "AAPL", "interval": "1h"},
        {"ticker": "MSFT", "interval": "1m"},
    ]
    assert client.delete("/api/v1/quant/AAPL/watch?interval=1h").json()["watching"] is False
    assert len(client.get("/api/v1/quant/watching").json()["watching"]) == 1


@pytest.mark.parametrize(
    "previous, position, code, headline",
    [
        (0.0, 0.4, "buy", "Buy"),
        (0.0, -0.4, "sell_short", "Sell short"),
        (0.0, 0.0, "hold_flat", "Hold"),
        (0.4, 0.42, "hold_long", "Hold"),
        (-0.3, -0.3, "hold_short", "Hold"),
        (0.4, 0.0, "sell", "Sell"),
        (-0.4, 0.0, "buy_to_cover", "Buy to cover"),
        (0.4, -0.2, "reverse_to_short", "Sell + short"),
        (-0.4, 0.2, "reverse_to_long", "Cover + buy"),
    ],
)
def test_instruction(previous, position, code, headline):
    out = qs.instruction(previous, position)
    assert out["code"] == code and out["headline"] == headline


def test_instruction_mentions_resizing():
    assert "add to it from 20% to 60%" in qs.instruction(0.2, 0.6)["sentence"]
    assert "trim it from 60% to 20%" in qs.instruction(-0.6, -0.2)["sentence"]
    assert qs.instruction(0.4, 0.42)["sentence"] == "Keep the long position."


def test_payload_has_instruction(db_session):
    payload = quant.get_quant(db_session, "ORD", "1d", fetch=_fetch(_daily(520)), cost={"bps": 2.0, "source": "default"})
    assert payload["action"]["instruction"]["code"] in {v[0] for v in qs.INSTRUCTIONS.values()}
    assert payload["action"]["instruction"]["previous_position"] == 0.0  # nothing held before the first signal
    assert payload["live"]["recent"][0]["instruction"] in {v[0] for v in qs.INSTRUCTIONS.values()}
