"""
Strategy 1 (moving-average crossover) signals and the risk limits. Pure functions; no network.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from api.services.bot.risk import RiskLimits, RiskManager
from api.services.bot.strategies import MACrossoverParams, MovingAverageCrossover, Signal, ma_chart
from api.services.market_calendar import NEW_YORK


def strategy(short=2, long=4):
    return MovingAverageCrossover(MACrossoverParams(tickers=["aapl"], short_window=short, long_window=long))


def closes(values):
    return pd.Series(values, index=pd.date_range("2026-01-01", periods=len(values)), dtype=float)


# ---------------------------------------------------------------- moving-average crossover


def test_params_validate_windows_and_tickers():
    p = MACrossoverParams(tickers=["aapl", "AAPL", "msft"])
    assert p.tickers == ["AAPL", "MSFT"]
    with pytest.raises(ValidationError):
        MACrossoverParams(tickers=["AAPL"], short_window=30, long_window=10)
    with pytest.raises(ValidationError):
        MACrossoverParams(tickers=["$$$"])
    with pytest.raises(ValidationError):
        MACrossoverParams(tickers=[])


def test_not_enough_bars_holds():
    s = strategy().on_bars("AAPL", closes([1, 2, 3]), 0)
    assert s.action == "hold" and "need 5" in s.reason


def test_cross_above_buys():
    # Falling, then a jump: the 2-bar average moves above the 4-bar one on the last bar.
    s = strategy().on_bars("AAPL", closes([10, 9, 8, 7, 6, 12]), 0)
    assert s.action == "buy" and s.kind == "entry" and s.price == 12
    assert "crossed above" in s.reason
    assert s.details["short_ma"] > s.details["long_ma"]


def test_cross_above_while_holding_holds():
    s = strategy().on_bars("AAPL", closes([10, 9, 8, 7, 6, 12]), 5)
    assert s.action == "hold" and "already hold" in s.reason


def test_cross_below_sells_only_with_a_position():
    series = closes([6, 7, 8, 9, 10, 4])
    assert strategy().on_bars("AAPL", series, 5).action == "sell"
    held_none = strategy().on_bars("AAPL", series, 0)
    assert held_none.action == "hold" and "no position" in held_none.reason


def test_no_new_cross_holds_with_reason():
    s = strategy().on_bars("AAPL", closes([1, 2, 3, 4, 5, 6, 7]), 0)
    assert s.action == "hold" and "above" in s.reason


def test_signal_rejects_unknown_action():
    with pytest.raises(ValueError):
        Signal("yolo", "AAPL", "no")


# ---------------------------------------------------------------- risk


class Clock:
    def __init__(self, when):
        self.when = when

    def __call__(self):
        return self.when


def risk(max_dollars=1000, max_orders=2, clock=None):
    return RiskManager(RiskLimits(max_position_dollars=max_dollars, max_orders_per_day=max_orders), clock=clock)


def test_entry_is_sized_in_whole_shares_under_the_limit():
    check = risk().size_entry(price=300)
    assert check.allowed and check.qty == 3


def test_existing_position_counts_against_the_limit():
    assert risk().size_entry(price=100, existing_value=750).qty == 2
    blocked = risk().size_entry(price=100, existing_value=950)
    assert not blocked.allowed and "less than one share" in blocked.reason


def test_expensive_stock_is_blocked():
    check = risk(max_dollars=100).size_entry(price=150)
    assert not check.allowed and check.qty == 0


def test_daily_order_limit_blocks_entries_but_not_exits():
    r = risk(max_orders=2)
    r.record_order()
    r.record_order()
    entry = r.size_entry(price=10)
    assert not entry.allowed and "2 of 2" in entry.reason
    assert r.allow_exit().allowed


def test_daily_count_resets_on_a_new_new_york_day():
    clock = Clock(datetime(2026, 10, 15, 15, 0, tzinfo=NEW_YORK))
    r = risk(max_orders=1, clock=clock)
    r.record_order()
    assert not r.size_entry(10).allowed
    clock.when += timedelta(days=1)
    assert r.size_entry(10).allowed and r.status()["orders_today"] == 0


def test_kill_switch_blocks_everything():
    r = risk()
    r.kill("you pressed the kill switch")
    assert not r.size_entry(10).allowed
    exit_check = r.allow_exit()
    assert not exit_check.allowed and "kill switch" in exit_check.reason.lower()
    assert r.status()["killed"] is True


def test_zero_price_is_blocked():
    assert not risk().size_entry(price=0).allowed


def test_chart_marks_every_cross_the_strategy_acts_on():
    # The chart and the strategy must agree: a marker is exactly a bar where on_bars sees a fresh cross.
    values = [10, 9, 8, 7, 6, 12, 13, 14, 4, 3, 2, 9, 10]
    s = closes(values)
    chart = ma_chart(s, 2, 4)
    from_strategy = []
    for end in range(5, len(values) + 1):
        reason = strategy().on_bars("AAPL", s.iloc[:end], 0).reason
        if "crossed above" in reason.lower():
            from_strategy.append((s.index[end - 1].isoformat(), "up"))
        elif "crossed below" in reason.lower():
            from_strategy.append((s.index[end - 1].isoformat(), "down"))
    assert [(c["t"], c["direction"]) for c in chart["crosses"]] == from_strategy
    assert {d for _, d in from_strategy} == {"up", "down"}


def test_chart_keeps_the_last_points_with_empty_averages_at_the_start():
    chart = ma_chart(closes(list(range(1, 11))), 2, 4, points=6)
    assert len(chart["bars"]) == 6 and chart["bars"][-1]["close"] == 10
    early = ma_chart(closes([1, 2, 3, 4, 5]), 2, 4)
    assert early["bars"][0]["long"] is None and early["bars"][-1]["long"] == 3.5
    assert early["gap"] == 1.0
