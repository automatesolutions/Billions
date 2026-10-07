from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from api.services import market_calendar as mc
from api.services.outlier_engine import (
    STRATEGIES,
    Strategy,
    compute_strategy_metrics,
    fresh_symbols,
    liquid_symbols,
    trailing_return,
)
from api.services.outliers import direction, outlier_score, reason, should_refresh
from api.services.universe import parse_nasdaq_trader


def test_trailing_return_counts_trading_days_exactly():
    close = pd.Series([100.0, 101.0, 102.0, 110.0])
    # 3 days back from the last close is the first value: 110 / 100 - 1
    assert trailing_return(close, 3) == pytest.approx(10.0)
    assert np.isnan(trailing_return(close, 4))


def _frame(n_days=300, n_symbols=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n_days)
    returns = rng.normal(0, 0.01, size=(n_days, n_symbols))
    close = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=[f"S{i}" for i in range(n_symbols)])
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=close.columns)
    return close, volume


def test_spike_is_flagged_as_outlier():
    close, volume = _frame()
    close.iloc[-5:, 0] *= 1.6  # S0 jumps 60% in the last week
    df = compute_strategy_metrics(close, volume, STRATEGIES["scalp"])
    assert len(df) == 40
    assert df.loc["S0", "is_outlier"]
    assert df["is_outlier"].sum() < 10


def test_liquidity_floor_and_cap():
    close, volume = _frame(n_symbols=5)
    volume["S0"] = 1.0  # $100/day
    assert "S0" not in liquid_symbols(close, volume, min_dollar_volume=1e6)
    assert len(liquid_symbols(close, volume, min_dollar_volume=0, limit=3)) == 3


def test_stale_symbols_are_dropped():
    close, _ = _frame(n_symbols=3)
    close.iloc[-10:, 1] = np.nan  # S1 stopped trading 10 sessions ago
    assert fresh_symbols(close) == ["S0", "S2"]


def test_too_few_symbols_returns_empty_scores():
    close, volume = _frame(n_symbols=2)
    df = compute_strategy_metrics(close, volume, STRATEGIES["swing"])
    assert df["z_x"].isna().all() or df.empty


def test_score_direction_reason():
    s = Strategy("scalp", 21, 5, "1 month", "1 week", 0)
    assert outlier_score(3.0, 4.0) == pytest.approx(5.0)
    assert direction(2.5, 0.5) == "up"
    assert direction(-0.3, -2.4) == "down"
    assert direction(2.5, -2.2) == "mixed"
    assert reason(s, 30.0, 2.0, 2.5, 0.3) == "Up 30.0% over 1 month, 2.5σ from the group"
    assert reason(s, -1.0, -12.0, -0.1, -3.1) == "Down 12.0% over 1 week, 3.1σ from the group"


def test_parse_nasdaq_trader_keeps_na_ticker_and_drops_etfs():
    text = (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AAPL|Apple Inc.|Q|N|N|100|N|N\n"
        "NA|Nano Labs|S|N|N|100|N|N\n"
        "QQQ|Invesco QQQ|G|N|N|100|Y|N\n"
        "ZZZT|Test|Q|Y|N|100|N|N\n"
        "File Creation Time: 1007202603:02|||||||\n"
    )
    assert parse_nasdaq_trader(text) == ["AAPL", "NA"]


class TestScheduler:
    def at(self, y, m, d, hh, mm):
        return datetime(y, m, d, hh, mm, tzinfo=mc.NEW_YORK).astimezone(timezone.utc)

    def test_runs_when_empty(self):
        assert should_refresh(self.at(2026, 10, 10, 12, 0), None)

    def test_interval_while_open(self):
        now = self.at(2026, 10, 7, 11, 0)
        assert not should_refresh(now, now - timedelta(minutes=10))
        assert should_refresh(now, now - timedelta(minutes=31))

    def test_once_after_close(self):
        after = self.at(2026, 10, 7, 16, 30)
        assert should_refresh(after, self.at(2026, 10, 7, 15, 50))
        assert not should_refresh(after, self.at(2026, 10, 7, 16, 25))

    def test_weekend_needs_no_refresh_after_friday_close_run(self):
        assert not should_refresh(self.at(2026, 10, 10, 12, 0), self.at(2026, 10, 9, 16, 25))

    def test_backoff_after_failure(self):
        now = self.at(2026, 10, 10, 12, 0)
        assert not should_refresh(now, None, last_failure=now - timedelta(minutes=5))
