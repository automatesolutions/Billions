"""
Price bars at the intervals the Quant page offers: 1 minute, 1 hour, 4 hours, daily.

All from Yahoo Finance (consolidated prices, one request per load). Intraday bars
are regular-session only (9:30 to 16:00 New York), indexed by their start time as
New York wall-clock (tz-naive), and only completed bars are returned.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from api.services import market_calendar, prices

NY = market_calendar.NEW_YORK
OPEN = time(9, 30)


@dataclass(frozen=True)
class Interval:
    key: str
    label: str
    minutes: int  # bar length; daily bars use 0
    periods_per_year: int  # for annualised Sharpe
    slot_minutes: Optional[int]  # time-of-session feature for XGBoost; None uses weekday
    slots: int
    flat_overnight: bool  # True: no position is held across the overnight gap
    grace_seconds: int  # wait after a bar ends before treating it as final

    @property
    def intraday(self) -> bool:
        return self.minutes > 0


INTERVALS: Dict[str, Interval] = {
    "1m": Interval("1m", "1 minute", 1, 390 * 252, 65, 6, True, 20),
    "1h": Interval("1h", "1 hour", 60, 7 * 252, 60, 7, False, 120),
    "4h": Interval("4h", "4 hours", 240, 2 * 252, 240, 2, False, 120),
    "1d": Interval("1d", "Daily", 0, 252, None, 0, False, 0),
}


def _session_close(day: date) -> datetime:
    return market_calendar.session_close(day).astimezone(NY).replace(tzinfo=None)


def bar_end(iv: Interval, start: pd.Timestamp) -> datetime:
    """New York wall-clock time the bar starting at `start` closes (capped at the session close)."""
    start = pd.Timestamp(start).to_pydatetime()
    if not iv.intraday:
        return _session_close(start.date())
    return min(start + timedelta(minutes=iv.minutes), _session_close(start.date()))


def next_bar_start(iv: Interval, start: pd.Timestamp) -> pd.Timestamp:
    """Start of the bar after the one starting at `start`, skipping nights, weekends and holidays."""
    start = pd.Timestamp(start)
    if not iv.intraday:
        return pd.Timestamp(market_calendar.next_trading_day(start.date()))
    candidate = start + pd.Timedelta(minutes=iv.minutes)
    if candidate.to_pydatetime() < _session_close(start.date()):
        return candidate
    return pd.Timestamp(datetime.combine(market_calendar.next_trading_day(start.date()), OPEN))


def _now_ny(now: Optional[datetime]) -> datetime:
    return (now or datetime.now(tz=NY)).astimezone(NY).replace(tzinfo=None)


def completed(iv: Interval, close: pd.Series, now: Optional[datetime] = None) -> pd.Series:
    """Drop bars that have not closed yet (plus the interval's grace period)."""
    if close.empty or not iv.intraday:
        return close
    cutoff = _now_ny(now) - timedelta(seconds=iv.grace_seconds)
    ends = pd.Series([bar_end(iv, t) for t in close.index], index=close.index)
    return close[ends <= cutoff]


def _yahoo_intraday(ticker: str, interval: str, period: str) -> pd.Series:
    history = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True, prepost=False)
    if history.empty or "Close" not in history:
        return pd.Series(dtype=float)
    close = history["Close"].dropna()
    index = pd.DatetimeIndex(close.index)
    close.index = (index.tz_convert(NY) if index.tz is not None else index.tz_localize(NY)).tz_localize(None)
    session = (close.index.time >= OPEN) & (close.index.time < time(16, 0))
    return close[session & close.index.map(lambda t: market_calendar.is_trading_day(t.date()))]


def to_four_hour(hourly: pd.Series) -> pd.Series:
    """9:30-13:30 and 13:30-close buckets from 9:30-aligned hourly bars; close = last close in the bucket."""
    if hourly.empty:
        return hourly
    starts = [
        datetime.combine(t.date(), OPEN if t.time() < time(13, 30) else time(13, 30)) for t in hourly.index
    ]
    return hourly.groupby(pd.DatetimeIndex(starts)).last()


def load(ticker: str, iv: Interval, now: Optional[datetime] = None) -> pd.Series:
    """Completed closes for `ticker` at interval `iv`, oldest first. Empty if Yahoo has nothing."""
    if iv.key == "1d":
        return prices.daily_closes(ticker, period="2y")
    if iv.key == "1m":
        raw = _yahoo_intraday(ticker, "1m", "8d")
        return completed(iv, raw, now)
    hourly = completed(INTERVALS["1h"], _yahoo_intraday(ticker, "60m", "730d"), now)
    return hourly if iv.key == "1h" else completed(iv, to_four_hour(hourly), now)
