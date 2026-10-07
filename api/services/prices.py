"""
Daily price history for one ticker, from Yahoo Finance.

Only completed sessions are returned: while the market is open, today's partial
bar is dropped, so an analysis stays the same all day and can be cached.
"""

import logging
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from api.services import market_calendar

logger = logging.getLogger(__name__)

TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def normalize_ticker(raw: str) -> Optional[str]:
    """Upper-case and validate a ticker. Returns None if it is not a plausible symbol."""
    ticker = (raw or "").strip().upper()
    return ticker if TICKER_PATTERN.match(ticker) else None


def completed_sessions(close: pd.Series, now: Optional[datetime] = None) -> pd.Series:
    """Drop a bar for today if today's session has not closed yet."""
    if close.empty:
        return close
    now = now or datetime.now(tz=market_calendar.NEW_YORK)
    local = now.astimezone(market_calendar.NEW_YORK)
    last = pd.Timestamp(close.index[-1]).date()
    if last == local.date() and market_calendar.is_trading_day(last) and local < market_calendar.session_close(last):
        return close.iloc[:-1]
    return close


def daily_closes(ticker: str, period: str = "2y") -> pd.Series:
    """Adjusted daily closes, oldest first, tz-naive date index. Empty if Yahoo has nothing."""
    history = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if history.empty or "Close" not in history:
        return pd.Series(dtype=float)
    close = history["Close"].dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return completed_sessions(close)
