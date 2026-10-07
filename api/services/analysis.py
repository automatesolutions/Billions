"""
Analysis service: fetch prices, run the quant analysis, cache the result.

Results depend only on completed daily sessions, so each (ticker, last session)
pair is computed once and kept until the next close.
"""

import logging
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from api.services import market_calendar, prices
from api.services import quant_analysis as qa

logger = logging.getLogger(__name__)

CACHE_SIZE = 256
PRICE_TTL_SECONDS = 600  # re-check Yahoo for a new session at most every 10 minutes per ticker


class TickerNotFound(Exception):
    pass


_cache: "OrderedDict[tuple, Dict]" = OrderedDict()
_cache_lock = threading.Lock()
_ticker_locks: Dict[str, threading.Lock] = {}
_price_cache: Dict[str, tuple] = {}


def _clean(value):
    """Make numpy/pandas values JSON-safe: floats stay floats, NaN and inf become None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_analysis(ticker: str, close: pd.Series) -> Dict:
    """Pure assembly of the API payload from closes. Raises TickerNotFound if there are no prices."""
    if close.empty:
        raise TickerNotFound(ticker)

    last_date = pd.Timestamp(close.index[-1]).date()
    next_date = pd.Timestamp(market_calendar.next_trading_day(last_date))
    base = {
        "ticker": ticker,
        "price": float(close.iloc[-1]),
        "as_of": last_date.isoformat(),
        "session_close": market_calendar.session_close(last_date).isoformat(),
        "history": qa.history_window(close),
        "default_cost_bps": qa.DEFAULT_ROUND_TRIP_COST_BPS,
    }
    returns = qa.log_returns(close)
    try:
        result = qa.analyze(close, next_date)
        payload = {**base, "status": "ok", **result}
    except ValueError as exc:
        payload = {
            **base,
            "status": "insufficient_history",
            "message": str(exc),
            "needed_days": qa.MIN_RETURNS,
            "history_days": int(len(returns)),
            "risk": qa.risk_context(returns) if len(returns) >= qa.MIN_RISK_RETURNS else None,
        }
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    return _clean(payload)


def get_analysis(ticker: str, fetch: Optional[Callable[[str], pd.Series]] = None) -> Dict:
    """Cached analysis for a validated ticker. `fetch` defaults to Yahoo daily closes."""
    use_price_cache = fetch is None
    fetch = fetch or prices.daily_closes
    with _cache_lock:
        lock = _ticker_locks.setdefault(ticker, threading.Lock())
    with lock:  # one computation per ticker at a time
        cached = _price_cache.get(ticker)
        if use_price_cache and cached and time.monotonic() - cached[0] < PRICE_TTL_SECONDS:
            close = cached[1]
        else:
            close = fetch(ticker)
            if use_price_cache:
                _price_cache[ticker] = (time.monotonic(), close)
        if close.empty:
            raise TickerNotFound(ticker)
        key = (ticker, pd.Timestamp(close.index[-1]).date().isoformat(), len(close))
        with _cache_lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]
        payload = build_analysis(ticker, close)
        with _cache_lock:
            _cache[key] = payload
            while len(_cache) > CACHE_SIZE:
                _cache.popitem(last=False)
        return payload


def warm(tickers: Iterable[str], limit: Optional[int] = None) -> int:
    """Pre-compute analyses (used after an outlier refresh). Returns how many succeeded."""
    done = 0
    for ticker in list(tickers)[:limit]:
        try:
            get_analysis(ticker)
            done += 1
        except Exception as exc:  # noqa: BLE001 - warming is best-effort
            logger.info("Warm-up skipped %s: %s", ticker, exc)
    return done
