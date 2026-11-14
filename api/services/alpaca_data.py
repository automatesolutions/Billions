"""
Historical intraday bars from Alpaca's market data API, a second source for lead-lag research.

The free plan allows about 200 requests a minute (Polygon's free plan allows 5), so
scans finish much faster. The tradeoff: with ALPACA_FEED=iex the bars count only
trades on the IEX exchange, a small slice of all volume, so quiet minutes can be
missing for less traded names. Alpaca has no second bars; the finest is 1 minute.
"""

import logging
from datetime import date, datetime, time, timezone
from typing import Callable, Dict, Optional

import pandas as pd
import requests

from api.config import settings
from api.services import market_calendar, polygon
from api.services.prices import normalize_ticker
from api.services.throttle import RateLimited, RateLimiter, TTLCache

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 20
PAGE_LIMIT = 10000
MAX_PAGES = 100
BARS_PER_DAY_UPPER = 16 * 60  # extended hours included

NOT_CONFIGURED = "Alpaca bars need ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."


class AlpacaDataError(Exception):
    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


limiter = RateLimiter(settings.ALPACA_DATA_REQUESTS_PER_MINUTE)
_cache = TTLCache(max_size=256)


def configured() -> bool:
    return bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY)


def clear_cache() -> None:
    _cache.clear()


def fetch_json(url: str, params: Dict, max_wait: Optional[float] = None) -> Dict:
    try:
        limiter.acquire(max_wait)
    except RateLimited as exc:
        raise AlpacaDataError("Alpaca's data rate limit was reached. Try again in a minute.", exc.retry_after) from None
    try:
        response = requests.get(
            url,
            params=params,
            headers={"APCA-API-KEY-ID": settings.ALPACA_API_KEY, "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning("Alpaca data request failed: %s", type(exc).__name__)
        raise AlpacaDataError("Alpaca's data API didn't respond. Try again in a minute.") from None
    if response.status_code == 429:
        raise AlpacaDataError("Alpaca's data rate limit was reached. Try again in a minute.", 60)
    if response.status_code in (401, 403):
        raise AlpacaDataError("Alpaca refused the request. Check your keys, and check that ALPACA_FEED matches your plan.")
    if response.status_code >= 400:
        raise AlpacaDataError(f"Alpaca's data API returned an error ({response.status_code}).")
    try:
        return response.json()
    except ValueError:
        raise AlpacaDataError("Alpaca sent a response that isn't valid JSON.") from None


def _rfc3339(day: date, at: time) -> str:
    return (
        datetime.combine(day, at, tzinfo=market_calendar.NEW_YORK).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    )


def get_intraday_bars(
    ticker: str,
    start: date,
    end: date,
    regular_hours: bool = True,
    fetch: Optional[Callable] = None,
    max_wait: Optional[float] = None,
) -> pd.DataFrame:
    """1-minute bars, indexed by UTC bar start. Same shape as polygon.get_intraday_bars."""
    symbol = normalize_ticker(ticker)
    if symbol is None:
        raise AlpacaDataError(f"'{ticker}' isn't a valid ticker.")
    if fetch is None and not configured():
        raise AlpacaDataError(NOT_CONFIGURED)
    key = (symbol, start.isoformat(), end.isoformat())
    df = _cache.get(key)
    if df is None:
        fetch = fetch or fetch_json
        url = f"{settings.ALPACA_DATA_URL.rstrip('/')}/v2/stocks/{symbol}/bars"
        base = {
            "timeframe": "1Min",
            "start": _rfc3339(start, time(0, 0)),
            "end": _rfc3339(end, time(23, 59)),
            "limit": PAGE_LIMIT,
            "feed": settings.ALPACA_FEED,
            "adjustment": "all",
        }
        rows, token = [], None
        for _ in range(MAX_PAGES):
            params = {**base, **({"page_token": token} if token else {})}
            data = fetch(url, params, max_wait)
            if not isinstance(data, dict):
                raise AlpacaDataError("Alpaca sent an unexpected response.")
            for bar in data.get("bars") or []:
                rows.append({"t": bar["t"], "o": bar["o"], "h": bar["h"], "l": bar["l"], "c": bar["c"], "v": bar["v"]})
            token = data.get("next_page_token")
            if not token:
                break
        df = _frame(rows)
        today = datetime.now(market_calendar.NEW_YORK).date()
        _cache.set(key, df, polygon.TTL_PAST_SECONDS if end < today else 60)
    return polygon.regular_session(df) if regular_hours else df


def _frame(rows) -> pd.DataFrame:
    if not rows:
        return polygon.frame([])
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})[polygon.COLUMNS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = None
    return df


def estimate_requests(start: date, end: date) -> int:
    days = polygon.trading_days_between(start, end)
    return max(1, -(-days * BARS_PER_DAY_UPPER // PAGE_LIMIT))


def is_cached(ticker: str, start: date, end: date) -> bool:
    return ((normalize_ticker(ticker) or ticker), start.isoformat(), end.isoformat()) in _cache
