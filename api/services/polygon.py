"""
Polygon.io price bars (aggregates) for the paper bot and lead-lag research.

The free plan allows 5 requests a minute and its data is delayed, so:
- every request goes through one shared rate limiter (see throttle.py),
- results are cached, so asking twice for the same bars costs one request,
- every upstream problem becomes a PolygonError with a plain message, so the bot
  and the research job can log it and carry on instead of crashing.

The key goes in an Authorization header, never in the URL, so it never shows up
in logs or error messages.
"""

import logging
import math
from datetime import date, datetime, timedelta
from typing import Callable, Dict, Optional

import pandas as pd
import requests

from api.config import settings
from api.services import market_calendar
from api.services.prices import normalize_ticker
from api.services.throttle import RateLimited, RateLimiter, TTLCache

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 20
PAGE_LIMIT = 50000  # Polygon's largest page of aggregates
MAX_PAGES = 80
TIMESPANS = ("second", "minute", "hour", "day")
TTL_RECENT_SECONDS = {"second": 30, "minute": 60, "hour": 300, "day": 3600}
TTL_PAST_SECONDS = 12 * 3600  # bars for days that are over don't change
# Upper bounds used to estimate pages: Polygon intraday bars include pre and post market (4:00 to 20:00 ET).
BARS_PER_DAY_UPPER = {"second": 16 * 3600, "minute": 16 * 60, "hour": 16, "day": 1}

NOT_CONFIGURED = "Polygon needs an API key. Set POLYGON_API_KEY in .env."
COLUMNS = ["open", "high", "low", "close", "volume"]


class PolygonError(Exception):
    """A plain-language reason the bars aren't available. `retry_after` is set for rate limits."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


limiter = RateLimiter(settings.POLYGON_REQUESTS_PER_MINUTE)
_cache = TTLCache(max_size=256)


def configured() -> bool:
    return bool(settings.POLYGON_API_KEY)


def clear_cache() -> None:
    _cache.clear()


def _base_url() -> str:
    return settings.POLYGON_URL.rstrip("/")


def fetch_json(url: str, params: Dict, max_wait: Optional[float] = None) -> Dict:
    """One rate-limited GET. Raises PolygonError for every failure, never anything else."""
    try:
        limiter.acquire(max_wait)
    except RateLimited as exc:
        raise PolygonError(
            f"Polygon allows {limiter.limit} requests a minute. Try again in {math.ceil(exc.retry_after)} seconds.",
            retry_after=exc.retry_after,
        ) from None
    try:
        response = requests.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {settings.POLYGON_API_KEY}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning("Polygon request failed: %s", type(exc).__name__)
        raise PolygonError("Polygon didn't respond. Try again in a minute.") from None

    if response.status_code == 429:
        raise PolygonError("Polygon's rate limit was reached. Try again in a minute.", retry_after=60)
    if response.status_code in (401, 403):
        raise PolygonError("Polygon refused the request. Check POLYGON_API_KEY, and check that your plan includes this data.")
    if response.status_code >= 400:
        raise PolygonError(f"Polygon returned an error ({response.status_code}).")
    try:
        return response.json()
    except ValueError:
        raise PolygonError("Polygon sent a response that isn't valid JSON.") from None


def frame(rows) -> pd.DataFrame:
    """Polygon aggregate rows ({t, o, h, l, c, v}) to a DataFrame indexed by UTC bar start."""
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC"), dtype=float)
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[[c for c in COLUMNS if c in df.columns]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = None
    return df


def _today_ny() -> date:
    return datetime.now(market_calendar.NEW_YORK).date()


def get_bars(
    ticker: str,
    multiplier: int,
    timespan: str,
    start: date,
    end: date,
    fetch: Optional[Callable] = None,
    max_wait: Optional[float] = None,
) -> pd.DataFrame:
    """
    Bars from `start` to `end` (inclusive calendar days). Cached.
    `fetch(url, params, max_wait)` is injectable so tests never touch the network.
    """
    symbol = normalize_ticker(ticker)
    if symbol is None:
        raise PolygonError(f"'{ticker}' isn't a valid ticker.")
    if timespan not in TIMESPANS or multiplier < 1:
        raise PolygonError(f"Unsupported bar size: {multiplier} {timespan}.")
    if start > end:
        raise PolygonError("The start date is after the end date.")
    if fetch is None and not configured():
        raise PolygonError(NOT_CONFIGURED)

    key = (symbol, multiplier, timespan, start.isoformat(), end.isoformat())
    cached = _cache.get(key)
    if cached is not None:
        return cached

    fetch = fetch or fetch_json
    base = _base_url()
    url = f"{base}/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start.isoformat()}/{end.isoformat()}"
    params: Dict = {"adjusted": "true", "sort": "asc", "limit": PAGE_LIMIT}
    rows = []
    for _ in range(MAX_PAGES):
        data = fetch(url, params, max_wait)
        if not isinstance(data, dict):
            raise PolygonError("Polygon sent an unexpected response.")
        if data.get("status") == "ERROR":
            raise PolygonError("Polygon couldn't serve that request. Check the ticker and your plan.")
        rows.extend(data.get("results") or [])
        next_url = data.get("next_url")
        if not next_url:
            break
        if not next_url.startswith(base + "/"):
            # Never send the key to a host we didn't choose.
            logger.warning("Ignoring a Polygon next_url on an unexpected host")
            break
        url, params = next_url, {}
    else:
        logger.warning("Polygon bars for %s stopped after %d pages", symbol, MAX_PAGES)

    df = frame(rows)
    ttl = TTL_PAST_SECONDS if end < _today_ny() else TTL_RECENT_SECONDS[timespan]
    _cache.set(key, df, ttl)
    return df


def regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep bars that start between 9:30 and 16:00 New York time (the early close at 13:00 is respected)."""
    if df.empty:
        return df
    days = pd.Series(df.index.tz_convert(market_calendar.NEW_YORK).date, index=df.index)
    # The calendar is asked once per day, not once per bar (second bars can be a million rows).
    opens, closes = {}, {}
    for day in set(days):
        if market_calendar.is_trading_day(day):
            opens[day] = market_calendar.session_open(day)
            closes[day] = market_calendar.session_close(day)
    start = pd.to_datetime(days.map(opens), utc=True)
    stop = pd.to_datetime(days.map(closes), utc=True)
    index = pd.Series(df.index, index=df.index)
    keep = (index >= start) & (index < stop)  # NaT (not a trading day) compares False
    return df[keep.to_numpy()]


def get_daily_bars(
    ticker: str, days: int = 200, fetch: Optional[Callable] = None, today: Optional[date] = None, max_wait=None
):
    """About the last `days` daily bars, oldest first."""
    end = today or _today_ny()
    start = end - timedelta(days=int(days * 1.5) + 10)  # calendar days cover weekends and holidays
    return get_bars(ticker, 1, "day", start, end, fetch=fetch, max_wait=max_wait).tail(days)


def get_intraday_bars(
    ticker: str,
    start: date,
    end: date,
    timespan: str = "minute",
    multiplier: int = 1,
    regular_hours: bool = True,
    fetch: Optional[Callable] = None,
    max_wait: Optional[float] = None,
) -> pd.DataFrame:
    """Second, minute, or hour bars. Regular session only unless `regular_hours` is False."""
    df = get_bars(ticker, multiplier, timespan, start, end, fetch=fetch, max_wait=max_wait)
    return regular_session(df) if regular_hours else df


def trading_days_between(start: date, end: date) -> int:
    count, day = 0, start
    while day <= end:
        count += market_calendar.is_trading_day(day)
        day += timedelta(days=1)
    return count


def estimate_requests(start: date, end: date, timespan: str = "minute", multiplier: int = 1) -> int:
    """Upper bound on the requests (pages) one ticker needs for this range."""
    if timespan == "day":
        return 1
    bars = trading_days_between(start, end) * BARS_PER_DAY_UPPER[timespan] / multiplier
    return max(1, math.ceil(bars / PAGE_LIMIT))


def is_cached(ticker: str, multiplier: int, timespan: str, start: date, end: date) -> bool:
    symbol = normalize_ticker(ticker) or ticker
    return (symbol, multiplier, timespan, start.isoformat(), end.isoformat()) in _cache
