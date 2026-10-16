"""
Live top-of-book quotes (best bid and ask) from Alpaca's market data API.

The free plan's "iex" feed shows quotes from the IEX exchange only, so the
spread is usually wider than the national best bid and offer. Without keys
the endpoint says so instead of estimating anything.
"""

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional

import requests

from api.config import settings
from api.services import market_calendar

logger = logging.getLogger(__name__)

QUOTE_TTL_SECONDS = 10  # one upstream call per ticker per 10 seconds
REQUEST_TIMEOUT_SECONDS = 5

NOT_CONFIGURED = (
    "Live quotes need free Alpaca API keys (ALPACA_API_KEY and ALPACA_SECRET_KEY). "
    "Yahoo Finance provides daily bars only, so these measures are not shown."
)

_cache: Dict[str, tuple] = {}
_lock = threading.Lock()


def configured() -> bool:
    return bool(settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY)


def fetch_alpaca_quote(ticker: str) -> Dict:
    """Raw latest quote: {'bp', 'bs', 'ap', 'as', 't', ...}. Raises on HTTP errors."""
    response = requests.get(
        f"{settings.ALPACA_DATA_URL.rstrip('/')}/v2/stocks/{ticker}/quotes/latest",
        params={"feed": settings.ALPACA_FEED},
        headers={"APCA-API-KEY-ID": settings.ALPACA_API_KEY, "APCA-API-SECRET-KEY": settings.ALPACA_SECRET_KEY},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("quote") or {}


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    # Alpaca sends RFC 3339 with up to nanoseconds; Python parses up to microseconds.
    text = re.sub(r"(\.\d{6})\d+", r"\1", value.replace("Z", "+00:00"))
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def measures(bid: float, ask: float, bid_size: float, ask_size: float) -> Dict:
    """Mid-price, spread, top-of-book imbalance and size-weighted micro-price."""
    mid = (bid + ask) / 2
    spread = ask - bid
    depth = bid_size + ask_size
    return {
        "mid": mid,
        "spread": spread,
        "spread_bps": spread / mid * 1e4,
        # +1: all resting size on the bid (buyers); -1: all on the ask (sellers).
        "imbalance": (bid_size - ask_size) / depth if depth > 0 else None,
        # Leans toward the side with less size, where the next trade is more likely to move the price.
        "microprice": (bid * ask_size + ask * bid_size) / depth if depth > 0 else None,
    }


def build_quote(ticker: str, raw: Dict, now: Optional[datetime] = None) -> Dict:
    now = now or datetime.now(timezone.utc)
    bid, ask = float(raw.get("bp") or 0), float(raw.get("ap") or 0)
    bid_size, ask_size = float(raw.get("bs") or 0), float(raw.get("as") or 0)
    quoted_at = _parse_time(raw.get("t"))
    market = market_calendar.market_state(now)
    base = {"ticker": ticker, "source": "Alpaca", "feed": settings.ALPACA_FEED, "market_open": market["is_open"]}
    if bid <= 0 or ask <= 0 or ask < bid:
        return {
            **base,
            "available": False,
            "reason": f"No two-sided quote for {ticker} on the {settings.ALPACA_FEED.upper()} feed right now.",
        }
    return {
        **base,
        "available": True,
        "bid": bid,
        "ask": ask,
        "bid_size": bid_size,
        "ask_size": ask_size,
        **measures(bid, ask, bid_size, ask_size),
        "quoted_at": quoted_at.isoformat() if quoted_at else None,
        # Clamped: a server clock a few seconds slow would otherwise give a negative age.
        "age_seconds": max(0.0, (now - quoted_at).total_seconds()) if quoted_at else None,
    }


def get_quote(ticker: str, fetch: Optional[Callable[[str], Dict]] = None) -> Dict:
    """Cached quote for a validated ticker. Errors from the upstream API propagate."""
    if fetch is None and not configured():
        return {"ticker": ticker, "available": False, "reason": NOT_CONFIGURED}
    fetch = fetch or fetch_alpaca_quote
    with _lock:
        cached = _cache.get(ticker)
        if cached and time.monotonic() - cached[0] < QUOTE_TTL_SECONDS:
            return cached[1]
    payload = build_quote(ticker, fetch(ticker))
    with _lock:
        _cache[ticker] = (time.monotonic(), payload)
    return payload
