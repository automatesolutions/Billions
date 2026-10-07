"""
The list of stocks the outlier scan starts from.

Default source: NASDAQ Trader's public symbol directory (no key needed).
Optional source: Alpha Vantage LISTING_STATUS, if ALPHA_VANTAGE_API_KEY is set.
"""

import io
import logging
from typing import List

import pandas as pd
import requests

from api.config import settings

logger = logging.getLogger(__name__)

NASDAQ_TRADER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={key}"


def _clean(symbols) -> List[str]:
    """Keep plain common-stock style symbols: 1-5 letters."""
    return sorted({s for s in symbols if isinstance(s, str) and s.isalpha() and 1 <= len(s) <= 5})


def parse_nasdaq_trader(text: str) -> List[str]:
    # keep_default_na=False: the ticker "NA" must not become NaN.
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    df = df[(df["Test Issue"] == "N") & (df["ETF"] == "N") & (df["Financial Status"] == "N")]
    return _clean(df["Symbol"])


def parse_alpha_vantage(text: str) -> List[str]:
    df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    df = df[(df["exchange"] == "NASDAQ") & (df["status"] == "Active") & (df["assetType"] == "Stock")]
    return _clean(df["symbol"])


def fetch_universe(timeout: int = 30) -> List[str]:
    """Return NASDAQ-listed common stock symbols. Raises RuntimeError if no source works."""
    sources = []
    if settings.ALPHA_VANTAGE_API_KEY:
        sources.append(("Alpha Vantage", ALPHA_VANTAGE_URL.format(key=settings.ALPHA_VANTAGE_API_KEY), parse_alpha_vantage))
    sources.append(("NASDAQ Trader", NASDAQ_TRADER_URL, parse_nasdaq_trader))

    for name, url, parse in sources:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            symbols = parse(response.text)
            if symbols:
                logger.info("Universe: %d symbols from %s", len(symbols), name)
                return symbols
        except Exception as exc:  # noqa: BLE001 - try the next source
            logger.warning("Universe source %s failed: %s", name, exc)
    raise RuntimeError("No symbol source is reachable")
