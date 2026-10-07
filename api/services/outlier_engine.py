"""
Outlier engine.

For each strategy, every liquid stock in the universe gets two trailing returns
(a longer and a shorter window, in trading days). Each return is turned into a
z-score across the universe. A stock is an outlier when either |z| > 2.

`compute_strategy_metrics` is pure (prices in, table out) and unit-tested.
`run_refresh` downloads prices once and stores all strategies.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import delete

from api.database import SessionLocal
from api.models import PerfMetric
from api.services.universe import fetch_universe

logger = logging.getLogger(__name__)

Z_THRESHOLD = 2.0
LIQUIDITY_WINDOW = 20  # trading days used for median dollar volume
MIN_PRICE = 3.0  # skip sub-$3 stocks: prices this low make % moves noisy
MAX_UNIVERSE = 1000  # most liquid names kept after screening
MAX_STALE_DAYS = 3  # drop symbols whose last close lags the market by more than this
DOWNLOAD_BATCH = 400


@dataclass(frozen=True)
class Strategy:
    name: str
    long_days: int
    short_days: int
    long_label: str
    short_label: str
    min_dollar_volume: float  # median daily $ volume over LIQUIDITY_WINDOW


STRATEGIES: Dict[str, Strategy] = {
    "scalp": Strategy("scalp", 21, 5, "1 month", "1 week", 25e6),
    "swing": Strategy("swing", 63, 21, "3 months", "1 month", 15e6),
    "longterm": Strategy("longterm", 252, 126, "1 year", "6 months", 50e6),
}


def trailing_return(close: pd.Series, days: int) -> float:
    """Percent return over the last `days` trading days: P[-1] / P[-1-days] - 1."""
    close = close.dropna()
    if len(close) < days + 1:
        return np.nan
    return float((close.iloc[-1] / close.iloc[-1 - days] - 1.0) * 100.0)


def zscores(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def liquid_symbols(
    close: pd.DataFrame, volume: pd.DataFrame, min_dollar_volume: float, limit: int = MAX_UNIVERSE
) -> List[str]:
    """Symbols above the price and dollar-volume floors, most liquid first, capped at `limit`."""
    dollar_volume = (close * volume).tail(LIQUIDITY_WINDOW).median()
    last_price = close.ffill().iloc[-1]
    ok = (dollar_volume >= min_dollar_volume) & (last_price >= MIN_PRICE)
    return dollar_volume[ok].sort_values(ascending=False).head(limit).index.tolist()


def fresh_symbols(close: pd.DataFrame, max_stale_days: int = MAX_STALE_DAYS) -> List[str]:
    """Symbols whose last valid close is within `max_stale_days` sessions of the newest row."""
    recent = close.tail(max_stale_days + 1).notna().any()
    return recent[recent].index.tolist()


def compute_strategy_metrics(close: pd.DataFrame, volume: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    """
    close, volume: date-indexed frames, one column per symbol.
    Returns a frame indexed by symbol with metric_x (long %), metric_y (short %), z_x, z_y, is_outlier.
    """
    fresh = fresh_symbols(close)
    symbols = liquid_symbols(close[fresh], volume[fresh], strategy.min_dollar_volume)
    rows = {}
    for symbol in symbols:
        series = close[symbol].dropna()
        long_ret = trailing_return(series, strategy.long_days)
        short_ret = trailing_return(series, strategy.short_days)
        if np.isfinite(long_ret) and np.isfinite(short_ret):
            rows[symbol] = (long_ret, short_ret)

    df = pd.DataFrame.from_dict(rows, orient="index", columns=["metric_x", "metric_y"])
    if len(df) < 3:
        return df.assign(z_x=pd.Series(dtype=float), z_y=pd.Series(dtype=float), is_outlier=pd.Series(dtype=bool))

    df["z_x"] = zscores(df["metric_x"])
    df["z_y"] = zscores(df["metric_y"])
    df["is_outlier"] = (df["z_x"].abs() > Z_THRESHOLD) | (df["z_y"].abs() > Z_THRESHOLD)
    return df


def download_prices(symbols: List[str], period: str = "2y") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bulk daily download from Yahoo Finance. Returns (close, volume) frames."""
    closes, volumes = [], []
    for start in range(0, len(symbols), DOWNLOAD_BATCH):
        batch = symbols[start : start + DOWNLOAD_BATCH]
        try:
            data = yf.download(
                batch, period=period, interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False
            )
        except Exception as exc:  # noqa: BLE001 - one bad batch should not stop the scan
            logger.warning("Price batch %d failed: %s", start // DOWNLOAD_BATCH + 1, exc)
            continue
        if data.empty:
            continue
        closes.append(data.xs("Close", axis=1, level=1))
        volumes.append(data.xs("Volume", axis=1, level=1))
        time.sleep(1)  # be polite to Yahoo between batches

    if not closes:
        raise RuntimeError("No price data downloaded")
    close = pd.concat(closes, axis=1).sort_index()
    volume = pd.concat(volumes, axis=1).sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    volume = volume.loc[:, ~volume.columns.duplicated()]
    return close, volume


def store_metrics(strategy: str, df: pd.DataFrame, price_date, computed_at: datetime) -> None:
    with SessionLocal() as session:
        session.execute(delete(PerfMetric).where(PerfMetric.strategy == strategy))
        session.add_all(
            PerfMetric(
                strategy=strategy,
                symbol=symbol,
                metric_x=float(row.metric_x),
                metric_y=float(row.metric_y),
                z_x=float(row.z_x),
                z_y=float(row.z_y),
                is_outlier=bool(row.is_outlier),
                price_date=price_date,
                inserted=computed_at,
            )
            for symbol, row in df.iterrows()
        )
        session.commit()


def run_refresh(strategies: Optional[List[str]] = None) -> Dict[str, int]:
    """Download the universe once and store metrics for each strategy. Returns {strategy: outlier count}."""
    strategies = strategies or list(STRATEGIES)
    symbols = fetch_universe()
    close, volume = download_prices(symbols)
    price_date = close.index[-1].date()
    computed_at = datetime.now(timezone.utc)

    counts = {}
    for name in strategies:
        df = compute_strategy_metrics(close, volume, STRATEGIES[name])
        if df.empty:
            logger.warning("No metrics for %s; keeping previous results", name)
            continue
        store_metrics(name, df, price_date, computed_at)
        counts[name] = int(df["is_outlier"].sum())
        logger.info("Stored %d rows for %s (%d outliers)", len(df), name, counts[name])
    return counts
