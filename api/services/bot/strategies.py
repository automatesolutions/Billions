"""
The strategy interface and strategy 1, the moving-average crossover.

A strategy only looks at prices and says what it would do and why (a Signal).
It never places orders, reads the database, or calls the network: the runner
checks risk limits and talks to the broker. That split keeps every strategy
small and testable with made-up prices.

To add a strategy: subclass BarStrategy (polls bars) or StreamStrategy (reacts to
live trades), give it a pydantic params model, and register both in runner.STRATEGIES.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from api.services.prices import normalize_ticker

# Signal actions: buy and sell go to the risk check and maybe the broker; hold and skip are only logged.
ACTIONS = ("buy", "sell", "hold", "skip")


@dataclass
class Signal:
    action: str
    symbol: str
    reason: str
    price: Optional[float] = None
    kind: str = "entry"  # entry opens a position, exit closes one
    details: Dict[str, Any] = field(default_factory=dict)
    ref: Any = None  # strategy-owned object the runner reports back on (lead-lag trades)

    def __post_init__(self):
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown signal action {self.action}")


def clean_tickers(values: List[str]) -> List[str]:
    out = []
    for value in values:
        symbol = normalize_ticker(value)
        if symbol is None:
            raise ValueError(f"'{value}' isn't a valid ticker.")
        if symbol not in out:
            out.append(symbol)
    return out


class Strategy(ABC):
    name: ClassVar[str]
    label: ClassVar[str]
    feed: ClassVar[str]  # "bars" or "stream"

    def __init__(self, params: BaseModel):
        self.params = params

    @abstractmethod
    def symbols(self) -> List[str]:
        """Every ticker this strategy needs prices for."""


class BarStrategy(Strategy):
    feed = "bars"

    @abstractmethod
    def on_bars(self, symbol: str, closes: pd.Series, position_qty: float) -> Signal:
        """Decide from completed bars (oldest first) and the current position size."""


class StreamStrategy(Strategy):
    feed = "stream"

    @abstractmethod
    def on_tick(self, tick) -> List[Signal]:
        """React to one live trade price."""

    def check_time_exits(self, now: datetime) -> List[Signal]:
        return []


# ---------------------------------------------------------------- strategy 1: moving-average crossover


class MACrossoverParams(BaseModel):
    tickers: List[str] = Field(min_length=1, max_length=10)
    short_window: int = Field(10, ge=2, le=200, description="Bars in the fast average")
    long_window: int = Field(30, ge=3, le=400, description="Bars in the slow average")
    timespan: Literal["day", "minute"] = "day"
    poll_seconds: int = Field(60, ge=15, le=3600, description="How often to check for new bars")

    @field_validator("tickers")
    @classmethod
    def _tickers(cls, value):
        return clean_tickers(value)

    @model_validator(mode="after")
    def _windows(self):
        if self.short_window >= self.long_window:
            raise ValueError("The short window must be smaller than the long window.")
        return self


class MovingAverageCrossover(BarStrategy):
    """
    Buy when the short average crosses above the long one; sell the position when it crosses below.
    Only a fresh cross (between the last two bars) triggers anything, so restarting the bot
    halfway through a trend doesn't buy late.
    """

    name = "ma_crossover"
    label = "Moving-average crossover"
    params: MACrossoverParams

    def symbols(self) -> List[str]:
        return list(self.params.tickers)

    def bars_needed(self) -> int:
        return self.params.long_window + 1

    def on_bars(self, symbol: str, closes: pd.Series, position_qty: float) -> Signal:
        p = self.params
        closes = closes.dropna()
        if len(closes) < self.bars_needed():
            return Signal("hold", symbol, f"Not enough bars yet: have {len(closes)}, need {self.bars_needed()}.")
        short = closes.rolling(p.short_window).mean()
        long = closes.rolling(p.long_window).mean()
        before, now = short.iloc[-2] - long.iloc[-2], short.iloc[-1] - long.iloc[-1]
        price = float(closes.iloc[-1])
        details = {"short_ma": float(short.iloc[-1]), "long_ma": float(long.iloc[-1]), "close": price}
        averages = f"{p.short_window}-bar average {short.iloc[-1]:.2f}, {p.long_window}-bar average {long.iloc[-1]:.2f}"

        if before <= 0 < now:
            if position_qty > 0:
                return Signal(
                    "hold",
                    symbol,
                    f"Crossed above, but you already hold {position_qty:g} shares. {averages}.",
                    price,
                    details=details,
                )
            return Signal("buy", symbol, f"Short average crossed above the long one. {averages}.", price, "entry", details)
        if before >= 0 > now:
            if position_qty > 0:
                return Signal("sell", symbol, f"Short average crossed below the long one. {averages}.", price, "exit", details)
            return Signal("hold", symbol, f"Crossed below, with no position to sell. {averages}.", price, details=details)
        side = "above" if now > 0 else "below"
        return Signal(
            "hold", symbol, f"No new cross. The short average is {side} the long one. {averages}.", price, details=details
        )


def bar_time(value) -> str:
    """A bar's timestamp as ISO 8601 (with its offset), or the value as text for a plain index."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def ma_chart(closes: pd.Series, short_window: int, long_window: int, points: int = 120) -> Dict[str, Any]:
    """
    The last `points` bars with both averages, and every cross inside them, for the Bot page chart.
    Crosses use the same rule as on_bars (before <= 0 < now is up, before >= 0 > now is down),
    so a marker on the chart is exactly a bar where the strategy would act.
    """
    closes = closes.dropna()
    short = closes.rolling(short_window).mean()
    long = closes.rolling(long_window).mean()
    gap = short - long
    crosses = []
    for i in range(1, len(closes)):
        before, now = gap.iloc[i - 1], gap.iloc[i]
        if pd.isna(before) or pd.isna(now):
            continue
        if before <= 0 < now or before >= 0 > now:
            crosses.append({"i": i, "direction": "up" if now > 0 else "down"})
    start = max(0, len(closes) - points)

    def value(series, i):
        v = series.iloc[i]
        return None if pd.isna(v) else round(float(v), 4)

    bars = [
        {"t": bar_time(closes.index[i]), "close": value(closes, i), "short": value(short, i), "long": value(long, i)}
        for i in range(start, len(closes))
    ]
    marks = [
        {"t": bar_time(closes.index[c["i"]]), "direction": c["direction"], "price": value(closes, c["i"])}
        for c in crosses
        if c["i"] >= start
    ]
    last_gap = value(gap, len(closes) - 1) if len(closes) else None
    return {"bars": bars, "crosses": marks, "gap": last_gap}
