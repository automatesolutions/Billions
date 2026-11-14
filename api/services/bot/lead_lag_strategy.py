"""
Strategy 2, lead-lag: trade a follower right after its leader moves.

Idea: some tickers react to news a few seconds after a related one (NVDA, then AMD;
SPY, then QQQ). When the leader moves more than `threshold_pct` within
`lookback_seconds` and the follower hasn't moved yet, buy the follower on an up
move (or short it on a down move, only if allow_short is on).

Exits, whichever comes first:
- take profit: the follower moved `take_profit_pct` our way from the fill,
- stop loss: it moved `stop_loss_pct` against us,
- time exit: `max_wait_seconds` after the fill, the follower didn't follow.

Cooldown: once a leader move triggers (a trade or a logged skip) for a follower,
the same follower stays quiet for `cooldown_seconds`, so one move counts once.

Windows use exchange trade times (Tick.ts). The strategy stamps `signal_at` with
the wall clock, so leader move -> signal -> fill delays can be compared later.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from api.services.bot.strategies import Signal, StreamStrategy, clean_tickers
from api.services.prices import normalize_ticker


class LeadLagParams(BaseModel):
    leader: str
    followers: List[str] = Field(min_length=1, max_length=5)
    threshold_pct: float = Field(0.3, gt=0, le=10, description="Leader move that triggers, in percent")
    lookback_seconds: float = Field(5, gt=0, le=600, description="Window the leader's move is measured over")
    max_wait_seconds: float = Field(30, gt=0, le=3600, description="Time exit if the follower doesn't follow")
    follower_max_move_pct: float = Field(0.1, ge=0, le=10, description="The follower 'hasn't moved' below this")
    take_profit_pct: float = Field(0.3, gt=0, le=20)
    stop_loss_pct: float = Field(0.2, gt=0, le=20)
    cooldown_seconds: float = Field(60, ge=0, le=3600)
    stale_seconds: float = Field(60, gt=0, le=3600, description="Ignore a follower whose last trade is older")
    allow_short: bool = False
    source: Literal["alpaca", "polygon"] = "alpaca"

    @field_validator("leader")
    @classmethod
    def _leader(cls, value):
        symbol = normalize_ticker(value)
        if symbol is None:
            raise ValueError(f"'{value}' isn't a valid ticker.")
        return symbol

    @field_validator("followers")
    @classmethod
    def _followers(cls, value):
        return clean_tickers(value)

    @model_validator(mode="after")
    def _distinct(self):
        if self.leader in self.followers:
            raise ValueError("The leader can't also be a follower.")
        return self


class RollingPrices:
    """Trade prices for one ticker, keeping only the last `keep_seconds`."""

    def __init__(self, keep_seconds: float):
        self.keep_seconds = keep_seconds
        self._items: Deque[Tuple[datetime, float]] = deque()

    def add(self, ts: datetime, price: float) -> None:
        if self._items and ts < self._items[-1][0]:
            ts = self._items[-1][0]  # out-of-order print: keep the window monotonic
        self._items.append((ts, price))
        while self._items and (ts - self._items[0][0]).total_seconds() > self.keep_seconds:
            self._items.popleft()

    @property
    def last(self) -> Optional[Tuple[datetime, float]]:
        return self._items[-1] if self._items else None

    def move_pct(self, lookback_seconds: float) -> Optional[Tuple[float, datetime]]:
        """(percent change from the oldest price within the lookback to the latest, time of that oldest price)."""
        if not self._items:
            return None
        last_ts, last_price = self._items[-1]
        for ts, price in self._items:
            if (last_ts - ts).total_seconds() <= lookback_seconds:
                return (last_price / price - 1) * 100, ts
        return 0.0, last_ts


@dataclass
class LeadLagTrade:
    follower: str
    leader: str
    side: str  # buy (long) or sell (short)
    leader_move_pct: float
    leader_move_started_at: datetime
    leader_move_at: datetime  # exchange time of the leader trade that crossed the threshold
    leader_price: float
    follower_price_at_signal: float
    follower_move_pct: float
    signal_at: datetime  # wall clock when we decided
    entry_reason: str
    status: str = "pending"  # pending -> open -> (exit pending) -> closed
    entry_price: Optional[float] = None
    entry_filled_at: Optional[datetime] = None
    exit_pending: bool = False
    last_price: Optional[float] = None
    qty: int = 0
    trade_id: Optional[int] = None  # database row, set by the runner

    def pnl_pct(self, price: float) -> float:
        base = self.entry_price or self.follower_price_at_signal
        move = (price / base - 1) * 100
        return move if self.side == "buy" else -move


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LeadLagStrategy(StreamStrategy):
    name = "lead_lag"
    label = "Lead-lag"
    params: LeadLagParams

    def __init__(self, params: LeadLagParams, clock: Optional[Callable[[], datetime]] = None):
        super().__init__(params)
        keep = max(params.lookback_seconds, params.stale_seconds) + 5
        self.windows: Dict[str, RollingPrices] = {s: RollingPrices(keep) for s in self.symbols()}
        self.trades: Dict[str, LeadLagTrade] = {}
        self.last_trigger: Dict[str, datetime] = {}
        self._clock = clock or _utcnow

    def symbols(self) -> List[str]:
        return [self.params.leader, *self.params.followers]

    # ------------------------------------------------------------ ticks

    def on_tick(self, tick) -> List[Signal]:
        window = self.windows.get(tick.symbol)
        if window is None or not tick.price or tick.price <= 0:
            return []
        window.add(tick.ts, tick.price)
        if tick.symbol == self.params.leader:
            return self._check_entries(tick)
        trade = self.trades.get(tick.symbol)
        if trade:
            trade.last_price = tick.price
        return self._check_price_exit(tick.symbol, tick.price)

    def _check_entries(self, tick) -> List[Signal]:
        p = self.params
        measured = self.windows[p.leader].move_pct(p.lookback_seconds)
        if measured is None:
            return []
        move, started_at = measured
        if abs(move) < p.threshold_pct:
            return []
        out = []
        for follower in p.followers:
            last = self.last_trigger.get(follower)
            if last is not None and (tick.ts - last).total_seconds() < p.cooldown_seconds:
                continue  # same move (or too soon after the last one): stay quiet
            self.last_trigger[follower] = tick.ts
            out.append(self._entry(follower, move, started_at, tick))
        return out

    def _entry(self, follower: str, move: float, started_at: datetime, tick) -> Signal:
        p = self.params
        secs = (tick.ts - started_at).total_seconds()
        said = f"{p.leader} moved {move:+.2f}% in {secs:.1f}s"
        details = {
            "leader": p.leader,
            "leader_move_pct": round(move, 4),
            "leader_move_started_at": started_at.isoformat(),
            "leader_move_at": tick.ts.isoformat(),
            "leader_price": tick.price,
        }
        if follower in self.trades:
            return Signal("skip", follower, f"{said}, but a {follower} trade is already open.", details=details)
        last = self.windows[follower].last
        if last is None:
            return Signal("skip", follower, f"{said}, but there's no {follower} price yet.", details=details)
        age = (tick.ts - last[0]).total_seconds()
        if age > p.stale_seconds:
            return Signal(
                "skip", follower, f"{said}, but {follower}'s last trade is {age:.0f}s old.", last[1], details=details
            )
        follower_move, _ = self.windows[follower].move_pct(p.lookback_seconds)
        details["follower_move_pct"] = round(follower_move, 4)
        details["follower_price"] = last[1]
        if abs(follower_move) >= p.follower_max_move_pct:
            return Signal(
                "skip",
                follower,
                f"{said}, but {follower} already moved {follower_move:+.2f}%, so the lag is gone.",
                last[1],
                details=details,
            )
        if move < 0 and not p.allow_short:
            return Signal("skip", follower, f"{said} down. Shorting is off, so no trade.", last[1], details=details)

        side = "buy" if move > 0 else "sell"
        signal_at = self._clock()
        verb = "buy" if side == "buy" else "short"
        reason = f"{said}; {follower} moved {follower_move:+.2f}%, so {verb} {follower} at about ${last[1]:,.2f}."
        trade = LeadLagTrade(
            follower=follower,
            leader=p.leader,
            side=side,
            leader_move_pct=move,
            leader_move_started_at=started_at,
            leader_move_at=tick.ts,
            leader_price=tick.price,
            follower_price_at_signal=last[1],
            follower_move_pct=follower_move,
            signal_at=signal_at,
            entry_reason=reason,
            last_price=last[1],
        )
        self.trades[follower] = trade
        details["signal_at"] = signal_at.isoformat()
        return Signal(side, follower, reason, last[1], "entry", details, ref=trade)

    # ------------------------------------------------------------ exits

    def _exit(self, trade: LeadLagTrade, price: float, reason: str) -> Signal:
        trade.exit_pending = True
        action = "sell" if trade.side == "buy" else "buy"
        return Signal(action, trade.follower, reason, price, "exit", {"pnl_pct": round(trade.pnl_pct(price), 4)}, ref=trade)

    def _check_price_exit(self, symbol: str, price: float) -> List[Signal]:
        p = self.params
        trade = self.trades.get(symbol)
        if not trade or trade.status != "open" or trade.exit_pending:
            return []
        pnl = trade.pnl_pct(price)
        if pnl >= p.take_profit_pct:
            return [self._exit(trade, price, f"Take profit: {symbol} is {pnl:+.2f}% from the ${trade.entry_price:,.2f} fill.")]
        if pnl <= -p.stop_loss_pct:
            return [self._exit(trade, price, f"Stop loss: {symbol} is {pnl:+.2f}% from the ${trade.entry_price:,.2f} fill.")]
        return []

    def check_time_exits(self, now: datetime) -> List[Signal]:
        p = self.params
        out = []
        for trade in list(self.trades.values()):
            if trade.status != "open" or trade.exit_pending or trade.entry_filled_at is None:
                continue
            held = (now - trade.entry_filled_at).total_seconds()
            if held >= p.max_wait_seconds:
                price = trade.last_price or trade.entry_price
                out.append(
                    self._exit(
                        trade,
                        price,
                        f"Time exit: {trade.follower} didn't follow within {p.max_wait_seconds:g}s "
                        f"({trade.pnl_pct(price):+.2f}% from the fill).",
                    )
                )
        return out

    # ------------------------------------------------------------ runner callbacks

    def on_entry_filled(self, symbol: str, price: float, filled_at: datetime) -> None:
        trade = self.trades.get(symbol)
        if trade:
            trade.entry_price, trade.entry_filled_at, trade.status = price, filled_at, "open"

    def on_entry_failed(self, symbol: str) -> None:
        self.trades.pop(symbol, None)

    def on_exit_filled(self, symbol: str) -> None:
        self.trades.pop(symbol, None)

    def on_exit_failed(self, symbol: str) -> None:
        trade = self.trades.get(symbol)
        if trade:
            trade.exit_pending = False  # try again on the next tick
