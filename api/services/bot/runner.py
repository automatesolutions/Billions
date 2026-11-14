"""
The paper bot's background loop.

One bot runs at a time, in a daemon thread like the app's other schedulers:

    prices -> strategy.signal -> risk check -> paper order -> decision log

- Strategy 1 (bars): every `poll_seconds`, load Polygon bars for each ticker and ask
  the strategy. Holds are logged when their reason changes, not every poll.
- Strategy 2 (stream): a WebSocket thread puts trades on a queue; this thread takes
  them off, asks the strategy, checks time exits, and polls Alpaca for fills.
- Orders only go out while the market is open (market_calendar).
- Kill switch: the risk manager refuses everything first, then the loop stops, then
  open orders are canceled (and positions closed, if you ask).

Every lead-lag trade records leader move -> signal -> order -> fill times, so you
can see how much of a few-second edge is left after latency.
"""

import logging
import queue
import re
import threading
import uuid
from datetime import datetime, time, timedelta, timezone
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from api.config import settings
from api.database import SessionLocal
from api.models import BotDecision, BotTrade
from api.services import market_calendar, polygon
from api.services.bot.lead_lag_strategy import LeadLagParams, LeadLagStrategy
from api.services.bot.risk import RiskLimits, RiskManager
from api.services.bot.strategies import MACrossoverParams, MovingAverageCrossover, Signal, bar_time, ma_chart
from api.services.bot.streams import make_stream, parse_rfc3339
from api.services.paper_broker import (
    BrokerError,
    BrokerNotConfigured,
    LiveTradingRefused,
    PaperBroker,
    account_summary,
    order_summary,
    position_summary,
)

logger = logging.getLogger(__name__)

STRATEGIES = {
    MovingAverageCrossover.name: (MovingAverageCrossover, MACrossoverParams),
    LeadLagStrategy.name: (LeadLagStrategy, LeadLagParams),
}
DEAD_ORDER = {"canceled", "expired", "rejected", "suspended", "done_for_day"}
RECENT_DECISIONS = 100
RECENT_TRADES = 50
ACCOUNT_TTL_SECONDS = 3.0
POSITIONS_TTL_SECONDS = 5.0
BARS_MAX_WAIT_SECONDS = 20  # never block on Polygon's rate limit for longer than this

CAVEAT_PAPER = (
    "Paper fills aren't real fills. Alpaca fills paper orders against the quote with no queue position or "
    "market impact, so real slippage would be worse."
)
CAVEAT_IEX = (
    "The free Alpaca stream shows trades from the IEX exchange only, a small share of all volume, "
    "so a move can show up late or not at all."
)
CAVEAT_DELAYED = (
    "Polygon's free plan is delayed by 15 minutes, so moving-average signals act on prices that are 15 minutes old."
)


def _default_limits() -> RiskLimits:
    return RiskLimits(
        max_position_dollars=settings.BOT_MAX_POSITION_DOLLARS, max_orders_per_day=settings.BOT_MAX_ORDERS_PER_DAY
    )


class BotConfig(BaseModel):
    strategy: Literal["ma_crossover", "lead_lag"]
    ma_crossover: Optional[MACrossoverParams] = None
    lead_lag: Optional[LeadLagParams] = None
    risk: RiskLimits = Field(default_factory=_default_limits)

    @model_validator(mode="after")
    def _has_params(self):
        if getattr(self, self.strategy) is None:
            raise ValueError(f"Add settings for the {self.strategy} strategy.")
        return self

    @property
    def params(self):
        return getattr(self, self.strategy)


class AlreadyRunning(RuntimeError):
    pass


# ---------------------------------------------------------------- time helpers


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Aware -> naive UTC, for the database."""
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def iso(dt: Optional[datetime]) -> Optional[str]:
    """Naive UTC from the database -> ISO 8601 with the offset."""
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _ms(later: Optional[datetime], earlier: Optional[datetime]) -> Optional[float]:
    if later is None or earlier is None:
        return None
    return round((later - earlier).total_seconds() * 1000, 1)


# ---------------------------------------------------------------- rows to API shapes


def decision_dict(row: BotDecision) -> Dict:
    return {
        "id": row.id,
        "at": iso(row.created_at),
        "strategy": row.strategy,
        "symbol": row.symbol,
        "action": row.action,
        "status": row.status,
        "reason": row.reason,
        "price": row.price,
        "qty": row.qty,
        "order_id": row.order_id,
        "details": row.details or {},
    }


def trade_timings(row: BotTrade) -> Dict:
    slippage = None
    if row.entry_price and row.follower_price_at_signal:
        move = (row.entry_price / row.follower_price_at_signal - 1) * 100
        slippage = round(move if row.side == "buy" else -move, 4)  # positive: the fill was worse than the signal price
    return {
        "leader_to_signal_ms": _ms(row.signal_at, row.leader_move_at),
        "signal_to_submit_ms": _ms(row.entry_submitted_at, row.signal_at),
        "submit_to_fill_ms": _ms(row.entry_filled_at, row.entry_submitted_at),
        "leader_to_fill_ms": _ms(row.entry_filled_at, row.leader_move_at),
        "slippage_pct": slippage,
    }


def trade_dict(row: BotTrade) -> Dict:
    return {
        "id": row.id,
        "leader": row.leader,
        "follower": row.follower,
        "side": row.side,
        "qty": row.qty,
        "status": row.status,
        "leader_move_pct": row.leader_move_pct,
        "leader_move_started_at": iso(row.leader_move_started_at),
        "leader_move_at": iso(row.leader_move_at),
        "leader_price": row.leader_price,
        "follower_price_at_signal": row.follower_price_at_signal,
        "follower_move_pct": row.follower_move_pct,
        "signal_at": iso(row.signal_at),
        "entry_reason": row.entry_reason,
        "entry_submitted_at": iso(row.entry_submitted_at),
        "entry_filled_at": iso(row.entry_filled_at),
        "entry_price": row.entry_price,
        "exit_reason": row.exit_reason,
        "exit_signal_at": iso(row.exit_signal_at),
        "exit_filled_at": iso(row.exit_filled_at),
        "exit_price": row.exit_price,
        "pnl_pct": row.pnl_pct,
        "timings": trade_timings(row),
    }


# ---------------------------------------------------------------- bars for strategy 1


def polygon_closes(symbol: str, params: MACrossoverParams, now: datetime) -> pd.Series:
    """Completed bars only: today's daily bar and the minute bar in progress are dropped."""
    today = now.astimezone(market_calendar.NEW_YORK).date()
    if params.timespan == "day":
        closes = polygon.get_daily_bars(symbol, days=params.long_window + 20, max_wait=BARS_MAX_WAIT_SECONDS)["close"]
        if not closes.empty and closes.index[-1].tz_convert(market_calendar.NEW_YORK).date() == today:
            if now < market_calendar.session_close(today):
                closes = closes.iloc[:-1]
        return closes
    closes = polygon.get_intraday_bars(symbol, today - timedelta(days=7), today, "minute", max_wait=BARS_MAX_WAIT_SECONDS)[
        "close"
    ]
    if not closes.empty and closes.index[-1] + pd.Timedelta(minutes=1) > pd.Timestamp(now):
        closes = closes.iloc[:-1]
    return closes


# ---------------------------------------------------------------- the runner


class BotRunner:
    def __init__(
        self,
        broker_factory: Callable[[], PaperBroker] = PaperBroker,
        bars_loader: Callable = polygon_closes,
        stream_factory: Callable = make_stream,
        session_factory=SessionLocal,
        now: Callable[[], datetime] = _utcnow,
        market_open: Optional[Callable[[datetime], bool]] = None,
        fill_poll_seconds: float = 1.0,
        fill_timeout_seconds: float = 60.0,
    ):
        self._broker_factory = broker_factory
        self._bars_loader = bars_loader
        self._stream_factory = stream_factory
        self._session_factory = session_factory
        self._now = now
        self._market_open = market_open or (lambda when: market_calendar.market_state(when)["is_open"])
        self._fill_poll = timedelta(seconds=fill_poll_seconds)
        self._fill_timeout = timedelta(seconds=fill_timeout_seconds)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reset()

    def _reset(self) -> None:
        self.config: Optional[BotConfig] = None
        self.strategy = None
        self.risk: Optional[RiskManager] = None
        self.broker: Optional[PaperBroker] = None
        self.stream = None
        self.started_at: Optional[datetime] = None
        self.stopped_at: Optional[datetime] = None
        self.stopped_reason: Optional[str] = None
        self.last_error: Optional[str] = None
        self._market_note: Optional[str] = None
        self._last_quiet: Dict[str, str] = {}
        self._pending: Dict[str, Dict] = {}
        self._bars: Dict[str, pd.Series] = {}  # the last bars each ticker was decided on, for the chart
        self._positions_cache = (datetime.min.replace(tzinfo=timezone.utc), {})
        self._account_cache = (datetime.min.replace(tzinfo=timezone.utc), None)

    @property
    def running(self) -> bool:
        return self.started_at is not None and not self._stop.is_set()

    # ------------------------------------------------------------ lifecycle

    def start(self, config: BotConfig, run_thread: bool = True) -> None:
        """Build everything first, so a bad URL, missing keys, or a refused stream fail here, not in the thread."""
        with self._lock:
            if self.running:
                raise AlreadyRunning("The bot is already running. Stop it first.")
            broker = self._broker_factory()
            cls, _ = STRATEGIES[config.strategy]
            # Stream strategies stamp signal times: they must use the same clock as the runner's order and fill times.
            strategy = cls(config.params, clock=self._now) if cls.feed == "stream" else cls(config.params)
            ticks: "queue.Queue" = queue.Queue(maxsize=10000)
            stream = None
            if strategy.feed == "stream":
                stream = self._stream_factory(config.params.source, strategy.symbols(), self._enqueue(ticks))
            self._reset()
            self.config, self.strategy, self.broker, self.stream = config, strategy, broker, stream
            self.risk = RiskManager(config.risk, orders_today=self._orders_placed_today())
            self.started_at = self._now()
            self._stop.clear()
            self._log(
                None,
                "start",
                f"Started {strategy.label} on {', '.join(strategy.symbols())}.",
                "info",
                details=config.model_dump(),
            )
            if run_thread:
                target = self._run_bars if strategy.feed == "bars" else (lambda: self._run_stream(ticks))
                self._thread = threading.Thread(target=self._guarded(target), name="paper-bot", daemon=True)
                self._thread.start()

    def stop(self, reason: str = "Stopped from the Bot page.") -> bool:
        with self._lock:
            was_running = self.running
            self._stop.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(10)
        if was_running:
            self.stopped_at, self.stopped_reason = self._now(), reason
            self._log(None, "stop", reason, "info")
        return was_running

    def kill(self, close_positions: bool = False, reason: str = "Kill switch pressed.") -> Dict:
        if self.risk:
            self.risk.kill(reason)  # first: anything mid-flight is refused from here on
        self.stop(reason)
        broker = self.broker or self._broker_factory()
        cancelled = broker.cancel_all_orders()
        closed = broker.close_all_positions(cancel_orders=True) if close_positions else 0
        self._pending.clear()
        with self._session_factory() as db:
            open_rows = db.query(BotTrade).filter(BotTrade.status.in_(("pending", "open", "closing"))).all()
            for row in open_rows:
                row.status = "killed"
                row.exit_reason = reason + (" Positions closed." if close_positions else " Position left open.")
            db.commit()
        message = f"{reason} Canceled {cancelled} open orders" + (
            f" and closed {closed} positions." if close_positions else "."
        )
        self._log(None, "kill", message, "info", details={"cancelled_orders": cancelled, "closed_positions": closed})
        self._account_cache = (datetime.min.replace(tzinfo=timezone.utc), None)
        return {"cancelled_orders": cancelled, "closed_positions": closed, "message": message}

    def _guarded(self, target):
        def run():
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - record and stop, never die silently
                logger.exception("Paper bot loop failed")
                self.last_error = f"The bot stopped after an unexpected error ({type(exc).__name__})."
                self._log(None, "error", self.last_error, "failed")
                self._stop.set()
                self.stopped_at, self.stopped_reason = self._now(), self.last_error

        return run

    @staticmethod
    def _enqueue(ticks: "queue.Queue"):
        def put(tick):
            try:
                ticks.put_nowait(tick)
            except queue.Full:
                logger.warning("Tick queue full; dropping a trade")

        return put

    # ------------------------------------------------------------ loops

    def _check_market(self, now: datetime) -> bool:
        is_open = self._market_open(now)
        note = "open" if is_open else "closed"
        if note != self._market_note:
            self._market_note = note
            text = (
                "Market is open. Watching prices."
                if is_open
                else "Market is closed. The bot waits for the next session and places no orders."
            )
            self._log(None, "info", text, "info")
        return is_open

    def _run_bars(self) -> None:
        while not self._stop.is_set():
            self.run_bars_once()
            self._stop.wait(self.strategy.params.poll_seconds)

    def run_bars_once(self) -> None:
        now = self._now()
        if not self._check_market(now):
            return
        try:
            positions = self._positions(now, force=True)
        except BrokerError as exc:
            self._log(None, "error", f"Couldn't read positions: {exc}", "failed")
            return
        for symbol in self.strategy.symbols():
            if self._stop.is_set():
                return
            try:
                closes = self._bars_loader(symbol, self.strategy.params, now)
            except polygon.PolygonError as exc:
                self._quiet(symbol, "skip", f"No bars for {symbol}: {exc}", "skipped")
                continue
            self._bars[symbol] = closes
            position = positions.get(symbol)
            signal = self.strategy.on_bars(symbol, closes, float(position["qty"] or 0) if position else 0.0)
            if not closes.empty:
                signal.details["bar_at"] = bar_time(closes.index[-1])  # places the order on the chart
            self._act(signal, position, now)

    def _run_stream(self, ticks: "queue.Queue") -> None:
        self.stream.start()
        try:
            while not self._stop.is_set():
                try:
                    tick = ticks.get(timeout=0.25)
                except queue.Empty:
                    tick = None
                self.process(tick)
        finally:
            self.stream.stop()

    def process(self, tick=None, now: Optional[datetime] = None) -> None:
        """One step of the stream loop. Tests call this directly with simulated ticks and times."""
        now = now or self._now()
        self.poll_fills(now)  # orders placed just before the close can still fill
        if not self._check_market(now):
            return
        signals: List[Signal] = self.strategy.on_tick(tick) if tick is not None else []
        signals += self.strategy.check_time_exits(now)
        for signal in signals:
            self._act(signal, None, now)

    # ------------------------------------------------------------ acting on signals

    def _quiet(self, symbol, action, reason, status, price=None, details=None) -> None:
        """Log a hold or skip only when its reason changes (numbers ignored), so the log stays readable."""
        key = f"{symbol}:{action}"
        shape = re.sub(r"[-+]?\d[\d,.]*", "#", reason)
        if self._last_quiet.get(key) == shape:
            return
        self._last_quiet[key] = shape
        self._log(symbol, action, reason, status, price=price, details=details)

    def _act(self, signal: Signal, position: Optional[Dict], now: datetime) -> None:
        if signal.action == "hold":
            self._quiet(signal.symbol, "hold", signal.reason, "held", signal.price, signal.details)
            return
        if signal.action == "skip":
            # Lead-lag skips are already one per leader move (cooldown), so each is logged.
            self._log(signal.symbol, "skip", signal.reason, "skipped", signal.price, details=signal.details)
            return
        self._last_quiet.pop(f"{signal.symbol}:hold", None)
        if signal.kind == "entry":
            self._enter(signal, position, now)
        else:
            self._exit(signal, position, now)

    def _submit(self, symbol: str, side: str, qty: float, tag: str) -> Dict:
        return self.broker.place_market_order(symbol, side, qty, client_order_id=f"billions-{tag}-{uuid.uuid4().hex[:12]}")

    def _enter(self, signal: Signal, position: Optional[Dict], now: datetime) -> None:
        trade = signal.ref
        if position is None and trade is not None:
            position = self._positions(now).get(signal.symbol)
        existing = float(position.get("market_value") or 0) if position else 0.0
        check = self.risk.size_entry(signal.price or 0, existing)
        details = {**signal.details, "risk": check.reason}
        if not check.allowed:
            if trade is not None:
                self.strategy.on_entry_failed(signal.symbol)
            self._log(
                signal.symbol,
                signal.action,
                f"{signal.reason} Blocked: {check.reason}",
                "blocked",
                signal.price,
                details=details,
            )
            return
        try:
            order = self._submit(signal.symbol, signal.action, check.qty, "entry")
        except (BrokerError, ValueError) as exc:
            if trade is not None:
                self.strategy.on_entry_failed(signal.symbol)
            self._log(
                signal.symbol,
                signal.action,
                f"{signal.reason} Order failed: {exc}",
                "failed",
                signal.price,
                check.qty,
                details=details,
            )
            return
        submitted = self._now()
        self.risk.record_order()
        if trade is not None:
            trade.qty = check.qty
            trade.trade_id = self._insert_trade(trade, order.get("id"), submitted)
            self._pending[order["id"]] = {
                "kind": "entry",
                "symbol": signal.symbol,
                "trade": trade,
                "submitted_at": submitted,
                "next_poll": now,
            }
            details["timings"] = {
                "leader_to_signal_ms": _ms(trade.signal_at, trade.leader_move_at),
                "signal_to_submit_ms": _ms(submitted, trade.signal_at),
            }
        self._log(
            signal.symbol,
            signal.action,
            f"{signal.reason} {check.reason}",
            "placed",
            signal.price,
            check.qty,
            order.get("id"),
            details,
        )

    def _exit(self, signal: Signal, position: Optional[Dict], now: datetime) -> None:
        trade = signal.ref
        check = self.risk.allow_exit()
        if not check.allowed:
            self._log(
                signal.symbol,
                signal.action,
                f"{signal.reason} Blocked: {check.reason}",
                "blocked",
                signal.price,
                details=signal.details,
            )
            return
        qty = trade.qty if trade is not None else abs(float(position["qty"])) if position else 0
        if not qty:
            self._log(signal.symbol, signal.action, f"{signal.reason} Nothing to sell.", "skipped", signal.price)
            return
        try:
            order = self._submit(signal.symbol, signal.action, qty, "exit")
        except (BrokerError, ValueError) as exc:
            if trade is not None:
                self.strategy.on_exit_failed(signal.symbol)
            self._quiet(signal.symbol, "error", f"Exit order failed: {exc}", "failed", signal.price)
            return
        self.risk.record_order()
        if trade is not None:
            self._update_trade(
                trade.trade_id,
                status="closing",
                exit_reason=signal.reason,
                exit_signal_at=naive(now),
                exit_order_id=order.get("id"),
            )
            self._pending[order["id"]] = {
                "kind": "exit",
                "symbol": signal.symbol,
                "trade": trade,
                "submitted_at": self._now(),
                "next_poll": now,
            }
        self._log(signal.symbol, signal.action, signal.reason, "placed", signal.price, qty, order.get("id"), signal.details)

    # ------------------------------------------------------------ fills

    def poll_fills(self, now: datetime) -> None:
        for order_id, item in list(self._pending.items()):
            if now < item["next_poll"]:
                continue
            item["next_poll"] = now + self._fill_poll
            try:
                order = self.broker.get_order(order_id)
            except BrokerError as exc:
                logger.info("Fill check failed for %s: %s", order_id, exc)
                continue
            status = order.get("status")
            if status == "filled":
                self._pending.pop(order_id, None)
                self._on_filled(item, order, now)
            elif status in DEAD_ORDER or now - item["submitted_at"] > self._fill_timeout:
                self._pending.pop(order_id, None)
                why = f"Alpaca says {status}" if status in DEAD_ORDER else f"not filled within {self._fill_timeout.seconds}s"
                if status not in DEAD_ORDER:
                    try:
                        self.broker.cancel_order(order_id)
                    except (BrokerError, ValueError):
                        pass
                self._on_dead(item, why)

    def _on_filled(self, item: Dict, order: Dict, now: datetime) -> None:
        trade, symbol = item["trade"], item["symbol"]
        price = float(order.get("filled_avg_price") or 0) or trade.last_price or trade.follower_price_at_signal
        filled_at = parse_rfc3339(order.get("filled_at")) or now
        if item["kind"] == "entry":
            self.strategy.on_entry_filled(symbol, price, filled_at)
            self._update_trade(trade.trade_id, status="open", entry_price=price, entry_filled_at=naive(filled_at))
            row = self._get_trade(trade.trade_id)
            timings = trade_timings(row) if row else {}
            self._log(
                symbol,
                "info",
                f"Filled: {'bought' if trade.side == 'buy' else 'shorted'} {trade.qty:g} {symbol} at ${price:,.2f}. "
                f"Leader move to fill took {timings.get('leader_to_fill_ms') or 0:,.0f} ms; "
                f"slippage {timings.get('slippage_pct') or 0:+.3f}% from the signal price.",
                "info",
                price,
                trade.qty,
                order.get("id"),
                {"timings": timings},
            )
        else:
            self.strategy.on_exit_filled(symbol)
            pnl = trade.pnl_pct(price) if price else None
            self._update_trade(trade.trade_id, status="closed", exit_price=price, exit_filled_at=naive(filled_at), pnl_pct=pnl)
            self._log(
                symbol,
                "info",
                f"Closed {symbol} at ${price:,.2f} ({pnl:+.2f}%).",
                "info",
                price,
                trade.qty,
                order.get("id"),
                {"pnl_pct": pnl},
            )

    def _on_dead(self, item: Dict, why: str) -> None:
        symbol, trade = item["symbol"], item["trade"]
        if item["kind"] == "entry":
            self.strategy.on_entry_failed(symbol)
            self._update_trade(trade.trade_id, status="failed")
            self._log(symbol, "error", f"Entry order for {symbol} didn't fill: {why}.", "failed")
        else:
            self.strategy.on_exit_failed(symbol)
            self._update_trade(trade.trade_id, status="open")
            self._log(symbol, "error", f"Exit order for {symbol} didn't fill: {why}. The bot will try again.", "failed")

    # ------------------------------------------------------------ broker reads

    def _positions(self, now: datetime, force: bool = False) -> Dict[str, Dict]:
        stamp, cached = self._positions_cache
        if not force and (now - stamp).total_seconds() < POSITIONS_TTL_SECONDS:
            return cached
        try:
            positions = {p["symbol"]: p for p in self.broker.get_positions()}
        except BrokerError:
            if force:
                raise
            return cached
        self._positions_cache = (now, positions)
        return positions

    def _account_snapshot(self) -> Dict:
        now = self._now()
        stamp, cached = self._account_cache
        if cached is not None and (now - stamp).total_seconds() < ACCOUNT_TTL_SECONDS:
            return cached
        try:
            broker = self.broker or self._broker_factory()
            snapshot = {
                "account": account_summary(broker.get_account() or {}),
                "positions": [position_summary(p) for p in broker.get_positions()],
                "orders": [order_summary(o) for o in broker.get_orders(limit=25)],
                "broker_error": None,
            }
        except (LiveTradingRefused, BrokerNotConfigured, BrokerError) as exc:
            snapshot = {"account": None, "positions": [], "orders": [], "broker_error": str(exc)}
        self._account_cache = (now, snapshot)
        return snapshot

    # ------------------------------------------------------------ database

    def _log(self, symbol, action, reason, status, price=None, qty=None, order_id=None, details=None) -> None:
        logger.info("Bot %s %s %s: %s", status, action, symbol or "", reason)
        try:
            with self._session_factory() as db:
                db.add(
                    BotDecision(
                        strategy=self.config.strategy if self.config else None,
                        symbol=symbol,
                        action=action,
                        status=status,
                        reason=reason,
                        price=price,
                        qty=qty,
                        order_id=order_id,
                        details=details or None,
                    )
                )
                db.commit()
        except Exception:  # noqa: BLE001 - a logging failure must not stop trading decisions
            logger.exception("Couldn't store a bot decision")

    def _orders_placed_today(self) -> int:
        midnight = datetime.combine(
            self._now().astimezone(market_calendar.NEW_YORK).date(), time(0), tzinfo=market_calendar.NEW_YORK
        )
        with self._session_factory() as db:
            return (
                db.query(BotDecision).filter(BotDecision.status == "placed", BotDecision.created_at >= naive(midnight)).count()
            )

    def _insert_trade(self, trade, order_id: Optional[str], submitted: datetime) -> int:
        with self._session_factory() as db:
            row = BotTrade(
                leader=trade.leader,
                follower=trade.follower,
                side=trade.side,
                qty=trade.qty,
                status="pending",
                leader_move_pct=trade.leader_move_pct,
                leader_move_started_at=naive(trade.leader_move_started_at),
                leader_move_at=naive(trade.leader_move_at),
                leader_price=trade.leader_price,
                follower_price_at_signal=trade.follower_price_at_signal,
                follower_move_pct=trade.follower_move_pct,
                signal_at=naive(trade.signal_at),
                entry_reason=trade.entry_reason,
                entry_order_id=order_id,
                entry_submitted_at=naive(submitted),
            )
            db.add(row)
            db.commit()
            return row.id

    def _update_trade(self, trade_id: Optional[int], **fields) -> None:
        if trade_id is None:
            return
        with self._session_factory() as db:
            row = db.get(BotTrade, trade_id)
            if row:
                for key, value in fields.items():
                    setattr(row, key, value)
                db.commit()

    def _get_trade(self, trade_id: Optional[int]) -> Optional[BotTrade]:
        if trade_id is None:
            return None
        with self._session_factory() as db:
            return db.get(BotTrade, trade_id)

    # ------------------------------------------------------------ status

    def chart(self) -> List[Dict]:
        """
        Per ticker: recent bars with both averages, the crosses, and this run's orders, for the Bot page.
        Uses the bars the loop already loaded, so it costs no extra Polygon requests.
        """
        if not self.config or self.config.strategy != "ma_crossover" or not self._bars:
            return []
        p = self.config.ma_crossover
        orders: Dict[str, List[Dict]] = {}
        with self._session_factory() as db:
            rows = (
                db.query(BotDecision)
                .filter(
                    BotDecision.action.in_(("buy", "sell")),
                    BotDecision.status == "placed",
                    BotDecision.created_at >= naive(self.started_at),
                )
                .order_by(BotDecision.id)
                .all()
            )
            for row in rows:
                orders.setdefault(row.symbol, []).append(
                    {
                        "t": (row.details or {}).get("bar_at"),
                        "side": row.action,
                        "price": row.price,
                        "qty": row.qty,
                        "at": iso(row.created_at),
                    }
                )
        out = []
        for symbol in self.strategy.symbols():
            closes = self._bars.get(symbol)
            if closes is None:
                continue
            data = ma_chart(closes, p.short_window, p.long_window)
            out.append(
                {
                    "symbol": symbol,
                    "timespan": p.timespan,
                    "short_window": p.short_window,
                    "long_window": p.long_window,
                    **data,
                    "orders": orders.get(symbol, []),
                }
            )
        return out

    def status(self) -> Dict:
        limits = (
            self.risk.status()
            if self.risk
            else {**_default_limits().model_dump(), "orders_today": 0, "killed": False, "kill_reason": None}
        )
        caveats = [CAVEAT_PAPER]
        if self.config and self.config.strategy == "lead_lag" and self.config.lead_lag.source == "alpaca":
            caveats.append(CAVEAT_IEX)
        if self.config and self.config.strategy == "ma_crossover" and not settings.POLYGON_REALTIME:
            caveats.append(CAVEAT_DELAYED)
        with self._session_factory() as db:
            decisions = db.query(BotDecision).order_by(BotDecision.id.desc()).limit(RECENT_DECISIONS).all()
            trades = db.query(BotTrade).order_by(BotTrade.id.desc()).limit(RECENT_TRADES).all()
            history = {"decisions": [decision_dict(d) for d in decisions], "trades": [trade_dict(t) for t in trades]}
        return {
            "enabled": settings.BOT_ENABLED,
            "running": self.running,
            "strategy": self.config.strategy if self.config else None,
            "strategy_label": self.strategy.label if self.strategy else None,
            "config": self.config.model_dump() if self.config else None,
            "started_at": iso(naive(self.started_at)),
            "stopped_at": iso(naive(self.stopped_at)),
            "stopped_reason": self.stopped_reason,
            "last_error": self.last_error,
            "market": market_calendar.market_state(),
            "risk": limits,
            "stream": self.stream.status() if self.stream else None,
            "caveats": caveats,
            **self._account_snapshot(),
            **history,
            "chart": self.chart(),
        }


def strategy_catalog() -> List[Dict]:
    """Each strategy's settings with defaults and limits, for the Bot page form."""
    out = []
    for name, (cls, params) in STRATEGIES.items():
        out.append({"name": name, "label": cls.label, "feed": cls.feed, "schema": params.model_json_schema()})
    return out


runner = BotRunner()


def get_runner() -> BotRunner:
    return runner
