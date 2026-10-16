"""
Quant page service: bars, cached model fits, the current action, and the live record.

Every time a (ticker, interval) is computed on a newly closed bar, the signal is
recorded. When the following bar closes, it is scored against what happened.
Watched pairs are computed by a background thread so the record keeps growing
with no page open. Nothing here places orders.
"""

import logging
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.config import settings
from api.database import SessionLocal
from api.models import QuantSignal, QuantWatch
from api.services import bars, market_calendar, quotes
from api.services import quant_strategy as qs
from api.services.analysis import _clean

logger = logging.getLogger(__name__)

DEFAULT_COST_BPS = 10.0
MIN_COST_BPS = 1.0
BARS_TTL_SECONDS = {"1m": 15, "1h": 60, "4h": 60, "1d": 600}
DAILY_UPDATE_DELAY = timedelta(minutes=15)  # Yahoo's daily bar settles a little after the close
FIT_CACHE_SIZE = 128
RECENT_SIGNALS = 30
LIVE_WINDOW = 500  # live record uses the most recent N signals
CHECK_EVERY_SECONDS = 15
FAILURE_BACKOFF = timedelta(minutes=5)


class TickerNotFound(Exception):
    pass


class WatchLimitReached(Exception):
    pass


_bars_cache: Dict[tuple, tuple] = {}
_fit_cache: "OrderedDict[tuple, Dict]" = OrderedDict()
_cache_lock = threading.Lock()
_pair_locks: Dict[tuple, threading.Lock] = {}
_stop = threading.Event()


def interval_list() -> List[Dict]:
    return [{"key": iv.key, "label": iv.label} for iv in bars.INTERVALS.values()]


# --------------------------------------------------------------------------- inputs


def _load_bars(ticker: str, iv: bars.Interval, fetch: Optional[Callable]) -> pd.Series:
    if fetch is not None:
        return fetch(ticker, iv)
    key = (ticker, iv.key)
    cached = _bars_cache.get(key)
    if cached and time.monotonic() - cached[0] < BARS_TTL_SECONDS[iv.key]:
        return cached[1]
    close = bars.load(ticker, iv)
    _bars_cache[key] = (time.monotonic(), close)
    return close


def round_trip_cost(ticker: str) -> Dict:
    """Live quoted spread when Alpaca keys are set (crossing it once each way costs about one spread), else a default."""
    if quotes.configured():
        try:
            quote = quotes.get_quote(ticker)
            if quote.get("available") and quote["spread_bps"] > 0:
                return {"bps": max(MIN_COST_BPS, round(quote["spread_bps"], 1)), "source": "live_spread"}
        except Exception as exc:  # noqa: BLE001 - cost falls back to the default
            logger.info("Spread unavailable for %s: %s", ticker, exc)
    return {"bps": DEFAULT_COST_BPS, "source": "default"}


def _iso_ny(naive_ny: datetime) -> str:
    return naive_ny.replace(tzinfo=bars.NY).isoformat()


def next_update_at(iv: bars.Interval, next_start: pd.Timestamp) -> datetime:
    """When the next bar will be final (New York, tz-aware)."""
    end = bars.bar_end(iv, next_start).replace(tzinfo=bars.NY)
    return end + (DAILY_UPDATE_DELAY if not iv.intraday else timedelta(seconds=iv.grace_seconds))


# --------------------------------------------------------------------------- live record


def score_signals(db: Session, ticker: str, iv: bars.Interval, close: pd.Series) -> None:
    """Fill in the realised return of every open signal whose target bar has closed."""
    last = pd.Timestamp(close.index[-1]).to_pydatetime()
    open_rows = (
        db.query(QuantSignal)
        .filter(
            QuantSignal.ticker == ticker,
            QuantSignal.interval == iv.key,
            QuantSignal.realized.is_(None),
            QuantSignal.void.is_(False),
            QuantSignal.target_time <= last,
        )
        .all()
    )
    for row in open_rows:
        target, start = pd.Timestamp(row.target_time), pd.Timestamp(row.bar_time)
        if target in close.index:
            base = float(close[start]) if start in close.index else row.close
            row.realized = math.log(float(close[target]) / base)
        else:
            row.void = True  # target bar missing from the data: never scored
    if open_rows:
        db.commit()


def record_signal(db: Session, ticker: str, iv: bars.Interval, fit: Dict, act: Dict, cost_bps: float) -> None:
    bar_time = pd.Timestamp(fit["last_bar_ts"]).to_pydatetime()
    exists = (
        db.query(QuantSignal.id)
        .filter(QuantSignal.ticker == ticker, QuantSignal.interval == iv.key, QuantSignal.bar_time == bar_time)
        .first()
    )
    if exists:
        return
    db.add(
        QuantSignal(
            ticker=ticker,
            interval=iv.key,
            bar_time=bar_time,
            target_time=pd.Timestamp(fit["next_bar"]).to_pydatetime(),
            close=fit["price"],
            forecast=act["forecast_log_return"],
            strength=act["strength"],
            action=act["action"],
            position=act["position"],
            cost_bps=cost_bps,
        )
    )
    try:
        db.commit()
    except IntegrityError:  # recorded by a concurrent request
        db.rollback()


def _signals(db: Session, ticker: str, iv: bars.Interval) -> List[QuantSignal]:
    rows = (
        db.query(QuantSignal)
        .filter(QuantSignal.ticker == ticker, QuantSignal.interval == iv.key)
        .order_by(QuantSignal.bar_time.desc())
        .limit(LIVE_WINDOW)
        .all()
    )
    return rows[::-1]


def _as_dict(row: QuantSignal) -> Dict:
    return {
        "bar_time": pd.Timestamp(row.bar_time),
        "target_time": pd.Timestamp(row.target_time),
        "forecast": row.forecast,
        "position": row.position,
        "cost_bps": row.cost_bps,
        "realized": row.realized,
        "void": row.void,
    }


def previous_position(db: Session, ticker: str, iv: bars.Interval, bar_time: pd.Timestamp) -> float:
    """Position the guide held over the bar that just closed: the signal that targeted it, else none."""
    row = (
        db.query(QuantSignal.position)
        .filter(
            QuantSignal.ticker == ticker,
            QuantSignal.interval == iv.key,
            QuantSignal.target_time == pd.Timestamp(bar_time).to_pydatetime(),
        )
        .first()
    )
    return float(row.position) if row else 0.0


def _recent(rows: List[QuantSignal], iv: bars.Interval) -> List[Dict]:
    held = {pd.Timestamp(r.target_time): r.position for r in rows}  # position held over each bar
    out = []
    for row in rows[-RECENT_SIGNALS:][::-1]:
        status = "void" if row.void else "open" if row.realized is None else "scored"
        direction = 0 if row.forecast == 0 else (1 if row.forecast > 0 else -1)
        out.append(
            {
                "bar_time": qs._stamp(row.bar_time, iv),
                "target_time": qs._stamp(row.target_time, iv),
                "action": row.action,
                "position": row.position,
                "instruction": qs.instruction(held.get(pd.Timestamp(row.bar_time), 0.0), row.position)["code"],
                "forecast_bps": row.forecast * 1e4,
                "realized_bps": row.realized * 1e4 if row.realized is not None else None,
                "correct": (direction * row.realized > 0) if row.realized not in (None, 0) and direction else None,
                "status": status,
            }
        )
    return out


# --------------------------------------------------------------------------- watch list


def is_watched(db: Session, ticker: str, key: str) -> bool:
    return db.query(QuantWatch.id).filter(QuantWatch.ticker == ticker, QuantWatch.interval == key).first() is not None


def watch(db: Session, ticker: str, key: str) -> None:
    if is_watched(db, ticker, key):
        return
    if db.query(QuantWatch).count() >= settings.QUANT_MAX_WATCHES:
        raise WatchLimitReached()
    db.add(QuantWatch(ticker=ticker, interval=key))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


def unwatch(db: Session, ticker: str, key: str) -> None:
    db.query(QuantWatch).filter(QuantWatch.ticker == ticker, QuantWatch.interval == key).delete()
    db.commit()


def watch_list(db: Session) -> List[Dict]:
    return [{"ticker": w.ticker, "interval": w.interval} for w in db.query(QuantWatch).order_by(QuantWatch.id).all()]


# --------------------------------------------------------------------------- main entry


def _fit(ticker: str, iv: bars.Interval, close: pd.Series, next_start: pd.Timestamp) -> Dict:
    key = (ticker, iv.key, pd.Timestamp(close.index[-1]).isoformat(), len(close))
    with _cache_lock:
        if key in _fit_cache:
            _fit_cache.move_to_end(key)
            return _fit_cache[key]
    fit = qs.fit_and_test(close, iv, next_start)
    with _cache_lock:
        _fit_cache[key] = fit
        while len(_fit_cache) > FIT_CACHE_SIZE:
            _fit_cache.popitem(last=False)
    return fit


def get_quant(db: Session, ticker: str, key: str, fetch: Optional[Callable] = None, cost: Optional[Dict] = None) -> Dict:
    """Payload for the Quant page. `fetch(ticker, interval)` and `cost` are injectable for tests."""
    iv = bars.INTERVALS[key]
    with _cache_lock:
        lock = _pair_locks.setdefault((ticker, key), threading.Lock())
    with lock:  # one computation per pair at a time
        close = _load_bars(ticker, iv, fetch)
        if close.empty:
            raise TickerNotFound(ticker)
        next_start = bars.next_bar_start(iv, close.index[-1])
        base = {
            "ticker": ticker,
            "interval": iv.key,
            "interval_label": iv.label,
            "price": float(close.iloc[-1]),
            "last_bar": qs._stamp(close.index[-1], iv),
            "next_bar": qs._stamp(next_start, iv),
            "next_update_at": next_update_at(iv, next_start).isoformat(),
            "last_update_at": next_update_at(iv, pd.Timestamp(close.index[-1])).isoformat(),  # when the last bar became final
            "market": market_calendar.market_state(),
            "watching": is_watched(db, ticker, key),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            fit = _fit(ticker, iv, close, next_start)
        except ValueError as exc:
            return _clean({**base, "status": "insufficient_history", "message": str(exc), "needed_bars": qs.MIN_BARS})

        cost = cost or round_trip_cost(ticker)
        net = qs.backtest_net(fit, cost["bps"], iv)
        score_signals(db, ticker, iv, close)
        live = qs.live_summary([_as_dict(r) for r in _signals(db, ticker, iv)], fit, iv)
        act = qs.action(fit, net, live["verdict"], iv)
        act["instruction"] = qs.instruction(previous_position(db, ticker, iv, fit["last_bar_ts"]), act["position"])
        record_signal(db, ticker, iv, fit, act, cost["bps"])
        rows = _signals(db, ticker, iv)
        live = qs.live_summary([_as_dict(r) for r in rows], fit, iv)  # now including the signal just recorded

        caveats = []
        if fit["split"]["test_bars"] < qs.qa.MIN_TEST_OBS:
            caveats.append(f"Small sample: only {fit['split']['test_bars']} test bars.")
        if not fit["regime"]["significant"]:
            caveats.append("The AR(1) weight is not statistically different from zero, so the regime label is weak.")
        if cost["source"] == "default":
            caveats.append(f"Costs use a default {DEFAULT_COST_BPS:.0f} bps round trip. Add Alpaca keys to use the live spread.")
        if iv.key == "1m":
            caveats.append("Yahoo's 1-minute bars arrive up to a minute late, so live 1-minute signals are already stale when shown.")

        return _clean(
            {
                **base,
                "status": "ok",
                "bars": fit["bars"],
                "cost": cost,
                "action": act,
                "explain": {
                    **fit["next"],
                    "strength_scale": qs.qa.STRENGTH_SCALE,
                    "neutral_band": qs.qa.NEUTRAL_BAND,
                    "slot_minutes": iv.slot_minutes,
                    "flat_overnight": iv.flat_overnight,
                },
                "backtest": {
                    "split": fit["split"],
                    "models": fit["models"],
                    "hit_rate": fit["hit_rate"],
                    "gross": fit["gross"],
                    "net": net,
                    "walk_forward": fit["walk_forward"],
                    "random_baseline": fit["random_baseline"],
                    "regime": fit["regime"],
                    "periods_per_year": iv.periods_per_year,
                },
                "live": {**live, "recent": _recent(rows, iv)},
                "caveats": caveats,
            }
        )

# --------------------------------------------------------------------------- scheduler


def _scheduler_loop() -> None:
    due: Dict[tuple, datetime] = {}
    while not _stop.is_set():
        now = datetime.now(timezone.utc)
        try:
            with SessionLocal() as db:
                pairs = [(w["ticker"], w["interval"]) for w in watch_list(db)]
                for pair in pairs:
                    if now < due.get(pair, now - timedelta(seconds=1)) or pair[1] not in bars.INTERVALS:
                        continue
                    try:
                        payload = get_quant(db, *pair)
                        due[pair] = datetime.fromisoformat(payload["next_update_at"]).astimezone(timezone.utc)
                    except Exception as exc:  # noqa: BLE001 - keep the scheduler alive
                        logger.info("Quant watch %s %s failed: %s", *pair, exc)
                        due[pair] = now + FAILURE_BACKOFF
        except Exception:  # noqa: BLE001
            logger.exception("Quant scheduler pass failed")
        _stop.wait(CHECK_EVERY_SECONDS)


def start_scheduler() -> None:
    _stop.clear()
    threading.Thread(target=_scheduler_loop, name="quant-scheduler", daemon=True).start()


def stop_scheduler() -> None:
    _stop.set()
