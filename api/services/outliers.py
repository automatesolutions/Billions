"""
Outlier service: scheduled refreshes and the read model for the outliers page.

Refreshes run in one background thread, never from a public request:
- at startup if there is no data,
- every REFRESH_INTERVAL_MINUTES while the market is open,
- once after each session close, to capture final prices.
"""

import logging
import math
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.config import settings
from api.database import SessionLocal
from api.models import PerfMetric
from api.services import market_calendar
from api.services.outlier_engine import STRATEGIES, Z_THRESHOLD, Strategy, run_refresh

logger = logging.getLogger(__name__)

POST_CLOSE_DELAY = timedelta(minutes=20)  # Yahoo needs a few minutes to settle final closes
FAILURE_BACKOFF = timedelta(minutes=15)
CHECK_EVERY_SECONDS = 60

_lock = threading.Lock()
_stop = threading.Event()
_status: Dict = {"is_running": False, "last_success": None, "last_error": None, "last_attempt": None}


# ---------------------------------------------------------------- read model


def strategy_info(name: str) -> Optional[Dict]:
    strategy = STRATEGIES.get(name)
    if strategy is None:
        return None
    return {
        "strategy": strategy.name,
        "x_label": strategy.long_label,
        "y_label": strategy.short_label,
        "x_days": strategy.long_days,
        "y_days": strategy.short_days,
        "min_dollar_volume": strategy.min_dollar_volume,
        "z_threshold": Z_THRESHOLD,
    }


def all_strategies() -> List[Dict]:
    return [strategy_info(name) for name in STRATEGIES]


def outlier_score(z_x: float, z_y: float) -> float:
    """Distance from the centre of the group, in standard deviations."""
    return math.hypot(z_x, z_y)


def direction(z_x: float, z_y: float) -> str:
    """'up' or 'down' from the axes that crossed the threshold; 'mixed' if they disagree."""
    signs = {z > 0 for z in (z_x, z_y) if abs(z) > Z_THRESHOLD}
    if not signs:
        signs = {(z_x if abs(z_x) >= abs(z_y) else z_y) > 0}
    if len(signs) == 2:
        return "mixed"
    return "up" if signs.pop() else "down"


def _move(pct: float, label: str) -> str:
    return f"{'up' if pct >= 0 else 'down'} {abs(pct):.1f}% over {label}"


def reason(strategy: Strategy, metric_x: float, metric_y: float, z_x: float, z_y: float) -> str:
    """One plain sentence on why this stock stands out."""
    long_hit, short_hit = abs(z_x) > Z_THRESHOLD, abs(z_y) > Z_THRESHOLD
    if long_hit and short_hit:
        text = f"{_move(metric_x, strategy.long_label)} and {_move(metric_y, strategy.short_label)}"
        z = max(abs(z_x), abs(z_y))
    elif long_hit:
        text, z = _move(metric_x, strategy.long_label), abs(z_x)
    else:
        text, z = _move(metric_y, strategy.short_label), abs(z_y)
    return f"{text[0].upper()}{text[1:]}, {z:.1f}σ from the group"


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:  # SQLite drops tzinfo; stored values are UTC
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def get_outliers(db: Session, name: str) -> Dict:
    strategy = STRATEGIES[name]
    rows = db.query(PerfMetric).filter(PerfMetric.strategy == name).all()

    points = [
        {"symbol": r.symbol, "x": r.metric_x, "y": r.metric_y, "z_x": r.z_x, "z_y": r.z_y, "is_outlier": bool(r.is_outlier)}
        for r in rows
    ]
    ranked = sorted((r for r in rows if r.is_outlier), key=lambda r: outlier_score(r.z_x, r.z_y), reverse=True)
    outliers = [
        {
            "rank": i + 1,
            "symbol": r.symbol,
            "score": round(outlier_score(r.z_x, r.z_y), 2),
            "direction": direction(r.z_x, r.z_y),
            "reason": reason(strategy, r.metric_x, r.metric_y, r.z_x, r.z_y),
            "x": r.metric_x,
            "y": r.metric_y,
            "z_x": r.z_x,
            "z_y": r.z_y,
        }
        for i, r in enumerate(ranked)
    ]
    first = rows[0] if rows else None
    return {
        **strategy_info(name),
        "as_of": first.price_date.isoformat() if first and first.price_date else None,
        "computed_at": _iso(first.inserted) if first else None,
        "market": market_calendar.market_state(),
        "refresh": refresh_status(),
        "universe_count": len(points),
        "outlier_count": len(outliers),
        "outliers": outliers,
        "points": points,
    }


# ---------------------------------------------------------------- scheduler


def refresh_status() -> Dict:
    return dict(_status)


def last_computed_at(db: Session) -> Optional[datetime]:
    value = db.query(func.min(PerfMetric.inserted)).scalar()
    if value is not None and value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def should_refresh(now: datetime, last_run: Optional[datetime], last_failure: Optional[datetime] = None) -> bool:
    """Pure decision rule for the scheduler. `now`, `last_run` are timezone-aware."""
    if last_failure is not None and now - last_failure < FAILURE_BACKOFF:
        return False
    if last_run is None:
        return True
    if market_calendar.market_state(now)["is_open"]:
        return now - last_run >= timedelta(minutes=settings.REFRESH_INTERVAL_MINUTES)
    close = market_calendar.last_session_close(now)
    return now >= close + POST_CLOSE_DELAY and last_run < close + POST_CLOSE_DELAY


def refresh_once() -> bool:
    """Run one refresh unless one is already running. Returns True if it ran successfully."""
    if not _lock.acquire(blocking=False):
        return False
    _status.update(is_running=True, last_attempt=datetime.now(timezone.utc).isoformat())
    try:
        run_refresh()
        _status.update(last_success=datetime.now(timezone.utc).isoformat(), last_error=None)
        return True
    except Exception as exc:  # noqa: BLE001 - keep the scheduler alive
        logger.exception("Outlier refresh failed")
        _status["last_error"] = {"at": datetime.now(timezone.utc).isoformat(), "message": str(exc)}
        return False
    finally:
        _status["is_running"] = False
        _lock.release()


def _scheduler_loop() -> None:
    last_failure: Optional[datetime] = None
    while not _stop.is_set():
        now = datetime.now(timezone.utc)
        with SessionLocal() as db:
            last_run = last_computed_at(db)
        if should_refresh(now, last_run, last_failure):
            last_failure = None if refresh_once() else datetime.now(timezone.utc)
        _stop.wait(CHECK_EVERY_SECONDS)


def start_scheduler() -> None:
    _stop.clear()
    threading.Thread(target=_scheduler_loop, name="outlier-scheduler", daemon=True).start()


def stop_scheduler() -> None:
    _stop.set()
