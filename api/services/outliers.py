"""
Outlier service: reads cached metrics and runs refreshes in a background thread.
"""

import logging
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from api.models import PerfMetric
from api.services.outlier_engine import STRATEGIES, run_outlier_detection

logger = logging.getLogger(__name__)

_refresh_lock = threading.Lock()
_refresh_status: Dict = {"is_running": False, "current_strategy": None, "started_at": None, "finished_at": None, "error": None}


def strategy_info(strategy: str) -> Optional[Dict]:
    if strategy not in STRATEGIES:
        return None
    x_label, y_label, back_x, back_y, min_market_cap = STRATEGIES[strategy]
    return {
        "strategy": strategy,
        "x_period": x_label,
        "y_period": y_label,
        "lookback_x_days": back_x,
        "lookback_y_days": back_y,
        "min_market_cap": min_market_cap,
    }


def all_strategies() -> List[Dict]:
    return [strategy_info(s) for s in STRATEGIES]


def get_metrics(db: Session, strategy: str) -> List[PerfMetric]:
    return db.query(PerfMetric).filter(PerfMetric.strategy == strategy).all()


def refresh_status() -> Dict:
    return dict(_refresh_status)


def _run_refresh(strategies: List[str]) -> None:
    try:
        for strategy in strategies:
            _refresh_status["current_strategy"] = strategy
            run_outlier_detection(strategy)
        _refresh_status["error"] = None
    except Exception as exc:  # noqa: BLE001 - log and surface any engine failure
        logger.exception("Outlier refresh failed")
        _refresh_status["error"] = str(exc)
    finally:
        _refresh_status.update(is_running=False, current_strategy=None, finished_at=datetime.now(timezone.utc).isoformat())
        _refresh_lock.release()


def start_refresh(strategies: Optional[List[str]] = None) -> bool:
    """Start a background refresh. Returns False if one is already running."""
    if not _refresh_lock.acquire(blocking=False):
        return False
    _refresh_status.update(is_running=True, started_at=datetime.now(timezone.utc).isoformat(), finished_at=None)
    threading.Thread(target=_run_refresh, args=(strategies or list(STRATEGIES),), daemon=True).start()
    return True
