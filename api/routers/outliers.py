"""
Outlier endpoints (read-only market data).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db
from api.services import outliers as service

router = APIRouter(prefix="/outliers", tags=["Outliers"])


def _check_strategy(strategy: str) -> str:
    if service.strategy_info(strategy) is None:
        raise HTTPException(status_code=404, detail=f"Unknown strategy '{strategy}'. Use scalp, swing or longterm.")
    return strategy


@router.get("/strategies")
def get_strategies():
    return {"strategies": service.all_strategies()}


@router.get("/refresh/status")
def get_refresh_status():
    return service.refresh_status()


@router.post("/refresh", status_code=202)
def refresh_all():
    started = service.start_refresh()
    return {"started": started, "status": service.refresh_status()}


@router.get("/{strategy}/info")
def get_strategy_info(strategy: str):
    return service.strategy_info(_check_strategy(strategy))


@router.get("/{strategy}")
def get_outliers(strategy: str, db: Session = Depends(get_db)):
    _check_strategy(strategy)
    rows = service.get_metrics(db, strategy)
    metrics = [
        {
            "symbol": r.symbol,
            "metric_x": r.metric_x,
            "metric_y": r.metric_y,
            "z_x": r.z_x,
            "z_y": r.z_y,
            "is_outlier": bool(r.is_outlier),
        }
        for r in rows
    ]
    return {"strategy": strategy, "count": len(metrics), "metrics": metrics}
