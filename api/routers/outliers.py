"""
Outlier endpoints (read-only).
"""

from fastapi import APIRouter, Depends, HTTPException, Response
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


@router.get("/status")
def get_status():
    """State of the background refresh. Refreshes are scheduled; they cannot be started from the API."""
    return service.refresh_status()


@router.get("/{strategy}/info")
def get_strategy_info(strategy: str):
    return service.strategy_info(_check_strategy(strategy))


@router.get("/{strategy}")
def get_outliers(strategy: str, response: Response, db: Session = Depends(get_db)):
    _check_strategy(strategy)
    response.headers["Cache-Control"] = "public, max-age=60, stale-while-revalidate=300"
    return service.get_outliers(db, strategy)
