"""
Quant page endpoints: the current action and live record for a ticker at a bar interval,
and the watch list that keeps recording signals in the background. No orders are placed.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session

from api.config import settings
from api.database import get_db
from api.limits import limiter
from api.services import quant as service
from api.services.bars import INTERVALS
from api.services.prices import normalize_ticker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quant", tags=["Quant"])

WATCH_LIMIT = "10/minute"


def _symbol(ticker: str) -> str:
    symbol = normalize_ticker(ticker)
    if symbol is None:
        raise HTTPException(
            status_code=422, detail="That is not a valid ticker. Use 1 to 10 letters, numbers, dots or dashes."
        )
    return symbol


def _interval(interval: str) -> str:
    if interval not in INTERVALS:
        raise HTTPException(status_code=422, detail=f"Unknown interval '{interval}'. Use {', '.join(INTERVALS)}.")
    return interval


@router.get("/intervals")
def get_intervals():
    return {"intervals": service.interval_list()}


@router.get("/watching")
def get_watching(db: Session = Depends(get_db)):
    return {"watching": service.watch_list(db), "max": settings.QUANT_MAX_WATCHES}


@router.get("/{ticker}")
@limiter.limit(settings.RATE_LIMIT_ANALYSIS)
def get_quant(ticker: str, request: Request, response: Response, interval: str = Query("1h"), db: Session = Depends(get_db)):
    symbol, key = _symbol(ticker), _interval(interval)
    try:
        payload = service.get_quant(db, symbol, key)
    except service.TickerNotFound:
        raise HTTPException(status_code=404, detail=f"No {INTERVALS[key].label.lower()} prices found for {symbol}.")
    except Exception:
        logger.exception("Quant failed for %s %s", symbol, key)
        raise HTTPException(status_code=502, detail="The price source did not respond. Try again in a minute.")
    response.headers["Cache-Control"] = "no-store"  # the live record changes as bars close
    return payload


@router.post("/{ticker}/watch")
@limiter.limit(WATCH_LIMIT)
def post_watch(ticker: str, request: Request, interval: str = Query("1h"), db: Session = Depends(get_db)):
    symbol, key = _symbol(ticker), _interval(interval)
    try:
        service.watch(db, symbol, key)
    except service.WatchLimitReached:
        raise HTTPException(
            status_code=409,
            detail=f"Already tracking {settings.QUANT_MAX_WATCHES} stocks, the most this server records. Stop one first.",
        )
    return {"ticker": symbol, "interval": key, "watching": True}


@router.delete("/{ticker}/watch")
@limiter.limit(WATCH_LIMIT)
def delete_watch(ticker: str, request: Request, interval: str = Query("1h"), db: Session = Depends(get_db)):
    symbol, key = _symbol(ticker), _interval(interval)
    service.unwatch(db, symbol, key)
    return {"ticker": symbol, "interval": key, "watching": False}
