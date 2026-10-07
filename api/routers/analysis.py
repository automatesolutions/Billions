"""
Per-stock analysis endpoint (read-only).
"""

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from api.config import settings
from api.limits import limiter
from api.services import analysis as service
from api.services.prices import normalize_ticker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.get("/{ticker}")
@limiter.limit(settings.RATE_LIMIT_ANALYSIS)
def get_analysis(ticker: str, request: Request, response: Response):
    symbol = normalize_ticker(ticker)
    if symbol is None:
        raise HTTPException(
            status_code=422, detail="That is not a valid ticker. Use 1 to 10 letters, numbers, dots or dashes."
        )
    try:
        payload = service.get_analysis(symbol)
    except service.TickerNotFound:
        raise HTTPException(status_code=404, detail=f"No price history found for {symbol}.")
    except Exception:
        logger.exception("Analysis failed for %s", symbol)
        raise HTTPException(status_code=502, detail="The price source did not respond. Try again in a minute.")
    response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=3600"
    return payload
