"""
Live top-of-book quote endpoint (read-only).
"""

import logging

from fastapi import APIRouter, HTTPException, Request, Response

from api.config import settings
from api.limits import limiter
from api.services import quotes as service
from api.services.prices import normalize_ticker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quotes", tags=["Quotes"])


@router.get("/{ticker}")
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
def get_quote(ticker: str, request: Request, response: Response):
    symbol = normalize_ticker(ticker)
    if symbol is None:
        raise HTTPException(
            status_code=422, detail="That is not a valid ticker. Use 1 to 10 letters, numbers, dots or dashes."
        )
    try:
        payload = service.get_quote(symbol)
    except Exception:
        logger.exception("Quote failed for %s", symbol)
        raise HTTPException(status_code=502, detail="The quote source did not respond. Try again in a minute.")
    response.headers["Cache-Control"] = f"public, max-age={service.QUOTE_TTL_SECONDS}"
    return payload
