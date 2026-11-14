"""
Lead-lag research endpoints: analyze a pair, scan the watchlist and candidates, results, history, watchlist.
Research only: nothing here trades.
"""

import logging
import math
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from api.config import settings
from api.database import get_db
from api.limits import limiter
from api.services import lead_lag_research as research
from api.services.lead_lag import EngineParams, analyze_pair

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/lead-lag", tags=["Lead-lag research"])

ANALYZE_MAX_WAIT_SECONDS = 30  # wait this long for a rate-limit slot, then tell the caller to retry
SCAN_LIMIT = "6/minute"


class PairBody(BaseModel):
    a: str
    b: str

    @model_validator(mode="after")
    def _clean(self):
        self.a, self.b = research.clean_pair(self.a, self.b)
        return self


class AnalyzeBody(PairBody):
    threshold_pct: float = Field(0.3, gt=0, le=10)
    lookback_seconds: float = Field(60, ge=1, le=3600)
    max_wait_seconds: float = Field(300, ge=1, le=7200)
    days: int = Field(60, ge=5, le=250)
    source: Optional[Literal["polygon", "alpaca"]] = None


class ScanBody(BaseModel):
    watchlist: bool = True
    candidates: bool = True


def _source(requested: Optional[str]) -> str:
    source = requested or settings.LEAD_LAG_SOURCE
    problem = research.check_source(source)
    if problem:
        raise HTTPException(status_code=400, detail=problem)
    return source


@router.post("/analyze")
@limiter.limit(settings.RATE_LIMIT_ANALYSIS)
def post_analyze(request: Request, response: Response, body: AnalyzeBody):
    source = _source(body.source)
    start, end = research.date_range(body.days)
    try:
        a = research.load_closes(body.a, source, start, end, max_wait=ANALYZE_MAX_WAIT_SECONDS)
        b = research.load_closes(body.b, source, start, end, max_wait=ANALYZE_MAX_WAIT_SECONDS)
    except research.DATA_ERRORS as exc:
        retry = getattr(exc, "retry_after", None)
        if retry:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": str(math.ceil(retry))})
        raise HTTPException(status_code=502, detail=str(exc))
    if a.empty or b.empty:
        missing = body.a if a.empty else body.b
        raise HTTPException(
            status_code=404, detail=f"No intraday bars found for {missing} in the last {body.days} trading days."
        )
    params = EngineParams(body.threshold_pct, body.lookback_seconds, body.max_wait_seconds)
    result = analyze_pair(body.a, a, body.b, b, params, bar_seconds=research.resolution(source)[0])
    response.headers["Cache-Control"] = "no-store"
    return {**result, "source": source, "start": start.isoformat(), "end": end.isoformat()}


@router.post("/scan/estimate")
def post_estimate(body: ScanBody, db: Session = Depends(get_db)):
    source = _source(None)
    pairs = research.scan_pairs(db, body.watchlist, body.candidates)
    return research.estimate(pairs, source, settings.LEAD_LAG_DAYS)


@router.post("/scan")
@limiter.limit(SCAN_LIMIT)
def post_scan(
    request: Request,
    body: ScanBody,
    db: Session = Depends(get_db),
    scans: research.ScanManager = Depends(research.get_manager),
):
    source = _source(None)
    pairs = research.scan_pairs(db, body.watchlist, body.candidates)
    if not pairs:
        raise HTTPException(status_code=422, detail="Nothing to scan. Add a pair to your watchlist or include the candidates.")
    try:
        return scans.start(pairs, source, settings.LEAD_LAG_DAYS, research.DAILY_PARAMS)
    except research.ScanRunning:
        raise HTTPException(status_code=409, detail="A scan is already running. Wait for it to finish.")


@router.get("/scan")
def get_scan(scans: research.ScanManager = Depends(research.get_manager)):
    return {"scan": scans.status()}


@router.get("/results")
def get_results(
    response: Response, db: Session = Depends(get_db), scans: research.ScanManager = Depends(research.get_manager)
):
    response.headers["Cache-Control"] = "no-store"
    return {**research.latest_results(db), "scan": scans.status()}


@router.get("/pairs/{a}/{b}")
def get_pair(a: str, b: str, db: Session = Depends(get_db)):
    try:
        a, b = research.clean_pair(a, b)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return research.pair_history(db, a, b)


@router.get("/candidates")
def get_candidates(db: Session = Depends(get_db)):
    return {"pairs": research.candidates(db)}


@router.get("/watchlist")
def get_watchlist(db: Session = Depends(get_db)):
    return {"pairs": research.watchlist(db), "max": settings.LEAD_LAG_MAX_WATCHLIST}


@router.post("/watchlist")
def post_watch(body: PairBody, db: Session = Depends(get_db)):
    try:
        research.add_watch(db, body.a, body.b)
    except research.WatchlistFull:
        raise HTTPException(
            status_code=409, detail=f"Your watchlist is full ({settings.LEAD_LAG_MAX_WATCHLIST} pairs). Remove one first."
        )
    return {"pairs": research.watchlist(db)}


@router.delete("/watchlist/{a}/{b}")
def delete_watch(a: str, b: str, db: Session = Depends(get_db)):
    try:
        a, b = research.clean_pair(a, b)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    research.remove_watch(db, a, b)
    return {"pairs": research.watchlist(db)}
