"""
Lead-lag research around the engine: data loading, candidate pairs, scans, storage, and the daily job.

- Data: the finest bars the configured plan allows. Polygon with POLYGON_SECOND_BARS=true
  gives second bars; otherwise minute bars (Polygon free, or Alpaca, which has no second bars).
- Candidates: universe.py has no sector data, so related names come from a small
  hand-kept map (sector peers, ETFs and their top holdings, index ETFs as stand-ins
  for index futures, which the free plans don't include). Nasdaq stocks are kept
  only if they're in the scanned universe, when one has been loaded.
- Scans: one at a time, in a background thread. Each ticker is downloaded once per
  scan and the shared rate limiter queues the requests, so a Polygon free scan runs
  at 5 requests a minute. estimate() says how long that takes before you start.
- Daily job: after each trading day's close (plus a delay for the data to settle),
  rescan your watchlist and the candidates and store the results under that date.
"""

import logging
import threading
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.config import settings
from api.database import SessionLocal
from api.models import LeadLagResult, LeadLagWatch, PerfMetric
from api.services import alpaca_data, market_calendar, polygon
from api.services.lead_lag import EngineParams, analyze_pair
from api.services.prices import normalize_ticker

logger = logging.getLogger(__name__)

DAILY_DELAY = timedelta(minutes=20)
CHECK_EVERY_SECONDS = 300
SOURCES = ("polygon", "alpaca")
DATA_ERRORS = (polygon.PolygonError, alpaca_data.AlpacaDataError)
DAILY_PARAMS = EngineParams(threshold_pct=0.3, lookback_seconds=60, max_wait_seconds=300)

# ---------------------------------------------------------------- candidate map

SECTOR_PEERS: Dict[str, List[str]] = {
    "Semiconductors": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "INTC", "AMAT", "LRCX", "MRVL"],
    "Megacap tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Energy": ["XOM", "CVX", "COP", "OXY"],
    "Autos and EVs": ["TSLA", "RIVN", "GM", "F"],
    "Payments": ["V", "MA", "PYPL"],
}
ETF_HOLDINGS: Dict[str, List[str]] = {
    "SMH": ["NVDA", "TSM", "AVGO", "AMD"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN"],
    "XLK": ["AAPL", "MSFT", "NVDA"],
    "XLF": ["JPM", "BAC", "WFC"],
    "XLE": ["XOM", "CVX"],
}
INDEX_PROXIES: List[Tuple[str, str, str]] = [
    ("SPY", "QQQ", "S&P 500 and Nasdaq 100 ETFs, stand-ins for ES and NQ futures"),
    ("SPY", "IWM", "S&P 500 and Russell 2000 ETFs, stand-ins for ES and RTY futures"),
    ("SPY", "DIA", "S&P 500 and Dow ETFs, stand-ins for ES and YM futures"),
    ("QQQ", "SMH", "Nasdaq 100 against its chip-heavy slice"),
]
ETFS = {"SPY", "QQQ", "IWM", "DIA", "SMH", "XLK", "XLF", "XLE"}
NOT_NASDAQ = {"JPM", "BAC", "WFC", "C", "GS", "MS", "XOM", "CVX", "COP", "OXY", "GM", "F", "V", "MA", "TSM"}


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted([a, b]))


def candidates(db: Session) -> List[Dict]:
    """Related pairs worth testing. Nasdaq stocks must be in the scanned universe, once it exists."""
    universe = {s for (s,) in db.query(PerfMetric.symbol).distinct()}

    def ok(symbol: str) -> bool:
        return symbol in ETFS or symbol in NOT_NASDAQ or not universe or symbol in universe

    out, seen = [], set()

    def add(a, b, group, why):
        key = pair_key(a, b)
        if a != b and ok(a) and ok(b) and key not in seen:
            seen.add(key)
            out.append({"a": a, "b": b, "group": group, "why": why})

    for group, names in SECTOR_PEERS.items():
        for other in names[1:]:
            add(names[0], other, group, f"Same sector; {names[0]} is the bellwether")
    for etf, holdings in ETF_HOLDINGS.items():
        for holding in holdings:
            add(etf, holding, "ETF and holding", f"{holding} is a top holding of {etf}")
    for a, b, why in INDEX_PROXIES:
        add(a, b, "Index", why)
    return out


# ---------------------------------------------------------------- data


def check_source(source: str) -> Optional[str]:
    """None if the source can be used, else what to fix."""
    if source not in SOURCES:
        return f"Unknown data source '{source}'. Use polygon or alpaca."
    if source == "polygon" and not polygon.configured():
        return "Lead-lag research needs POLYGON_API_KEY in .env, or set LEAD_LAG_SOURCE=alpaca to use your Alpaca keys."
    if source == "alpaca" and not alpaca_data.configured():
        return "Lead-lag research with Alpaca needs ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."
    return None


def resolution(source: str) -> Tuple[int, str]:
    if source == "polygon" and settings.POLYGON_SECOND_BARS:
        return 1, "second"
    return 60, "minute"


def date_range(days: int, now: Optional[datetime] = None) -> Tuple[date, date]:
    """The last `days` completed trading sessions, as calendar dates."""
    now = now or datetime.now(market_calendar.NEW_YORK)
    end = market_calendar.last_session_close(now).astimezone(market_calendar.NEW_YORK).date()
    start = end
    for _ in range(max(days, 1) - 1):
        start = market_calendar.previous_trading_day(start)
    return start, end


def load_closes(ticker: str, source: str, start: date, end: date, max_wait: Optional[float] = None) -> pd.Series:
    """Regular-session closes. Raises PolygonError or AlpacaDataError with a plain message."""
    if source == "alpaca":
        df = alpaca_data.get_intraday_bars(ticker, start, end, max_wait=max_wait)
    else:
        _, span = resolution(source)
        df = polygon.get_intraday_bars(ticker, start, end, timespan=span, max_wait=max_wait)
    return df["close"]


def _cached(ticker: str, source: str, start: date, end: date) -> bool:
    if source == "alpaca":
        return alpaca_data.is_cached(ticker, start, end)
    return polygon.is_cached(ticker, 1, resolution(source)[1], start, end)


def tickers_of(pairs: Iterable[Dict]) -> List[str]:
    return sorted({t for p in pairs for t in (p["a"], p["b"])})


def estimate(pairs: List[Dict], source: str, days: int, now: Optional[datetime] = None) -> Dict:
    start, end = date_range(days, now)
    tickers = tickers_of(pairs)
    todo = [t for t in tickers if not _cached(t, source, start, end)]
    if source == "alpaca":
        per, limiter = alpaca_data.estimate_requests(start, end), alpaca_data.limiter
    else:
        per, limiter = polygon.estimate_requests(start, end, resolution(source)[1]), polygon.limiter
    requests = per * len(todo)
    return {
        "pairs": len(pairs),
        "tickers": len(tickers),
        "cached_tickers": len(tickers) - len(todo),
        "requests": requests,
        "seconds": round(limiter.seconds_for(requests) + 2 * len(pairs)),  # plus a rough second or two per analysis
        "source": source,
        "resolution": resolution(source)[1],
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


# ---------------------------------------------------------------- storage


def strip_details(payload: Dict) -> Dict:
    return {k: v for k, v in payload.items() if k != "details"}


def store_result(db: Session, result: Dict, run_date: date, trigger: str) -> None:
    payload = {**result, "run_date": run_date.isoformat(), "last_updated": datetime.now(timezone.utc).isoformat()}
    row = (
        db.query(LeadLagResult)
        .filter(LeadLagResult.run_date == run_date, LeadLagResult.pair_key == result["pair_key"])
        .first()
    )
    if row is None:
        row = LeadLagResult(run_date=run_date, pair_key=result["pair_key"])
        db.add(row)
    row.leader, row.follower = result["leader"], result["follower"]
    row.trigger, row.source = trigger, result.get("source")
    row.confidence = result["confidence"]
    row.accuracy, row.events = result.get("accuracy"), result.get("events", 0)
    row.oos_accuracy = result["out_of_sample"]["accuracy"]
    row.oos_events = result["out_of_sample"]["events"]
    row.lag_seconds, row.avg_follower_move_pct = result.get("lag_seconds"), result.get("avg_follower_move_pct")
    row.payload = payload
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


def latest_results(db: Session) -> Dict:
    run_date = db.query(LeadLagResult.run_date).order_by(LeadLagResult.run_date.desc()).limit(1).scalar()
    if run_date is None:
        return {"run_date": None, "results": []}
    rows = db.query(LeadLagResult).filter(LeadLagResult.run_date == run_date).all()
    return {"run_date": run_date.isoformat(), "results": [strip_details(r.payload) for r in rows]}


def pair_history(db: Session, a: str, b: str) -> Dict:
    key = pair_key(a, b)
    rows = db.query(LeadLagResult).filter(LeadLagResult.pair_key == key).order_by(LeadLagResult.run_date).all()
    return {
        "pair_key": key,
        "history": [strip_details(r.payload) for r in rows],
        "latest": rows[-1].payload if rows else None,
    }


# ---------------------------------------------------------------- watchlist


class WatchlistFull(Exception):
    pass


def watchlist(db: Session) -> List[Dict]:
    return [{"a": w.ticker_a, "b": w.ticker_b} for w in db.query(LeadLagWatch).order_by(LeadLagWatch.id)]


def add_watch(db: Session, a: str, b: str) -> None:
    key = pair_key(a, b)
    if db.query(LeadLagWatch.id).filter(LeadLagWatch.pair_key == key).first():
        return
    if db.query(LeadLagWatch).count() >= settings.LEAD_LAG_MAX_WATCHLIST:
        raise WatchlistFull()
    db.add(LeadLagWatch(pair_key=key, ticker_a=a, ticker_b=b))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()


def remove_watch(db: Session, a: str, b: str) -> None:
    db.query(LeadLagWatch).filter(LeadLagWatch.pair_key == pair_key(a, b)).delete()
    db.commit()


def scan_pairs(db: Session, include_watchlist: bool, include_candidates: bool) -> List[Dict]:
    pairs, seen = [], set()
    for p in (watchlist(db) if include_watchlist else []) + (candidates(db) if include_candidates else []):
        key = pair_key(p["a"], p["b"])
        if key not in seen:
            seen.add(key)
            pairs.append(p)
    return pairs


# ---------------------------------------------------------------- scans


class ScanRunning(Exception):
    pass


class ScanManager:
    """At most one scan at a time. The rate limiter inside each data client does the queuing."""

    def __init__(self, session_factory=SessionLocal, loader: Optional[Callable] = None):
        self._session_factory = session_factory
        self._loader = loader
        self._lock = threading.Lock()
        self.job: Optional[Dict] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return bool(self.job and self.job["state"] == "running")

    def status(self) -> Optional[Dict]:
        return dict(self.job) if self.job else None

    def start(
        self,
        pairs: List[Dict],
        source: str,
        days: int,
        params: EngineParams,
        trigger: str = "manual",
        run_date: Optional[date] = None,
        background: bool = True,
    ) -> Dict:
        with self._lock:
            if self.running:
                raise ScanRunning()
            est = estimate(pairs, source, days)
            self.job = {
                "id": uuid.uuid4().hex[:12],
                "state": "running",
                "trigger": trigger,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "total_pairs": len(pairs),
                "done_pairs": 0,
                "errors": [],
                "estimate": est,
            }
            job = self.job
        run_date = run_date or date.fromisoformat(est["end"])
        args = (
            job,
            pairs,
            source,
            date.fromisoformat(est["start"]),
            date.fromisoformat(est["end"]),
            params,
            trigger,
            run_date,
        )
        if background:
            self._thread = threading.Thread(target=self._run, args=args, name="lead-lag-scan", daemon=True)
            self._thread.start()
        else:
            self._run(*args)
        return dict(job)

    def _run(self, job, pairs, source, start, end, params, trigger, run_date) -> None:
        loader = self._loader or load_closes
        bar_seconds, _ = resolution(source)
        try:
            closes: Dict[str, pd.Series] = {}
            for ticker in tickers_of(pairs):
                try:
                    closes[ticker] = loader(ticker, source, start, end)
                except DATA_ERRORS as exc:
                    job["errors"].append(f"{ticker}: {exc}")
            with self._session_factory() as db:
                for pair in pairs:
                    a, b = pair["a"], pair["b"]
                    if a in closes and b in closes:
                        try:
                            result = analyze_pair(a, closes[a], b, closes[b], params, bar_seconds=bar_seconds)
                            store_result(db, {**result, "source": source}, run_date, trigger)
                        except Exception as exc:  # noqa: BLE001 - one bad pair must not end the scan
                            logger.exception("Lead-lag analysis failed for %s/%s", a, b)
                            job["errors"].append(f"{a}/{b}: analysis failed ({type(exc).__name__})")
                    job["done_pairs"] += 1
            job["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Lead-lag scan failed")
            job["errors"].append(f"Scan failed ({type(exc).__name__})")
            job["state"] = "failed"
        finally:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(
                "Lead-lag %s scan %s: %d pairs, %d errors", trigger, job["state"], job["done_pairs"], len(job["errors"])
            )


manager = ScanManager()


def get_manager() -> ScanManager:
    return manager


# ---------------------------------------------------------------- daily job


def research_due(now: datetime, last_run: Optional[date]) -> Optional[date]:
    """The trading day to research now, or None. Runs once per trading day, after the close plus a delay."""
    local = now.astimezone(market_calendar.NEW_YORK)
    day = local.date()
    if not market_calendar.is_trading_day(day):
        return None
    if local < market_calendar.session_close(day) + DAILY_DELAY:
        return None
    if last_run is not None and last_run >= day:
        return None
    return day


def last_daily_run(db: Session) -> Optional[date]:
    return (
        db.query(LeadLagResult.run_date)
        .filter(LeadLagResult.trigger == "daily")
        .order_by(LeadLagResult.run_date.desc())
        .limit(1)
        .scalar()
    )


class DailyResearch:
    def __init__(self, scans: ScanManager = manager, session_factory=SessionLocal):
        self.scans = scans
        self._session_factory = session_factory
        self.last_run: Optional[date] = None
        self._stop = threading.Event()

    def run_if_due(self, now: Optional[datetime] = None) -> Optional[Dict]:
        now = now or datetime.now(timezone.utc)
        with self._session_factory() as db:
            stored = last_daily_run(db)
            last = max(d for d in (self.last_run, stored) if d is not None) if (self.last_run or stored) else None
            day = research_due(now, last)
            if day is None or self.scans.running:
                return None
            self.last_run = day  # one attempt per day, even if it fails
            source = settings.LEAD_LAG_SOURCE
            problem = check_source(source)
            if problem:
                logger.warning("Daily lead-lag research skipped: %s", problem)
                return None
            pairs = scan_pairs(db, include_watchlist=True, include_candidates=True)
        logger.info("Daily lead-lag research for %s: %d pairs", day, len(pairs))
        return self.scans.start(pairs, source, settings.LEAD_LAG_DAYS, DAILY_PARAMS, "daily", day, background=False)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_if_due()
            except Exception:  # noqa: BLE001 - keep the scheduler alive
                logger.exception("Daily lead-lag research failed")
            self._stop.wait(CHECK_EVERY_SECONDS)

    def start(self) -> None:
        self._stop.clear()
        threading.Thread(target=self._loop, name="lead-lag-research", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


daily = DailyResearch()


def start_scheduler() -> None:
    daily.start()


def stop_scheduler() -> None:
    daily.stop()


def clean_pair(a: str, b: str) -> Tuple[str, str]:
    sa, sb = normalize_ticker(a), normalize_ticker(b)
    if sa is None or sb is None:
        raise ValueError("Use 1 to 10 letters, numbers, dots, or dashes for each ticker.")
    if sa == sb:
        raise ValueError("Pick two different tickers.")
    return sa, sb
