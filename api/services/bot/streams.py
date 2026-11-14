"""
Real-time trade streams over WebSocket for the lead-lag strategy.

Polling bars can't see a move that lasts a few seconds, so lead-lag listens to
every trade as it prints. Two sources share one interface (PriceStream):

- AlpacaStream (default): free plan, trades from the IEX exchange only. IEX is a
  small share of volume, so prints are sparser than the whole market.
- PolygonStream: only with a real-time Polygon plan (POLYGON_REALTIME=true).
  Polygon's free plan is delayed by 15 minutes, which makes a seconds-level
  strategy meaningless, so make_stream refuses it and logs a warning.

Each stream runs its own asyncio event loop in a background thread and calls
on_tick(Tick) for every trade. It reconnects with backoff if the socket drops.
"""

import asyncio
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, ClassVar, Dict, List, Optional, Tuple

import websockets

from api.config import settings
from api.services.quotes import _parse_time as parse_rfc3339

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 30
RECV_TIMEOUT_SECONDS = 1.0

POLYGON_DELAYED = (
    "Lead-lag needs real-time prices, and Polygon's free plan is delayed by 15 minutes, so the Polygon stream "
    "is off. Set POLYGON_REALTIME=true only if your Polygon plan includes real-time data. Otherwise use the "
    "Alpaca IEX stream."
)


@dataclass(frozen=True)
class Tick:
    symbol: str
    price: float
    ts: datetime  # when the trade printed at the exchange (UTC)
    received_at: datetime  # when we received it (UTC)


class StreamNotAvailable(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PriceStream(ABC):
    source: ClassVar[str]

    def __init__(self, symbols: List[str], on_tick: Callable[[Tick], None], connect=None):
        self.symbols = sorted(set(symbols))
        self.on_tick = on_tick
        self._connect = connect or websockets.connect
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.last_message_at: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.ticks = 0
        self.fatal = False  # bad keys or plan: stop retrying

    @property
    @abstractmethod
    def url(self) -> str: ...

    @abstractmethod
    def auth_message(self) -> Dict: ...

    @abstractmethod
    def subscribe_message(self) -> Dict: ...

    @abstractmethod
    def parse(self, raw, received_at: datetime) -> Tuple[List[Tick], List[Tuple[str, str]]]:
        """Return (ticks, notes). A note is (level, text); level "fatal" stops the stream."""

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), name=f"{self.source}-stream", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def status(self) -> Dict:
        return {
            "source": self.source,
            "symbols": self.symbols,
            "connected": self.connected,
            "ticks": self.ticks,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "last_error": self.last_error,
        }

    def handle(self, raw) -> None:
        received = _utcnow()
        self.last_message_at = received
        ticks, notes = self.parse(raw, received)
        for level, text in notes:
            if level in ("error", "fatal"):
                self.last_error = text
                logger.warning("%s stream: %s", self.source, text)
                self.fatal = self.fatal or level == "fatal"
            else:
                logger.info("%s stream: %s", self.source, text)
        for tick in ticks:
            self.ticks += 1
            try:
                self.on_tick(tick)
            except Exception:  # noqa: BLE001 - one bad callback must not kill the stream
                logger.exception("on_tick failed")

    async def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set() and not self.fatal:
            try:
                async with self._connect(self.url, open_timeout=10, ping_interval=20) as ws:
                    await ws.send(json.dumps(self.auth_message()))
                    await ws.send(json.dumps(self.subscribe_message()))
                    self.connected = True
                    backoff = 1.0
                    while not self._stop.is_set() and not self.fatal:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SECONDS)
                        except asyncio.TimeoutError:
                            continue
                        self.handle(raw)
            except Exception as exc:  # noqa: BLE001 - reconnect on any socket failure
                self.last_error = f"Connection lost ({type(exc).__name__}). Reconnecting."
                logger.warning("%s stream: %s", self.source, self.last_error)
            finally:
                self.connected = False
            waited = 0.0
            while waited < backoff and not self._stop.is_set() and not self.fatal:
                await asyncio.sleep(0.2)
                waited += 0.2
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)


def _messages(raw) -> list:
    data = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    return data if isinstance(data, list) else [data]


class AlpacaStream(PriceStream):
    source = "alpaca"

    @property
    def url(self) -> str:
        return f"{settings.ALPACA_STREAM_URL.rstrip('/')}/{settings.ALPACA_FEED}"

    def auth_message(self) -> Dict:
        return {"action": "auth", "key": settings.ALPACA_API_KEY, "secret": settings.ALPACA_SECRET_KEY}

    def subscribe_message(self) -> Dict:
        return {"action": "subscribe", "trades": self.symbols}

    def parse(self, raw, received_at):
        ticks, notes = [], []
        for msg in _messages(raw):
            kind = msg.get("T")
            if kind == "t":
                ts = parse_rfc3339(msg.get("t"))
                price = msg.get("p")
                if ts and price:
                    ticks.append(Tick(msg.get("S"), float(price), ts, received_at))
            elif kind == "error":
                # 402 auth failed, 406 connection limit (the free plan allows one), 409 plan doesn't include the feed.
                code = msg.get("code")
                level = "fatal" if code in (402, 404, 409) else "error"
                notes.append((level, f"Alpaca error {code}: {msg.get('msg')}"))
            elif kind in ("success", "subscription"):
                notes.append(("info", str(msg.get("msg") or msg.get("trades"))))
        return ticks, notes


class PolygonStream(PriceStream):
    source = "polygon"

    @property
    def url(self) -> str:
        return settings.POLYGON_WS_URL

    def auth_message(self) -> Dict:
        return {"action": "auth", "params": settings.POLYGON_API_KEY}

    def subscribe_message(self) -> Dict:
        return {"action": "subscribe", "params": ",".join(f"T.{s}" for s in self.symbols)}

    def parse(self, raw, received_at):
        ticks, notes = [], []
        for msg in _messages(raw):
            ev = msg.get("ev")
            if ev == "T" and msg.get("p") and msg.get("t"):
                ts = datetime.fromtimestamp(msg["t"] / 1000, tz=timezone.utc)
                ticks.append(Tick(msg.get("sym"), float(msg["p"]), ts, received_at))
            elif ev == "status":
                status = msg.get("status")
                level = "fatal" if status in ("auth_failed", "max_connections") else "info"
                notes.append((level, f"Polygon {status}: {msg.get('message')}"))
        return ticks, notes


def make_stream(source: str, symbols: List[str], on_tick: Callable[[Tick], None], connect=None) -> PriceStream:
    if source == "polygon":
        if not settings.POLYGON_REALTIME:
            logger.warning(POLYGON_DELAYED)
            raise StreamNotAvailable(POLYGON_DELAYED)
        if "delayed" in settings.POLYGON_WS_URL:
            logger.warning("POLYGON_WS_URL points at Polygon's delayed feed. Lead-lag signals from it are 15 minutes late.")
        if not settings.POLYGON_API_KEY:
            raise StreamNotAvailable("The Polygon stream needs POLYGON_API_KEY in .env.")
        return PolygonStream(symbols, on_tick, connect)
    if source == "alpaca":
        if not (settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY):
            raise StreamNotAvailable("The Alpaca stream needs ALPACA_API_KEY and ALPACA_SECRET_KEY in .env.")
        return AlpacaStream(symbols, on_tick, connect)
    raise StreamNotAvailable(f"Unknown stream source '{source}'. Use alpaca or polygon.")
