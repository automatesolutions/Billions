"""
Configuration for the BILLIONS API.

Every value can be overridden with an environment variable of the same name.
"""

from pathlib import Path
from typing import Annotated, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore", case_sensitive=True)

    APP_NAME: str = "BILLIONS API"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Comma-separated list, e.g. "https://billions.vercel.app,http://localhost:3000"
    CORS_ORIGINS: Annotated[List[str], NoDecode] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    DATABASE_URL: str = f"sqlite:///{(ROOT_DIR / 'data' / 'billions.db').as_posix()}"
    CACHE_DIR: Path = ROOT_DIR / "data" / "cache"

    # Outlier refresh: background scheduler on/off, and interval while the market is open.
    OUTLIER_SCHEDULER: bool = True
    REFRESH_INTERVAL_MINUTES: int = 30

    # Quant page: background recording of watched (ticker, interval) pairs, and how many can be watched.
    QUANT_SCHEDULER: bool = True
    QUANT_MAX_WATCHES: int = 20

    # Per-IP rate limits (slowapi syntax). Analysis is compute-heavy, so it gets a lower limit.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_DEFAULT: str = "120/minute"
    RATE_LIMIT_ANALYSIS: str = "20/minute"

    # Optional. Only used if set; the outlier universe works without it.
    ALPHA_VANTAGE_API_KEY: str = ""

    # Optional. Free Alpaca keys turn on live top-of-book quotes (microstructure panel).
    # The free plan's "iex" feed is quotes from the IEX exchange only, not the whole market.
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_DATA_URL: str = "https://data.alpaca.markets"
    ALPACA_FEED: str = "iex"

    # Paper trading bot. Orders only ever go to Alpaca's paper endpoint; the broker
    # service refuses any other host, so changing this cannot turn on live trading.
    ALPACA_TRADING_URL: str = "https://paper-api.alpaca.markets"

    # Optional. Polygon.io market data for the bot and lead-lag research.
    # The free plan allows 5 requests per minute and its data is delayed.
    POLYGON_API_KEY: str = ""
    POLYGON_URL: str = "https://api.polygon.io"
    POLYGON_REQUESTS_PER_MINUTE: int = 5
    # Set these only if your Polygon plan includes them. The free plan has neither.
    POLYGON_SECOND_BARS: bool = False  # second-level aggregates for lead-lag research
    POLYGON_REALTIME: bool = False  # real-time WebSocket trades for the lead-lag bot
    POLYGON_WS_URL: str = "wss://socket.polygon.io/stocks"

    # Alpaca's real-time stream. The free plan streams trades from the IEX exchange.
    ALPACA_STREAM_URL: str = "wss://stream.data.alpaca.markets/v2"
    ALPACA_DATA_REQUESTS_PER_MINUTE: int = 190  # free plan allows 200

    # Paper bot. Off by default. When BOT_ADMIN_TOKEN is set, every bot endpoint needs it
    # in an X-Bot-Token header. Set one whenever the API is reachable by anyone but you.
    BOT_ENABLED: bool = False
    BOT_ADMIN_TOKEN: str = ""
    BOT_MAX_POSITION_DOLLARS: float = 1000.0
    BOT_MAX_ORDERS_PER_DAY: int = 20

    # Lead-lag research: daily rescan after the close, its data source, and history length.
    LEAD_LAG_SCHEDULER: bool = True
    LEAD_LAG_SOURCE: str = "polygon"  # polygon or alpaca
    LEAD_LAG_DAYS: int = 60
    LEAD_LAG_MAX_WATCHLIST: int = 50

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
