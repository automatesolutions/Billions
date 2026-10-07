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

    # Optional. Only used if set; the outlier universe works without it.
    ALPHA_VANTAGE_API_KEY: str = ""

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
