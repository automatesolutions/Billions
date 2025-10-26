"""
Configuration management for BILLIONS API
"""

from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "BILLIONS API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Database
    DATABASE_URL: str = "sqlite:///./billions.db"
    
    # Get the absolute path to the database file in the parent directory
    @property
    def database_path(self) -> str:
        """Get absolute path to database"""
        return str(Path(__file__).parent.parent / "billions.db")
    
    # External APIs
    ALPHA_VANTAGE_API_KEY: str = ""
    FRED_API_KEY: str = ""
    POLYGON_API_KEY: str = ""
    
    # Enhanced News Service API Keys
    NEWS_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # Alpaca Trading API (set via .env)
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets/v2"
    
    # HFT optional settings (can be overridden via .env)
    HFT_EDGE_THRESHOLD: float = 0.0
    HFT_MAX_POSITION_SIZE: int = 0
    HFT_MAX_DAILY_LOSS: float = 0.0
    HFT_MAX_LEVERAGE: float = 0.0
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Models
    MODEL_PATH: str = "../funda/model"
    CACHE_PATH: str = "../funda/cache"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

