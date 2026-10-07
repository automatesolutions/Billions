"""
Market Data Service
Handles fetching and caching market data from yfinance
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

from api.config import settings

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for fetching and caching market data"""

    def __init__(self):
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Market data service initialized. Cache: {self.cache_dir}")

    def get_cache_path(self, ticker: str, interval: str = "1d") -> Path:
        """Get cache file path for a ticker"""
        return self.cache_dir / f"{ticker}_{interval}.csv"

    def is_cache_valid(self, ticker: str, interval: str = "1d", max_age_hours: int = 1) -> bool:
        """Check if cached data is still valid"""
        cache_path = self.get_cache_path(ticker, interval)

        if not cache_path.exists():
            return False

        # Check file age
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_time

        return age < timedelta(hours=max_age_hours)

    def load_from_cache(self, ticker: str, interval: str = "1d") -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            cache_path = self.get_cache_path(ticker, interval)

            if not cache_path.exists():
                return None

            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {ticker} from cache ({len(df)} rows)")
            return df

        except Exception as e:
            logger.error(f"Error loading cache for {ticker}: {e}")
            return None

    def save_to_cache(self, ticker: str, df: pd.DataFrame, interval: str = "1d"):
        """Save data to cache"""
        try:
            cache_path = self.get_cache_path(ticker, interval)
            df.to_csv(cache_path)
            logger.info(f"Saved {ticker} to cache ({len(df)} rows)")
        except Exception as e:
            logger.error(f"Error saving cache for {ticker}: {e}")

    def fetch_stock_data(
        self, ticker: str, period: str = "1y", interval: str = "1d", use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data with caching"""
        try:
            # Check cache first
            if use_cache and self.is_cache_valid(ticker, interval):
                df = self.load_from_cache(ticker, interval)
                if df is not None:
                    return df

            # Fetch from yfinance
            logger.info(f"Fetching {ticker} from yfinance (period={period}, interval={interval})")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None

            # Save to cache
            if use_cache:
                self.save_to_cache(ticker, df, interval)

            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return None

    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """Get stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                "symbol": ticker,
                "name": info.get("longName", ticker),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
            }

        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return None


# Global service instance
market_data_service = MarketDataService()
