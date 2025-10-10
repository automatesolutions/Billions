"""
Market Data Service
Handles fetching and caching market data from yfinance
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import csv

logger = logging.getLogger(__name__)


class MarketDataService:
    """Service for fetching and caching market data"""
    
    def __init__(self):
        self.cache_dir = Path(__file__).parent.parent.parent / "funda" / "cache"
        self.cache_dir.mkdir(exist_ok=True)
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
        self, 
        ticker: str, 
        period: str = "1y", 
        interval: str = "1d",
        use_cache: bool = True
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
    
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tickers (basic implementation)"""
        # This is a simplified version. In production, use a proper ticker database
        common_tickers = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "TSLA", "name": "Tesla, Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corporation"},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust"},
        ]
        
        query = query.upper()
        results = [
            t for t in common_tickers 
            if query in t["symbol"] or query in t["name"].upper()
        ]
        
        return results[:limit]


# Global service instance
market_data_service = MarketDataService()

