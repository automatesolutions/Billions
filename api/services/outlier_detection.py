"""
Outlier Detection Service
Handles outlier detection for different trading strategies
"""

import pandas as pd
from scipy.stats import zscore
import logging
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from funda.outlier_engine import run_outlier_detection, STRATEGIES
from api.database import get_db
from db.models import PerfMetric

logger = logging.getLogger(__name__)


class OutlierDetectionService:
    """Service for detecting outliers in stock performance"""
    
    def __init__(self):
        self.strategies = STRATEGIES
        logger.info("Outlier detection service initialized")
    
    def refresh_outliers(self, strategy: str, tickers: Optional[List[str]] = None) -> Dict:
        """
        Refresh outlier detection for a strategy
        
        Args:
            strategy: One of 'scalp', 'swing', 'longterm'
            tickers: Optional list of tickers, if None will fetch NASDAQ tickers
        
        Returns:
            Dict with refresh status
        """
        try:
            if strategy not in self.strategies:
                raise ValueError(f"Invalid strategy: {strategy}")
            
            logger.info(f"Starting outlier refresh for {strategy}")
            
            # Run outlier detection (from existing code)
            run_outlier_detection(strategy, tickers)
            
            # Count results
            from api.database import SessionLocal
            with SessionLocal() as db:
                total_count = db.query(PerfMetric).filter(
                    PerfMetric.strategy == strategy
                ).count()
                
                outlier_count = db.query(PerfMetric).filter(
                    PerfMetric.strategy == strategy,
                    PerfMetric.is_outlier == True
                ).count()
            
            logger.info(f"Outlier refresh complete: {outlier_count}/{total_count} outliers")
            
            return {
                "strategy": strategy,
                "total_stocks": total_count,
                "outliers_found": outlier_count,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error refreshing outliers for {strategy}: {e}")
            return {
                "strategy": strategy,
                "status": "error",
                "error": str(e)
            }
    
    def get_strategy_info(self, strategy: str) -> Optional[Dict]:
        """Get information about a strategy"""
        if strategy not in self.strategies:
            return None
        
        x_label, y_label, back_x, back_y, min_market_cap = self.strategies[strategy]
        
        return {
            "strategy": strategy,
            "x_period": x_label,
            "y_period": y_label,
            "lookback_x_days": back_x,
            "lookback_y_days": back_y,
            "min_market_cap": min_market_cap,
        }
    
    def get_all_strategies(self) -> List[Dict]:
        """Get all available strategies"""
        return [
            self.get_strategy_info(strategy) 
            for strategy in self.strategies.keys()
        ]


# Global service instance
outlier_service = OutlierDetectionService()

