from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from api.services.enhanced_capitulation_detector import enhanced_capitulation_detector

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/capitulation/test")
async def test_capitulation():
    """Test endpoint to verify capitulation router is working"""
    return {
        "message": "Capitulation router is working",
        "status": "success",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/capitulation/screen")
async def screen_capitulation(
    limit: int = Query(20, ge=1, le=100, description="Maximum number of capitulation stocks to return")
):
    """Screen all NASDAQ stocks for capitulation signals"""
    try:
        logger.info(f"Screening NASDAQ stocks for capitulation signals (limit: {limit})")
        
        result = await enhanced_capitulation_detector.screen_nasdaq_enhanced(limit)
        
        if result.get("status") == "error":
            logger.error(f"Capitulation screening failed: {result.get('error')}")
            raise HTTPException(status_code=500, detail=result.get("error", "Screening failed"))
        
        logger.info(f"Capitulation screening completed successfully: {result.get('capitulation_count', 0)} stocks found")
        return result
        
    except Exception as e:
        logger.error(f"Error screening capitulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capitulation/summary")
async def get_capitulation_summary():
    """Get current market capitulation summary"""
    try:
        logger.info("Getting capitulation summary")
        
        summary = await enhanced_capitulation_detector.get_market_summary_enhanced()
        return summary
        
    except Exception as e:
        logger.error(f"Error getting capitulation summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capitulation/analyze/{symbol}")
async def analyze_stock_capitulation(symbol: str):
    """Analyze a specific stock for capitulation signals"""
    try:
        symbol = symbol.upper()
        logger.info(f"Analyzing {symbol} for capitulation signals")
        
        result = await enhanced_capitulation_detector.analyze_stock_enhanced(symbol)
        
        if result is None:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol} for capitulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capitulation/indicators")
async def get_capitulation_indicators():
    """Get explanation of enhanced capitulation indicators"""
    return {
        "indicators": {
            "volume_spike_20": {
                "description": "Volume spikes 2.5x+ above 20-day average",
                "weight": 3,
                "significance": "High selling pressure"
            },
            "volume_elevated_20": {
                "description": "Volume elevated 1.8x+ above 20-day average",
                "weight": 2,
                "significance": "Elevated selling pressure"
            },
            "volume_spike_50": {
                "description": "Volume spikes 2x+ above 50-day average",
                "weight": 2,
                "significance": "Sustained selling pressure"
            },
            "rsi_extreme_oversold": {
                "description": "RSI drops below 25 (extreme oversold)",
                "weight": 4,
                "significance": "Extreme oversold condition"
            },
            "rsi_oversold": {
                "description": "RSI drops below 30 (oversold)",
                "weight": 3,
                "significance": "Oversold condition"
            },
            "rsi_near_oversold": {
                "description": "RSI drops below 35 (near oversold)",
                "weight": 2,
                "significance": "Approaching oversold"
            },
            "rsi_weak": {
                "description": "RSI drops below 40 (weak momentum)",
                "weight": 1,
                "significance": "Weak momentum"
            },
            "macd_bearish": {
                "description": "MACD shows bearish momentum",
                "weight": 2,
                "significance": "Downward momentum confirmation"
            },
            "stoch_oversold": {
                "description": "Stochastic oscillator oversold",
                "weight": 2,
                "significance": "Oversold momentum"
            },
            "williams_oversold": {
                "description": "Williams %R below -80",
                "weight": 2,
                "significance": "Extreme oversold momentum"
            },
            "extreme_down_day": {
                "description": "Single day drop of 8%+",
                "weight": 4,
                "significance": "Extreme price decline"
            },
            "large_down_day": {
                "description": "Single day drop of 5%+",
                "weight": 3,
                "significance": "Large price decline"
            },
            "moderate_down_day": {
                "description": "Single day drop of 3%+",
                "weight": 2,
                "significance": "Moderate price decline"
            },
            "small_down_day": {
                "description": "Single day drop of 1.5%+",
                "weight": 1,
                "significance": "Small price decline"
            },
            "extreme_3d_decline": {
                "description": "3-day decline of 15%+",
                "weight": 4,
                "significance": "Extreme multi-day decline"
            },
            "large_3d_decline": {
                "description": "3-day decline of 10%+",
                "weight": 3,
                "significance": "Large multi-day decline"
            },
            "moderate_3d_decline": {
                "description": "3-day decline of 5%+",
                "weight": 2,
                "significance": "Moderate multi-day decline"
            },
            "extreme_5d_decline": {
                "description": "5-day decline of 20%+",
                "weight": 4,
                "significance": "Extreme weekly decline"
            },
            "large_5d_decline": {
                "description": "5-day decline of 12%+",
                "weight": 3,
                "significance": "Large weekly decline"
            },
            "moderate_5d_decline": {
                "description": "5-day decline of 7%+",
                "weight": 2,
                "significance": "Moderate weekly decline"
            },
            "far_below_sma20": {
                "description": "Price 10%+ below 20-day SMA",
                "weight": 3,
                "significance": "Far below short-term trend"
            },
            "below_sma20": {
                "description": "Price 5%+ below 20-day SMA",
                "weight": 2,
                "significance": "Below short-term trend"
            },
            "near_sma20": {
                "description": "Price 2%+ below 20-day SMA",
                "weight": 1,
                "significance": "Near short-term trend"
            },
            "far_below_sma50": {
                "description": "Price 15%+ below 50-day SMA",
                "weight": 3,
                "significance": "Far below medium-term trend"
            },
            "below_sma50": {
                "description": "Price 8%+ below 50-day SMA",
                "weight": 2,
                "significance": "Below medium-term trend"
            },
            "far_below_sma200": {
                "description": "Price 20%+ below 200-day SMA",
                "weight": 4,
                "significance": "Far below long-term trend"
            },
            "below_sma200": {
                "description": "Price 10%+ below 200-day SMA",
                "weight": 3,
                "significance": "Below long-term trend"
            },
            "high_volatility": {
                "description": "ATR volatility 8%+ of price",
                "weight": 2,
                "significance": "High price volatility"
            },
            "elevated_volatility": {
                "description": "ATR volatility 5%+ of price",
                "weight": 1,
                "significance": "Elevated price volatility"
            },
            "hammer_pattern": {
                "description": "Hammer candlestick pattern",
                "weight": 2,
                "significance": "Potential reversal signal"
            },
            "long_lower_tail": {
                "description": "Long lower tail (30%+ of range)",
                "weight": 1,
                "significance": "Potential support"
            },
            "doji_pattern": {
                "description": "Doji candlestick pattern",
                "weight": 1,
                "significance": "Market indecision"
            },
            "gap_down": {
                "description": "Gap down 5%+ from previous close",
                "weight": 2,
                "significance": "Overnight selling pressure"
            },
            "lower_lows_pattern": {
                "description": "3+ consecutive lower lows",
                "weight": 2,
                "significance": "Downtrend continuation"
            }
        },
        "scoring": {
            "threshold": 3,
            "max_score": 20,
            "description": "Enhanced detection: Stocks with score >= 3 are considered in capitulation"
        },
        "enhancements": {
            "more_sensitive": "Lowered thresholds for more detection",
            "multi_timeframe": "Analysis across multiple time periods",
            "comprehensive_coverage": "Extended NASDAQ stock coverage",
            "advanced_indicators": "Additional technical indicators",
            "confidence_scoring": "Dynamic confidence calculation"
        },
        "market_indicators": {
            "vix": "Volatility index - higher values indicate fear",
            "spy_change": "S&P 500 change for market context",
            "qqq_change": "NASDAQ 100 change for tech context",
            "market_condition": "Overall market fear level",
            "market_trend": "Current market trend direction"
        }
    }

@router.get("/capitulation/stats")
async def get_capitulation_stats():
    """Get capitulation statistics"""
    try:
        # Get recent screening results
        result = await enhanced_capitulation_detector.screen_nasdaq_enhanced(50)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Stats failed"))
        
        # Calculate additional statistics
        capitulation_stocks = result.get("capitulation_stocks", [])
        
        # Sector breakdown
        sector_counts = {}
        for stock in capitulation_stocks:
            sector = stock.get("sector", "Unknown")
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
        # Score distribution
        scores = [stock.get("capitulation_score", 0) for stock in capitulation_stocks]
        avg_score = sum(scores) / len(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        return {
            "total_analyzed": result.get("total_stocks_analyzed", 0),
            "capitulation_count": result.get("capitulation_count", 0),
            "capitulation_rate": result.get("capitulation_rate", 0),
            "sector_breakdown": sector_counts,
            "score_statistics": {
                "average_score": round(avg_score, 2),
                "max_score": max_score,
                "score_distribution": {
                    "extreme_capitulation": len([s for s in scores if s >= 8]),
                    "high_capitulation": len([s for s in scores if 6 <= s < 8]),
                    "moderate_capitulation": len([s for s in scores if 4 <= s < 6]),
                    "low_capitulation": len([s for s in scores if 3 <= s < 4])
                }
            },
            "analysis_date": datetime.now().isoformat(),
            "analysis_type": "Enhanced Capitulation Detection"
        }
        
    except Exception as e:
        logger.error(f"Error getting capitulation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
