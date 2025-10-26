"""
Valuation API endpoints using Black-Scholes-Merton model
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Optional
import logging
from api.services.black_scholes import bsm_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/valuation", tags=["valuation"])


@router.get("/{ticker}")
async def get_stock_valuation(
    ticker: str,
    days_back: int = Query(default=252, ge=30, le=1000, description="Days to look back for analysis")
) -> Dict:
    """
    Get Black-Scholes-Merton fair value analysis for a stock
    
    Args:
        ticker: Stock symbol
        days_back: Number of days to analyze (default: 252 trading days)
        
    Returns:
        Dictionary with fair value analysis
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Generating BSM valuation for {ticker}")
        
        # Perform Black-Scholes-Merton analysis
        result = bsm_analyzer.analyze_stock_valuation(ticker, days_back)
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze {ticker}: {result['error']}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in valuation endpoint for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}/fair-value")
async def get_fair_value_only(
    ticker: str,
    days_back: int = Query(default=252, ge=30, le=1000, description="Days to look back for analysis")
) -> Dict:
    """
    Get simplified fair value analysis
    
    Args:
        ticker: Stock symbol
        days_back: Number of days to analyze
        
    Returns:
        Dictionary with simplified fair value data
    """
    try:
        ticker = ticker.upper()
        logger.info(f"Generating fair value for {ticker}")
        
        # Perform analysis
        result = bsm_analyzer.analyze_stock_valuation(ticker, days_back)
        
        if "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to analyze {ticker}: {result['error']}"
            )
        
        # Return simplified version
        return {
            "ticker": result["ticker"],
            "current_price": result["current_price"],
            "fair_value": result["fair_value"],
            "valuation_status": result["valuation_status"],
            "valuation_color": result["valuation_color"],
            "valuation_ratio": result["valuation_ratio"],
            "volatility": result["volatility"],
            "analysis_date": result["analysis_date"]
        }
        
    except Exception as e:
        logger.error(f"Error in fair value endpoint for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for valuation service"""
    return {"status": "healthy", "service": "black_scholes_valuation"}
