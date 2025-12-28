"""
ML Prediction endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import logging

from api.services.predictions import prediction_service
from api.services.market_data import market_data_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["ML Predictions"])


@router.get("/{ticker}")
async def get_prediction(
    ticker: str,
    days: int = Query(default=30, ge=1, le=30, description="Number of days to predict")
):
    """
    Generate ML prediction for a stock ticker
    
    - **ticker**: Stock symbol (e.g., TSLA, AAPL)
    - **days**: Number of days to predict (1-30)
    """
    ticker = ticker.upper()
    
    try:
        logger.info(f"Generating prediction for {ticker}, {days} days")
        
        # Generate prediction
        result = prediction_service.generate_prediction(ticker, days)
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate prediction for {ticker}"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info/{ticker}")
async def get_ticker_info(ticker: str):
    """
    Get detailed information about a stock ticker
    """
    ticker = ticker.upper()
    
    try:
        info = market_data_service.get_stock_info(ticker)
        
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not find information for {ticker}"
            )
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting info for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_tickers(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Search for stock tickers
    
    - **q**: Search query (ticker symbol or company name)
    - **limit**: Maximum number of results
    """
    try:
        results = market_data_service.search_tickers(q, limit)
        return {"query": q, "results": results}
    except Exception as e:
        logger.error(f"Error searching tickers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

