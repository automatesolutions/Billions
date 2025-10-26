from fastapi import APIRouter, HTTPException
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/{ticker}/historical")
async def get_historical_data(ticker: str, period: str = "6mo"):
    """
    Get historical stock data for a ticker symbol.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'TSLA', 'AAPL')
        period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
        List of historical OHLC data points
    """
    try:
        logger.info(f"Fetching historical data for {ticker} with period {period}")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist = stock.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {ticker}")
        
        # Convert to list of dictionaries
        historical_data = []
        for date, row in hist.iterrows():
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']) if 'Volume' in row else 0
            })
        
        logger.info(f"Successfully fetched {len(historical_data)} days of historical data for {ticker}")
        
        return {
            "ticker": ticker,
            "period": period,
            "data": historical_data,
            "count": len(historical_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")

@router.get("/{ticker}/historical/range")
async def get_historical_data_range(
    ticker: str, 
    start_date: str, 
    end_date: str
):
    """
    Get historical stock data for a specific date range.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    
    Returns:
        List of historical OHLC data points
    """
    try:
        logger.info(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data for date range
        hist = stock.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data found for {ticker} in date range {start_date} to {end_date}")
        
        # Convert to list of dictionaries
        historical_data = []
        for date, row in hist.iterrows():
            historical_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']) if 'Volume' in row else 0
            })
        
        logger.info(f"Successfully fetched {len(historical_data)} days of historical data for {ticker}")
        
        return {
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "data": historical_data,
            "count": len(historical_data)
        }
        
    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching historical data: {str(e)}")
