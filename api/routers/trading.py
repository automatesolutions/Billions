from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import asyncio
import time

from api.services.trading_service import trading_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.on_event("startup")
async def startup_event():
    """Initialize trading services on startup"""
    try:
        success = await trading_service.initialize()
        if not success:
            logger.warning("Trading services not fully initialized - check API keys")
    except Exception as e:
        logger.error(f"Error initializing trading services: {e}")

@router.get("/trading/status")
async def get_trading_status():
    """Get trading service status"""
    try:
        return {
            "connected": trading_service.is_connected,
            "polygon_available": trading_service.polygon.api_key is not None,
            "alpaca_available": trading_service.alpaca.api_key is not None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trading/account")
async def get_account_info():
    """Get Alpaca account information"""
    try:
        account_info = await trading_service.alpaca.get_account()
        if account_info.get("status") == "success":
            return account_info
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch account info")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trading/positions")
async def get_positions():
    """Get current positions from Alpaca"""
    try:
        positions = await trading_service.alpaca.get_positions()
        return {
            "positions": positions,
            "total_positions": len(positions),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trading/orders")
async def get_orders(status: str = "all"):
    """Get order history from Alpaca"""
    try:
        orders = await trading_service.alpaca.get_orders(status)
        return {
            "orders": orders,
            "total_orders": len(orders),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote from Polygon"""
    try:
        quote = await trading_service.polygon.get_real_time_quote(symbol.upper())
        return quote
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading/orderbook/{symbol}")
async def get_orderbook(symbol: str):
    """Get orderbook data from Polygon"""
    try:
        orderbook = await trading_service.polygon.get_orderbook(symbol.upper())
        return orderbook
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading/market-data")
async def get_market_data(symbols: List[str]):
    """Get market data for multiple symbols"""
    try:
        if len(symbols) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        market_data = await trading_service.get_market_data(symbols)
        return market_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading/execute")
async def execute_trade(
    symbol: str,
    qty: int,
    side: str,
    order_type: str = "market"
):
    """Execute a paper trade through Alpaca"""
    try:
        if side not in ["buy", "sell"]:
            raise HTTPException(status_code=400, detail="Side must be 'buy' or 'sell'")
        
        if qty <= 0:
            raise HTTPException(status_code=400, detail="Quantity must be positive")
        
        if order_type not in ["market", "limit", "stop", "stop_limit"]:
            raise HTTPException(status_code=400, detail="Invalid order type")
        
        result = await trading_service.execute_trade(
            symbol.upper(), 
            qty, 
            side, 
            order_type
        )
        
        if result.get("status") == "success":
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Trade execution failed"))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trading/portfolio")
async def get_portfolio():
    """Get complete portfolio data"""
    try:
        portfolio_data = await trading_service.get_portfolio_data()
        return portfolio_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trading/market-status")
async def get_market_status():
    """Get current market status"""
    try:
        market_status = await trading_service.polygon.get_market_status()
        return market_status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading/sync-portfolio")
async def sync_portfolio():
    """Sync portfolio data between website and Alpaca"""
    try:
        # Get current positions from Alpaca
        positions = await trading_service.alpaca.get_positions()
        account = await trading_service.alpaca.get_account()
        
        # This would typically update a local database
        # For now, we'll just return the data
        return {
            "positions": positions,
            "account": account,
            "sync_timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading/bulk-quotes")
async def get_bulk_quotes(symbols: List[str]):
    """Get quotes for multiple symbols efficiently"""
    try:
        if len(symbols) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed")
        
        # Create tasks for all symbols
        tasks = [trading_service.polygon.get_real_time_quote(symbol.upper()) for symbol in symbols]
        quotes = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        api_failed = False
        
        for i, quote in enumerate(quotes):
            if isinstance(quote, Exception):
                api_failed = True
                # Generate mock data when API fails
                symbol = symbols[i].upper()
                import random
                base_price = 100 + (hash(symbol) % 26) * 10
                variation = (random.random() - 0.5) * 2
                current_price = round(base_price + variation, 2)
                spread = round(0.01 + random.random() * 0.05, 2)
                
                results.append({
                    "symbol": symbol,
                    "bid": round(current_price - spread / 2, 2),
                    "ask": round(current_price + spread / 2, 2),
                    "last_price": current_price,
                    "timestamp": int(time.time() * 1000),
                    "status": "mock_data"
                })
            else:
                results.append(quote)
        
        return {
            "quotes": results,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
