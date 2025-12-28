"""
HFT Trading Router
API endpoints for High-Frequency Trading functionality
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from api.services.trading_service import trading_service

router = APIRouter(prefix="/hft", tags=["HFT Trading"])

# Request Models
class HFTOrderRequest(BaseModel):
    """HFT Order Request"""
    order_type: str  # market, limit, twap, vwap
    symbol: str
    side: str  # buy, sell
    quantity: int
    price: Optional[float] = None
    time_in_force: Optional[str] = None  # DAY, GTC, FOK, IOC, OPG, CLS
    duration_minutes: Optional[int] = None
    interval_seconds: Optional[int] = None
    volume_weight: Optional[float] = None

class HFTConfigUpdate(BaseModel):
    """HFT Configuration Update"""
    edge_threshold: Optional[float] = None
    max_position_size: Optional[int] = None
    max_daily_loss: Optional[float] = None
    max_leverage: Optional[float] = None
    trading_symbols: Optional[List[str]] = None

# Endpoints

@router.get("/status")
async def get_hft_status():
    """Get HFT engine status"""
    try:
        status = trading_service.get_hft_status()
        return {
            "status": "success",
            "data": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_hft_performance():
    """Get HFT performance metrics"""
    try:
        metrics = trading_service.get_hft_performance_metrics()
        if not metrics:
            return {
                "status": "success",
                "data": {
                    "message": "No performance data available",
                    "total_trades": 0,
                    "total_pnl": 0.0
                },
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "status": "success",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start")
async def start_hft_engine():
    """Start the HFT engine"""
    try:
        if not trading_service.hft_available:
            raise HTTPException(
                status_code=400, 
                detail="HFT engine not available. Please check configuration."
            )
        
        success = await trading_service.start_hft_engine()
        
        if success:
            return {
                "status": "success",
                "message": "HFT engine started successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start HFT engine")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_hft_engine():
    """Stop the HFT engine"""
    try:
        await trading_service.stop_hft_engine()
        
        return {
            "status": "success",
            "message": "HFT engine stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/{order_id}/status")
async def get_order_status(order_id: str):
    """Get the status of a specific order"""
    try:
        if not trading_service.hft_manager:
            raise HTTPException(status_code=404, detail="HFT engine not available")
        
        order_status = await trading_service.hft_manager.get_order_status(order_id)
        
        if not order_status:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "order_status": order_status.get("status"),
                "symbol": order_status.get("symbol"),
                "side": order_status.get("side"),
                "order_type": order_status.get("order_type"),
                "quantity": order_status.get("qty"),
                "filled_qty": order_status.get("filled_qty"),
                "filled_avg_price": order_status.get("filled_avg_price"),
                "limit_price": order_status.get("limit_price"),
                "time_in_force": order_status.get("time_in_force"),
                "created_at": order_status.get("created_at"),
                "updated_at": order_status.get("updated_at")
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orders/open")
async def get_open_orders():
    """Get all open orders"""
    try:
        if not trading_service.hft_manager:
            raise HTTPException(status_code=404, detail="HFT engine not available")
        
        open_orders = await trading_service.hft_manager.get_open_orders()
        
        return {
            "status": "success",
            "data": {
                "orders": open_orders,
                "count": len(open_orders)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/orders/all")
async def clear_all_orders():
    """Cancel all open orders"""
    try:
        if not trading_service.hft_manager:
            raise HTTPException(status_code=404, detail="HFT engine not available")
        
        cancelled_count = await trading_service.hft_manager.cancel_all_orders()
        
        return {
            "status": "success",
            "data": {
                "cancelled_count": cancelled_count,
                "message": f"Successfully cancelled {cancelled_count} orders"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str):
    """Cancel a specific order"""
    try:
        if not trading_service.hft_manager:
            raise HTTPException(status_code=404, detail="HFT engine not available")
        
        success = await trading_service.hft_manager.cancel_order(order_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Order {order_id} cancelled successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to cancel order {order_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orders")
async def submit_hft_order(request: HFTOrderRequest):
    """Submit an HFT order"""
    try:
        # Validate order type
        valid_order_types = ["market", "limit", "twap", "vwap"]
        if request.order_type not in valid_order_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid order type. Must be one of: {', '.join(valid_order_types)}"
            )
        
        # Validate side
        if request.side not in ["buy", "sell"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid side. Must be 'buy' or 'sell'"
            )
        
        # Validate quantity
        if request.quantity <= 0:
            raise HTTPException(
                status_code=400,
                detail="Quantity must be greater than 0"
            )
        
        # Normalize/validate time_in_force when provided (for limit orders)
        tif = None
        if request.time_in_force:
            tif_upper = request.time_in_force.upper()
            valid_tif = {"DAY", "GTC", "FOK", "IOC", "OPG", "CLS"}
            if tif_upper not in valid_tif:
                raise HTTPException(status_code=400, detail=f"Invalid time_in_force. Must be one of: {', '.join(sorted(valid_tif))}")
            tif = tif_upper

        # Submit order (auto-starts engine if needed)
        order_id = await trading_service.submit_hft_order(
            order_type=request.order_type,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=request.price,
            time_in_force=tif,
            duration_minutes=request.duration_minutes,
            interval_seconds=request.interval_seconds,
            volume_weight=request.volume_weight
        )
        
        # Check if order was actually accepted by Alpaca
        if not order_id or order_id == "Unknown" or order_id == "":
            raise HTTPException(
                status_code=400,
                detail="Order rejected: Please check for conflicting orders or insufficient account balance. Try cancelling existing orders first."
            )
        
        return {
            "status": "success",
            "data": {
                "order_id": order_id,
                "order_type": request.order_type,
                "symbol": request.symbol,
                "side": request.side,
                "quantity": request.quantity,
                "time_in_force": tif,
                "submitted_at": datetime.now().isoformat()
            },
            "message": f"{request.order_type.upper()} order accepted by Alpaca",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/order-types")
async def get_order_types():
    """Get available HFT order types"""
    return {
        "status": "success",
        "data": {
            "order_types": [
                {
                    "type": "market",
                    "name": "Market Order",
                    "description": "Execute immediately at current market price",
                    "parameters": ["symbol", "side", "quantity"]
                },
                {
                    "type": "limit",
                    "name": "Limit Order",
                    "description": "Execute only at specified price or better",
                    "parameters": ["symbol", "side", "quantity", "price"]
                },
                {
                    "type": "twap",
                    "name": "TWAP Order",
                    "description": "Time-Weighted Average Price - Execute over a time period",
                    "parameters": ["symbol", "side", "quantity", "duration_minutes", "interval_seconds"]
                },
                {
                    "type": "vwap",
                    "name": "VWAP Order",
                    "description": "Volume-Weighted Average Price - Execute based on volume",
                    "parameters": ["symbol", "side", "quantity", "volume_weight"]
                }
            ]
        },
        "timestamp": datetime.now().isoformat()
    }

@router.get("/symbols")
async def get_trading_symbols():
    """Get configured trading symbols"""
    try:
        if trading_service.hft_manager and hasattr(trading_service.hft_manager, 'config'):
            symbols = trading_service.hft_manager.config.trading_symbols
        else:
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]  # Default symbols
        
        return {
            "status": "success",
            "data": {
                "symbols": symbols,
                "count": len(symbols)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def hft_health_check():
    """Health check for HFT engine"""
    try:
        status = trading_service.get_hft_status()
        
        health = {
            "healthy": status.get("available", False) and status.get("running", False),
            "available": status.get("available", False),
            "running": status.get("running", False),
            "uptime_seconds": status.get("uptime_seconds", 0),
            "total_trades": status.get("total_trades", 0),
            "total_pnl": status.get("total_pnl", 0.0)
        }
        
        return {
            "status": "success",
            "data": health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trades")
async def get_hft_trades():
    """Get HFT trade history"""
    try:
        # For now, return mock data
        # In full implementation, this would query trade history from the HFT engine
        
        if not trading_service.hft_manager:
            return {
                "status": "success",
                "data": {
                    "trades": [],
                    "count": 0
                },
                "timestamp": datetime.now().isoformat()
            }
        
        trades = []
        
        return {
            "status": "success",
            "data": {
                "trades": trades,
                "count": len(trades),
                "total_pnl": trading_service.hft_manager.total_pnl if trading_service.hft_manager else 0.0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config")
async def get_hft_config():
    """Get HFT engine configuration"""
    try:
        if not trading_service.hft_manager or not hasattr(trading_service.hft_manager, 'config'):
            raise HTTPException(status_code=404, detail="HFT engine not configured")
        
        config = trading_service.hft_manager.config
        
        return {
            "status": "success",
            "data": {
                "edge_threshold": config.edge_threshold,
                "max_position_size": config.max_position_size,
                "max_daily_loss": config.max_daily_loss,
                "max_leverage": config.max_leverage,
                "trading_symbols": config.trading_symbols,
                "paper_trading": config.paper_trading
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

