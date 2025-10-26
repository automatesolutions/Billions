"""
Market data endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from api.database import get_db
from db.models import PerfMetric
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/market", tags=["Market Data"])

# Import refresh functions
try:
    from funda.refresh_outliers import start_refresh_thread, get_refresh_status
    REFRESH_AVAILABLE = True
except ImportError:
    logger.warning("Refresh functions not available")
    REFRESH_AVAILABLE = False


@router.get("/outliers/{strategy}")
async def get_outliers(
    strategy: str,
    db: Session = Depends(get_db)
):
    """
    Get outliers for a specific strategy
    
    Strategies: scalp, swing, longterm
    """
    valid_strategies = ["scalp", "swing", "longterm"]
    
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
        )
    
    try:
        # Query outliers from database
        outliers = db.query(PerfMetric).filter(
            PerfMetric.strategy == strategy,
            PerfMetric.is_outlier == True
        ).all()
        
        # Convert to dict
        result = []
        for outlier in outliers:
            result.append({
                "symbol": outlier.symbol,
                "metric_x": float(outlier.metric_x) if outlier.metric_x else None,
                "metric_y": float(outlier.metric_y) if outlier.metric_y else None,
                "z_x": float(outlier.z_x) if outlier.z_x else None,
                "z_y": float(outlier.z_y) if outlier.z_y else None,
                "is_outlier": outlier.is_outlier,
                "inserted": outlier.inserted.isoformat() if outlier.inserted else None
            })
        
        return {
            "strategy": strategy,
            "count": len(result),
            "outliers": result
        }
        
    except Exception as e:
        logger.error(f"Error fetching outliers for {strategy}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/{strategy}")
async def get_performance_metrics(
    strategy: str,
    db: Session = Depends(get_db)
):
    """
    Get all performance metrics for a strategy
    """
    valid_strategies = ["scalp", "swing", "longterm"]
    
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
        )
    
    try:
        metrics = db.query(PerfMetric).filter(
            PerfMetric.strategy == strategy
        ).all()
        
        result = []
        for metric in metrics:
            result.append({
                "symbol": metric.symbol,
                "metric_x": float(metric.metric_x) if metric.metric_x else None,
                "metric_y": float(metric.metric_y) if metric.metric_y else None,
                "z_x": float(metric.z_x) if metric.z_x else None,
                "z_y": float(metric.z_y) if metric.z_y else None,
                "is_outlier": metric.is_outlier,
            })
        
        return {
            "strategy": strategy,
            "count": len(result),
            "metrics": result
        }
        
    except Exception as e:
        logger.error(f"Error fetching performance metrics for {strategy}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def trigger_refresh(background_tasks: BackgroundTasks):
    """
    Trigger a refresh of all market data and outlier detection
    
    This will:
    - Fetch latest NASDAQ tickers
    - Download market data
    - Calculate performance metrics
    - Detect outliers for all strategies
    
    Returns immediately with status. Use GET /refresh/status to check progress.
    """
    if not REFRESH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Refresh functionality not available"
        )
    
    # Check if already running
    status = get_refresh_status()
    if status['is_running']:
        return {
            "message": "Refresh already in progress",
            "status": status
        }
    
    # Start refresh in background
    success = start_refresh_thread()
    
    if success:
        return {
            "message": "Data refresh started",
            "status": get_refresh_status()
        }
    else:
        raise HTTPException(
            status_code=500,
            detail="Failed to start refresh"
        )


@router.get("/refresh/status")
async def get_refresh_status_endpoint():
    """
    Get current refresh status
    
    Returns:
    - is_running: bool
    - progress: int (0-100)
    - current_strategy: str
    - message: str
    - start_time: timestamp
    - estimated_completion: timestamp
    """
    if not REFRESH_AVAILABLE:
        return {
            "is_running": False,
            "progress": 0,
            "message": "Refresh functionality not available"
        }
    
    return get_refresh_status()

