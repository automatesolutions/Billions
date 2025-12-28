"""
Outlier Detection endpoints  
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import logging

from api.services.outlier_detection import outlier_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/outliers", tags=["Outlier Detection"])


@router.get("/strategies")
async def get_strategies():
    """Get all available outlier detection strategies"""
    return {
        "strategies": outlier_service.get_all_strategies()
    }


@router.get("/{strategy}/info")
async def get_strategy_info(strategy: str):
    """Get information about a specific strategy"""
    info = outlier_service.get_strategy_info(strategy)
    
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy}' not found. Valid strategies: scalp, swing, longterm"
        )
    
    return info


@router.post("/{strategy}/refresh")
async def refresh_outliers(
    strategy: str,
    background_tasks: BackgroundTasks
):
    """
    Refresh outlier detection for a strategy
    
    This operation runs in the background as it can take several minutes.
    """
    if strategy not in ["scalp", "swing", "longterm"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy}. Must be one of: scalp, swing, longterm"
        )
    
    # Run in background
    background_tasks.add_task(outlier_service.refresh_outliers, strategy, None)
    
    return {
        "message": f"Outlier refresh started for {strategy}",
        "status": "processing",
        "note": "This may take several minutes. Check the outliers endpoint for results."
    }

