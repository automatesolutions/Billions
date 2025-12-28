"""
Behavioral Trading API endpoints
API endpoints for managing trade annotations, exit decisions, and behavioral insights
"""

from fastapi import APIRouter, HTTPException, Query, Body
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime

from ..services.behavioral_service import behavioral_service
from ..models.behavioral_models import (
    TradeRationale, PositionAnnotation, ExitDecision, 
    AdditionDecision, BehavioralInsight, TradePerformance,
    TradeActionType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/behavioral", tags=["Behavioral Trading"])


@router.post("/rationale")
async def add_trade_rationale(rationale: TradeRationale):
    """
    Add or update trade rationale for decision-making context
    
    - **trade_id**: ID of the trade/position
    - **action_type**: Type of action (entry, addition, partial_exit, full_exit, etc.)
    - **rationale**: Detailed explanation of the decision
    - **confidence_level**: Confidence level (1-10)
    - **market_conditions**: Current market conditions
    - **technical_indicators**: Technical indicators used
    - **fundamental_factors**: Fundamental factors considered
    """
    try:
        result = await behavioral_service.add_trade_rationale(rationale)
        return {
            "status": "success",
            "rationale_id": result.id,
            "message": f"Trade rationale added for {rationale.action_type}"
        }
    except Exception as e:
        logger.error(f"Error adding trade rationale: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rationale/{trade_id}")
async def get_trade_rationales(trade_id: str):
    """Get all rationales for a specific trade"""
    try:
        rationales = [
            TradeRationale(**r) for r in behavioral_service.data["trade_rationales"]
            if r["trade_id"] == trade_id
        ]
        return {
            "trade_id": trade_id,
            "rationales": rationales,
            "count": len(rationales)
        }
    except Exception as e:
        logger.error(f"Error getting trade rationales: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/position-annotation")
async def add_position_annotation(annotation: PositionAnnotation):
    """
    Add or update position annotation with behavioral context
    
    - **symbol**: Stock symbol
    - **position_id**: Position identifier
    - **annotations**: List of trade rationales
    - **current_allocation**: Current position allocation
    - **target_allocation**: Target allocation
    - **risk_level**: Risk level assessment
    - **notes**: Additional notes
    - **tags**: Tags for categorization
    """
    try:
        result = await behavioral_service.add_position_annotation(annotation)
        return {
            "status": "success",
            "annotation_id": result.id,
            "message": f"Position annotation added for {annotation.symbol}"
        }
    except Exception as e:
        logger.error(f"Error adding position annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/position-annotations")
async def get_position_annotations(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """Get position annotations, optionally filtered by symbol"""
    try:
        annotations = await behavioral_service.get_position_annotations(symbol)
        return {
            "annotations": annotations,
            "count": len(annotations),
            "symbol_filter": symbol
        }
    except Exception as e:
        logger.error(f"Error getting position annotations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/exit-decision")
async def execute_exit_decision(exit_decision: ExitDecision):
    """
    Execute exit decision with detailed reasoning
    
    - **position_id**: Position identifier
    - **symbol**: Stock symbol
    - **exit_type**: Type of exit (partial, full, stop_loss, take_profit)
    - **exit_percentage**: Percentage of position to exit (0.01 to 1.0)
    - **exit_quantity**: Number of shares to exit
    - **exit_price**: Price at which to exit
    - **exit_reason**: Detailed reason for exit
    - **market_context**: Market context at time of exit
    - **technical_reason**: Technical analysis reason
    - **fundamental_reason**: Fundamental analysis reason
    - **emotional_factors**: Emotional factors influencing decision
    - **lessons_learned**: Lessons learned from this trade
    """
    try:
        result = await behavioral_service.execute_exit_decision(exit_decision)
        return result
    except Exception as e:
        logger.error(f"Error executing exit decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/addition-decision")
async def execute_addition_decision(addition_decision: AdditionDecision):
    """
    Execute addition to position with detailed reasoning
    
    - **position_id**: Position identifier
    - **symbol**: Stock symbol
    - **addition_quantity**: Number of shares to add
    - **addition_price**: Price at which to add
    - **addition_reason**: Detailed reason for addition
    - **market_opportunity**: Market opportunity identified
    - **technical_setup**: Technical setup for addition
    - **fundamental_catalyst**: Fundamental catalyst
    - **risk_reward_ratio**: Risk/reward ratio assessment
    - **position_sizing_logic**: Position sizing logic
    """
    try:
        result = await behavioral_service.execute_addition_decision(addition_decision)
        return result
    except Exception as e:
        logger.error(f"Error executing addition decision: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights")
async def get_behavioral_insights(
    limit: int = Query(10, ge=1, le=50, description="Number of insights to return")
):
    """Get recent behavioral insights and recommendations"""
    try:
        insights = await behavioral_service.get_behavioral_insights(limit)
        return {
            "insights": insights,
            "count": len(insights),
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting behavioral insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-analysis")
async def get_trade_performance_analysis(
    symbol: Optional[str] = Query(None, description="Filter by symbol")
):
    """Get trade performance analysis for behavioral insights"""
    try:
        analysis = await behavioral_service.get_trade_performance_analysis(symbol)
        return analysis
    except Exception as e:
        logger.error(f"Error getting trade performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/holdings/{symbol}/context")
async def get_holding_context(symbol: str):
    """Get complete behavioral context for a specific holding"""
    try:
        # Get position annotations
        annotations = await behavioral_service.get_position_annotations(symbol)
        
        # Get trade rationales for this symbol
        rationales = [
            TradeRationale(**r) for r in behavioral_service.data["trade_rationales"]
            if any(ann["symbol"] == symbol for ann in behavioral_service.data["position_annotations"]
                   if ann["position_id"] == r["trade_id"])
        ]
        
        # Get exit decisions
        exit_decisions = [
            ExitDecision(**e) for e in behavioral_service.data["exit_decisions"]
            if e["symbol"] == symbol.upper()
        ]
        
        # Get addition decisions
        addition_decisions = [
            AdditionDecision(**a) for a in behavioral_service.data["addition_decisions"]
            if a["symbol"] == symbol.upper()
        ]
        
        # Get performance data
        performance = [
            TradePerformance(**p) for p in behavioral_service.data["trade_performance"]
            if p["symbol"] == symbol.upper()
        ]
        
        return {
            "symbol": symbol.upper(),
            "annotations": annotations,
            "rationales": rationales,
            "exit_decisions": exit_decisions,
            "addition_decisions": addition_decisions,
            "performance": performance,
            "summary": {
                "total_rationales": len(rationales),
                "total_exits": len(exit_decisions),
                "total_additions": len(addition_decisions),
                "performance_records": len(performance)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting holding context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rationale/{rationale_id}")
async def update_trade_rationale(
    rationale_id: str,
    rationale_update: Dict[str, Any] = Body(...)
):
    """Update an existing trade rationale"""
    try:
        # Find the rationale
        rationale_index = next(
            (i for i, r in enumerate(behavioral_service.data["trade_rationales"])
             if r["id"] == rationale_id),
            None
        )
        
        if rationale_index is None:
            raise HTTPException(status_code=404, detail="Rationale not found")
        
        # Update the rationale
        rationale_data = behavioral_service.data["trade_rationales"][rationale_index]
        rationale_data.update(rationale_update)
        rationale_data["updated_at"] = datetime.now().isoformat()
        
        behavioral_service._save_data()
        
        return {
            "status": "success",
            "rationale_id": rationale_id,
            "message": "Rationale updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating trade rationale: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rationale/{rationale_id}")
async def delete_trade_rationale(rationale_id: str):
    """Delete a trade rationale"""
    try:
        # Find and remove the rationale
        rationale_index = next(
            (i for i, r in enumerate(behavioral_service.data["trade_rationales"])
             if r["id"] == rationale_id),
            None
        )
        
        if rationale_index is None:
            raise HTTPException(status_code=404, detail="Rationale not found")
        
        behavioral_service.data["trade_rationales"].pop(rationale_index)
        behavioral_service._save_data()
        
        return {
            "status": "success",
            "rationale_id": rationale_id,
            "message": "Rationale deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting trade rationale: {e}")
        raise HTTPException(status_code=500, detail=str(e))
