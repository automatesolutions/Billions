"""
Behavioral Trading Service
Service for managing trade annotations, exit decisions, and behavioral insights
"""

import os
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import uuid

from ..models.behavioral_models import (
    TradeRationale, PositionAnnotation, ExitDecision, 
    AdditionDecision, BehavioralInsight, TradePerformance,
    TradeActionType
)

logger = logging.getLogger(__name__)

class BehavioralTradingService:
    """Service for managing behavioral trading data and insights"""
    
    def __init__(self):
        self.data_file = "behavioral_trading_data.json"
        self.insights_file = "behavioral_insights.json"
        self._load_data()
    
    def _load_data(self):
        """Load behavioral data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.data = json.load(f)
            else:
                self.data = {
                    "trade_rationales": [],
                    "position_annotations": [],
                    "exit_decisions": [],
                    "addition_decisions": [],
                    "trade_performance": []
                }
            
            if os.path.exists(self.insights_file):
                with open(self.insights_file, 'r') as f:
                    self.insights = json.load(f)
            else:
                self.insights = {
                    "behavioral_insights": []
                }
                
        except Exception as e:
            logger.error(f"Error loading behavioral data: {e}")
            self.data = {
                "trade_rationales": [],
                "position_annotations": [],
                "exit_decisions": [],
                "addition_decisions": [],
                "trade_performance": []
            }
            self.insights = {"behavioral_insights": []}
    
    def _save_data(self):
        """Save behavioral data to file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
            with open(self.insights_file, 'w') as f:
                json.dump(self.insights, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving behavioral data: {e}")
    
    async def add_trade_rationale(self, rationale: TradeRationale) -> TradeRationale:
        """Add or update trade rationale"""
        try:
            rationale.id = str(uuid.uuid4())
            rationale.created_at = datetime.now()
            
            # Check if rationale already exists for this trade/action
            existing = next(
                (r for r in self.data["trade_rationales"] 
                 if r["trade_id"] == rationale.trade_id and r["action_type"] == rationale.action_type),
                None
            )
            
            if existing:
                # Update existing rationale
                existing.update(rationale.dict())
                existing["updated_at"] = datetime.now()
            else:
                # Add new rationale
                self.data["trade_rationales"].append(rationale.dict())
            
            self._save_data()
            logger.info(f"Added trade rationale for {rationale.trade_id} - {rationale.action_type}")
            return rationale
            
        except Exception as e:
            logger.error(f"Error adding trade rationale: {e}")
            raise
    
    async def get_position_annotations(self, symbol: Optional[str] = None) -> List[PositionAnnotation]:
        """Get position annotations, optionally filtered by symbol"""
        try:
            annotations = []
            for annotation_data in self.data["position_annotations"]:
                if symbol is None or annotation_data["symbol"] == symbol.upper():
                    # Convert trade rationales back to objects
                    rationales = [
                        TradeRationale(**r) for r in annotation_data.get("annotations", [])
                    ]
                    annotation_data["annotations"] = rationales
                    annotations.append(PositionAnnotation(**annotation_data))
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error getting position annotations: {e}")
            return []
    
    async def add_position_annotation(self, annotation: PositionAnnotation) -> PositionAnnotation:
        """Add or update position annotation"""
        try:
            annotation.id = str(uuid.uuid4())
            annotation.created_at = datetime.now()
            
            # Check if annotation already exists
            existing_index = next(
                (i for i, a in enumerate(self.data["position_annotations"])
                 if a["symbol"] == annotation.symbol and a["position_id"] == annotation.position_id),
                None
            )
            
            if existing_index is not None:
                # Update existing annotation
                self.data["position_annotations"][existing_index] = annotation.dict()
            else:
                # Add new annotation
                self.data["position_annotations"].append(annotation.dict())
            
            self._save_data()
            logger.info(f"Added position annotation for {annotation.symbol}")
            return annotation
            
        except Exception as e:
            logger.error(f"Error adding position annotation: {e}")
            raise
    
    async def execute_exit_decision(self, exit_decision: ExitDecision) -> Dict[str, Any]:
        """Execute exit decision and log reasoning"""
        try:
            exit_decision.id = str(uuid.uuid4())
            exit_decision.created_at = datetime.now()
            
            # Add to exit decisions
            self.data["exit_decisions"].append(exit_decision.dict())
            
            # Update trade performance if this is a full exit
            if exit_decision.exit_type == "full":
                await self._update_trade_performance(exit_decision)
            
            self._save_data()
            
            # Generate behavioral insight
            await self._generate_exit_insight(exit_decision)
            
            logger.info(f"Executed exit decision for {exit_decision.symbol}: {exit_decision.exit_type}")
            
            return {
                "status": "success",
                "exit_decision_id": exit_decision.id,
                "message": f"Exit decision logged for {exit_decision.symbol}"
            }
            
        except Exception as e:
            logger.error(f"Error executing exit decision: {e}")
            raise
    
    async def execute_addition_decision(self, addition_decision: AdditionDecision) -> Dict[str, Any]:
        """Execute addition to position and log reasoning"""
        try:
            addition_decision.id = str(uuid.uuid4())
            addition_decision.created_at = datetime.now()
            
            # Add to addition decisions
            self.data["addition_decisions"].append(addition_decision.dict())
            
            self._save_data()
            
            # Generate behavioral insight
            await self._generate_addition_insight(addition_decision)
            
            logger.info(f"Executed addition decision for {addition_decision.symbol}")
            
            return {
                "status": "success",
                "addition_decision_id": addition_decision.id,
                "message": f"Addition decision logged for {addition_decision.symbol}"
            }
            
        except Exception as e:
            logger.error(f"Error executing addition decision: {e}")
            raise
    
    async def get_behavioral_insights(self, limit: int = 10) -> List[BehavioralInsight]:
        """Get recent behavioral insights"""
        try:
            insights = []
            for insight_data in self.insights["behavioral_insights"][-limit:]:
                insights.append(BehavioralInsight(**insight_data))
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting behavioral insights: {e}")
            return []
    
    async def get_trade_performance_analysis(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get trade performance analysis for behavioral insights"""
        try:
            performances = []
            for perf_data in self.data["trade_performance"]:
                if symbol is None or perf_data["symbol"] == symbol.upper():
                    performances.append(TradePerformance(**perf_data))
            
            if not performances:
                return {"message": "No trade performance data available"}
            
            # Calculate metrics
            total_trades = len(performances)
            completed_trades = [p for p in performances if p.exit_date is not None]
            avg_return = sum(p.return_percentage or 0 for p in completed_trades) / len(completed_trades) if completed_trades else 0
            win_rate = len([p for p in completed_trades if (p.return_percentage or 0) > 0]) / len(completed_trades) if completed_trades else 0
            avg_hold_duration = sum(p.hold_duration_days or 0 for p in completed_trades) / len(completed_trades) if completed_trades else 0
            
            return {
                "total_trades": total_trades,
                "completed_trades": len(completed_trades),
                "average_return_percentage": round(avg_return, 2),
                "win_rate": round(win_rate * 100, 2),
                "average_hold_duration_days": round(avg_hold_duration, 1),
                "performances": performances[-10:]  # Last 10 trades
            }
            
        except Exception as e:
            logger.error(f"Error getting trade performance analysis: {e}")
            return {"error": str(e)}
    
    async def _update_trade_performance(self, exit_decision: ExitDecision):
        """Update trade performance when position is fully exited"""
        try:
            # Find the corresponding trade performance record
            for perf_data in self.data["trade_performance"]:
                if (perf_data["symbol"] == exit_decision.symbol and 
                    perf_data["exit_date"] is None):
                    
                    perf_data["exit_date"] = exit_decision.created_at
                    perf_data["exit_price"] = exit_decision.exit_price
                    perf_data["total_return"] = (exit_decision.exit_price - perf_data["entry_price"]) * perf_data["quantity"]
                    perf_data["return_percentage"] = ((exit_decision.exit_price - perf_data["entry_price"]) / perf_data["entry_price"]) * 100
                    perf_data["hold_duration_days"] = (exit_decision.created_at - datetime.fromisoformat(perf_data["entry_date"])).days
                    
                    break
                    
        except Exception as e:
            logger.error(f"Error updating trade performance: {e}")
    
    async def _generate_exit_insight(self, exit_decision: ExitDecision):
        """Generate behavioral insight from exit decision"""
        try:
            # Simple insight generation - in production, this would use AI
            insight = BehavioralInsight(
                id=str(uuid.uuid4()),
                user_id="default_user",  # In production, get from auth
                insight_type="pattern",
                title=f"Exit Pattern Analysis: {exit_decision.symbol}",
                description=f"Exit decision for {exit_decision.symbol} shows {exit_decision.exit_type} exit with reasoning: {exit_decision.exit_reason[:100]}...",
                confidence_score=0.7,
                supporting_data={
                    "symbol": exit_decision.symbol,
                    "exit_type": exit_decision.exit_type,
                    "exit_percentage": exit_decision.exit_percentage
                },
                actionable_recommendations=[
                    "Review exit timing patterns",
                    "Analyze emotional factors in exit decisions",
                    "Consider setting systematic exit rules"
                ]
            )
            
            self.insights["behavioral_insights"].append(insight.dict())
            
        except Exception as e:
            logger.error(f"Error generating exit insight: {e}")
    
    async def _generate_addition_insight(self, addition_decision: AdditionDecision):
        """Generate behavioral insight from addition decision"""
        try:
            insight = BehavioralInsight(
                id=str(uuid.uuid4()),
                user_id="default_user",
                insight_type="recommendation",
                title=f"Position Addition Analysis: {addition_decision.symbol}",
                description=f"Addition to {addition_decision.symbol} position based on: {addition_decision.addition_reason[:100]}...",
                confidence_score=0.6,
                supporting_data={
                    "symbol": addition_decision.symbol,
                    "addition_quantity": addition_decision.addition_quantity,
                    "addition_price": addition_decision.addition_price
                },
                actionable_recommendations=[
                    "Track addition success rates",
                    "Analyze market opportunity timing",
                    "Review position sizing logic"
                ]
            )
            
            self.insights["behavioral_insights"].append(insight.dict())
            
        except Exception as e:
            logger.error(f"Error generating addition insight: {e}")

# Global instance
behavioral_service = BehavioralTradingService()
