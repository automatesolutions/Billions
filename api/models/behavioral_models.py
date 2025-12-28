"""
Behavioral Trading Data Models
Enhanced models for capturing decision-making context and trade annotations
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TradeActionType(str, Enum):
    """Types of trade actions"""
    ENTRY = "entry"
    ADDITION = "addition" 
    PARTIAL_EXIT = "partial_exit"
    FULL_EXIT = "full_exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class TradeRationale(BaseModel):
    """Trade rationale and decision-making context"""
    id: Optional[str] = None
    trade_id: str
    action_type: TradeActionType
    rationale: str = Field(..., min_length=10, max_length=1000)
    market_conditions: Optional[str] = None  # "bull", "bear", "sideways", "volatile"
    technical_indicators: Optional[List[str]] = None  # ["RSI", "MACD", "Support/Resistance"]
    fundamental_factors: Optional[List[str]] = None  # ["earnings", "news", "guidance"]
    risk_assessment: Optional[str] = None  # "low", "medium", "high"
    confidence_level: int = Field(..., ge=1, le=10)  # 1-10 scale
    expected_hold_time: Optional[str] = None  # "day", "week", "month", "quarter", "year"
    target_price: Optional[float] = None
    stop_loss_price: Optional[float] = None
    position_size_reasoning: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class PositionAnnotation(BaseModel):
    """Annotations for position management"""
    id: Optional[str] = None
    symbol: str
    position_id: str
    annotations: List[TradeRationale] = []
    current_allocation: float = 0.0
    target_allocation: Optional[float] = None
    risk_level: Optional[str] = None
    notes: Optional[str] = None
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

class ExitDecision(BaseModel):
    """Exit decision with reasoning"""
    id: Optional[str] = None
    position_id: str
    symbol: str
    exit_type: str  # "partial", "full", "stop_loss", "take_profit"
    exit_percentage: float = Field(..., ge=0.01, le=1.0)  # 0.01 to 1.0 (1% to 100%)
    exit_quantity: int
    exit_price: float
    exit_reason: str = Field(..., min_length=10, max_length=1000)
    market_context: Optional[str] = None
    technical_reason: Optional[str] = None
    fundamental_reason: Optional[str] = None
    emotional_factors: Optional[str] = None  # "fear", "greed", "patience", "impatience"
    lessons_learned: Optional[str] = None
    would_reenter: Optional[bool] = None
    reentry_conditions: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class AdditionDecision(BaseModel):
    """Decision to add to existing position"""
    id: Optional[str] = None
    position_id: str
    symbol: str
    addition_quantity: int
    addition_price: float
    addition_reason: str = Field(..., min_length=10, max_length=1000)
    market_opportunity: Optional[str] = None
    technical_setup: Optional[str] = None
    fundamental_catalyst: Optional[str] = None
    risk_reward_ratio: Optional[float] = None
    position_sizing_logic: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

class BehavioralInsight(BaseModel):
    """AI-generated insights from behavioral patterns"""
    id: Optional[str] = None
    user_id: str
    insight_type: str  # "pattern", "recommendation", "warning", "success"
    title: str
    description: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    supporting_data: Dict[str, Any] = {}
    actionable_recommendations: List[str] = []
    created_at: datetime = Field(default_factory=datetime.now)

class TradePerformance(BaseModel):
    """Performance metrics for behavioral analysis"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    quantity: int
    total_return: Optional[float] = None
    return_percentage: Optional[float] = None
    hold_duration_days: Optional[int] = None
    max_drawdown: Optional[float] = None
    max_gain: Optional[float] = None
    rationale_quality_score: Optional[float] = None  # AI assessment of rationale quality
    decision_consistency_score: Optional[float] = None  # How consistent with stated strategy
    created_at: datetime = Field(default_factory=datetime.now)
