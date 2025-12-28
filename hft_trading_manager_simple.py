"""
Simplified HFT Trading Manager (Python-only version)
This version works without the C++ engine for immediate testing
"""

import os
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class HFTConfig:
    """Configuration for HFT Trading Engine"""
    polygon_api_key: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    paper_trading: bool = True
    edge_threshold: float = 0.001  # 0.1%
    max_position_size: int = 1000
    max_daily_loss: float = 5000.0
    max_leverage: float = 2.0
    trading_symbols: List[str] = None
    
    def __post_init__(self):
        if self.trading_symbols is None:
            self.trading_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

class SimplifiedHFTTradingManager:
    """Simplified HFT Trading Manager (Python-only)"""
    
    def __init__(self, config: HFTConfig):
        self.config = config
        self.is_running = False
        self.is_initialized = False
        self.trade_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.performance_callbacks: List[Callable] = []
        
        # Performance tracking
        self.total_trades = 0
        self.total_pnl = 0.0
        self.start_time = None
        
        # Alpaca session
        self.alpaca_session = None
        
    def initialize(self) -> bool:
        """Initialize the simplified HFT manager"""
        try:
            logger.info("Initializing Simplified HFT Trading Manager...")
            
            # Create Alpaca session
            headers = {
                "APCA-API-KEY-ID": self.config.alpaca_api_key,
                "APCA-API-SECRET-KEY": self.config.alpaca_secret_key,
                "Content-Type": "application/json"
            }
            
            self.alpaca_session = aiohttp.ClientSession(headers=headers)
            self.is_initialized = True
            
            logger.info("Simplified HFT Trading Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HFT manager: {e}")
            return False
    
    def start(self) -> bool:
        """Start the HFT manager"""
        if not self.is_initialized:
            logger.error("HFT manager not initialized")
            return False
        
        try:
            self.is_running = True
            self.start_time = datetime.now()
            logger.info("Simplified HFT Trading Manager started")
            return True
        except Exception as e:
            logger.error(f"Failed to start HFT manager: {e}")
            return False
    
    def stop(self):
        """Stop the HFT manager"""
        if self.is_running:
            self.is_running = False
            logger.info("Simplified HFT Trading Manager stopped")
        
        if self.alpaca_session:
            asyncio.create_task(self.alpaca_session.close())
    
    async def submit_market_order(self, symbol: str, side: str, quantity: int) -> Optional[str]:
        """Submit a market order"""
        if not self.is_running:
            logger.error("HFT manager not running")
            return None
        
        try:
            order_data = {
                "symbol": symbol,
                "qty": str(quantity),
                "side": side,
                "type": "market",
                "time_in_force": "day"
            }
            
            url = f"{self.config.alpaca_base_url}/v2/orders"
            async with self.alpaca_session.post(url, json=order_data) as response:
                if response.status == 200:
                    data = await response.json()
                    order_id = data.get("id", "")
                    
                    # Simulate trade execution callback
                    self._simulate_trade_execution(symbol, side, quantity, 150.0)  # Mock price
                    
                    logger.info(f"Market order submitted: {side} {quantity} {symbol} -> {order_id}")
                    return order_id
                else:
                    error_data = await response.json()
                    logger.error(f"Market order failed: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error submitting market order: {e}")
            return None
    
    async def submit_limit_order(self, symbol: str, side: str, quantity: int, price: float) -> Optional[str]:
        """Submit a limit order"""
        if not self.is_running:
            logger.error("HFT manager not running")
            return None
        
        try:
            order_data = {
                "symbol": symbol,
                "qty": str(quantity),
                "side": side,
                "type": "limit",
                "limit_price": str(price),
                "time_in_force": "day"
            }
            
            url = f"{self.config.alpaca_base_url}/v2/orders"
            async with self.alpaca_session.post(url, json=order_data) as response:
                if response.status == 200:
                    data = await response.json()
                    order_id = data.get("id", "")
                    
                    logger.info(f"Limit order submitted: {side} {quantity} {symbol} @ ${price} -> {order_id}")
                    return order_id
                else:
                    error_data = await response.json()
                    logger.error(f"Limit order failed: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error submitting limit order: {e}")
            return None
    
    async def submit_twap_order(self, symbol: str, side: str, quantity: int, 
                               duration_minutes: int, interval_seconds: int) -> Optional[str]:
        """Submit a TWAP order (simplified implementation)"""
        if not self.is_running:
            logger.error("HFT manager not running")
            return None
        
        try:
            # For now, submit as a regular market order
            # In a full implementation, this would split the order over time
            logger.info(f"TWAP order submitted as market order: {side} {quantity} {symbol} "
                       f"(would execute over {duration_minutes} minutes)")
            
            return await self.submit_market_order(symbol, side, quantity)
                    
        except Exception as e:
            logger.error(f"Error submitting TWAP order: {e}")
            return None
    
    async def submit_vwap_order(self, symbol: str, side: str, quantity: int, 
                               volume_weight: float) -> Optional[str]:
        """Submit a VWAP order (simplified implementation)"""
        if not self.is_running:
            logger.error("HFT manager not running")
            return None
        
        try:
            # For now, submit as a regular market order
            # In a full implementation, this would execute based on volume
            logger.info(f"VWAP order submitted as market order: {side} {quantity} {symbol} "
                       f"(volume weight: {volume_weight})")
            
            return await self.submit_market_order(symbol, side, quantity)
                    
        except Exception as e:
            logger.error(f"Error submitting VWAP order: {e}")
            return None
    
    def get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics"""
        if not self.is_running:
            return None
        
        try:
            # Mock performance metrics for demonstration
            return {
                "total_trades": self.total_trades,
                "successful_trades": self.total_trades,  # Assume all successful for demo
                "failed_trades": 0,
                "total_pnl": self.total_pnl,
                "win_rate": 0.75,  # Mock 75% win rate
                "avg_execution_time_ms": 50.0,  # Mock 50ms execution time
                "fill_rate": 0.95,  # Mock 95% fill rate
                "sharpe_ratio": 1.2,  # Mock Sharpe ratio
                "max_drawdown": 0.05  # Mock 5% max drawdown
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return None
    
    def get_execution_metrics(self) -> Optional[Dict[str, Any]]:
        """Get execution metrics"""
        if not self.is_running:
            return None
        
        try:
            # Mock execution metrics
            return {
                "total_orders": self.total_trades,
                "successful_orders": self.total_trades,
                "failed_orders": 0,
                "avg_execution_time_ms": 50.0,
                "total_slippage": 0.0,
                "avg_slippage": 0.0,
                "fill_rate": 0.95
            }
        except Exception as e:
            logger.error(f"Error getting execution metrics: {e}")
            return None
    
    def add_trade_callback(self, callback: Callable):
        """Add a trade execution callback"""
        self.trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add an error callback"""
        self.error_callbacks.append(callback)
    
    def add_performance_callback(self, callback: Callable):
        """Add a performance update callback"""
        self.performance_callbacks.append(callback)
    
    def _simulate_trade_execution(self, symbol: str, side: str, quantity: int, price: float):
        """Simulate trade execution for demo purposes"""
        self.total_trades += 1
        
        # Calculate mock P&L
        if side == "buy":
            self.total_pnl -= quantity * price  # Cost
        else:
            self.total_pnl += quantity * price  # Revenue
        
        logger.info(f"Trade executed: {side} {quantity} {symbol} @ ${price:.2f}")
        
        # Call registered callbacks
        for callback in self.trade_callbacks:
            try:
                callback(symbol, side, quantity, price)
            except Exception as e:
                logger.error(f"Error in trade callback: {e}")

# Create aliases for compatibility
HFTTradingManager = SimplifiedHFTTradingManager

# Mock classes for compatibility
class TradingMetrics:
    def __init__(self):
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.avg_execution_time_ms = 0.0
        self.fill_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0

class ExecutionMetrics:
    def __init__(self):
        self.total_orders = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.avg_execution_time_ms = 0.0
        self.total_slippage = 0.0
        self.avg_slippage = 0.0
        self.fill_rate = 0.0
