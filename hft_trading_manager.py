"""
HFT Trading Manager - Enhanced Version with WebSocket Support
This version supports both simplified Python-only and C++ WebSocket trading
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp

logger = logging.getLogger(__name__)

# Try to import the Alpaca WebSocket HFT manager
try:
    from alpaca_websocket_hft_manager import create_alpaca_hft_manager
    WEBSOCKET_HFT_AVAILABLE = True
    logger.info("Alpaca WebSocket HFT manager available")
except ImportError:
    WEBSOCKET_HFT_AVAILABLE = False
    logger.warning("Alpaca WebSocket HFT manager not available, using simplified version")

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

class HFTTradingManager:
    """Enhanced HFT Trading Manager with WebSocket Support"""
    
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
        
        # WebSocket support
        self.use_websocket = WEBSOCKET_HFT_AVAILABLE and getattr(config, 'use_websocket', True)
        self.websocket_manager = None
        
        if self.use_websocket:
            logger.info("HFT Trading Manager initialized (WebSocket-enabled version)")
        else:
            logger.info("HFT Trading Manager initialized (Simplified Python-only version)")
        
    def initialize(self) -> bool:
        """Initialize the HFT Trading Engine"""
        try:
            logger.info("Initializing HFT Trading Manager...")
            
            if self.use_websocket and WEBSOCKET_HFT_AVAILABLE:
                # Initialize WebSocket-based HFT manager
                try:
                    self.websocket_manager = create_alpaca_hft_manager(
                        api_key=self.config.alpaca_api_key,
                        secret_key=self.config.alpaca_secret_key,
                        base_url=self.config.alpaca_base_url,
                        paper_trading=self.config.paper_trading
                    )
                    self.is_initialized = True
                    logger.info("WebSocket HFT Trading Manager initialized successfully")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to initialize WebSocket manager, falling back to simplified: {e}")
                    self.use_websocket = False
            
            # Fallback to simplified Python-only version
            headers = {
                "APCA-API-KEY-ID": self.config.alpaca_api_key,
                "APCA-API-SECRET-KEY": self.config.alpaca_secret_key,
                "Content-Type": "application/json"
            }
            
            self.alpaca_session = aiohttp.ClientSession(headers=headers)
            self.is_initialized = True
            
            logger.info("HFT Trading Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HFT manager: {e}")
            return False
    
    def start(self) -> bool:
        """Start the HFT Trading Engine"""
        if not self.is_initialized:
            logger.error("HFT manager not initialized")
            return False
        
        try:
            if self.use_websocket and self.websocket_manager:
                # Start WebSocket manager
                asyncio.create_task(self.websocket_manager.start())
                logger.info("WebSocket HFT Trading Manager started")
            else:
                logger.info("Simplified HFT Trading Manager started")
            
            self.is_running = True
            self.start_time = datetime.now()
            logger.info("HFT Trading Manager started")
            return True
        except Exception as e:
            logger.error(f"Failed to start HFT manager: {e}")
            return False
    
    def stop(self):
        """Stop the HFT Trading Engine"""
        if self.is_running:
            self.is_running = False
            
            if self.use_websocket and self.websocket_manager:
                # Stop WebSocket manager
                asyncio.create_task(self.websocket_manager.stop())
                logger.info("WebSocket HFT Trading Manager stopped")
            
            logger.info("HFT Trading Manager stopped")
        
        if self.alpaca_session:
            # Close session properly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.alpaca_session.close())
                else:
                    loop.run_until_complete(self.alpaca_session.close())
            except Exception as e:
                logger.error(f"Error closing Alpaca session: {e}")
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders for a symbol or all symbols"""
        try:
            url = f"{self.config.alpaca_base_url}/v2/orders"
            params = {"status": "open"}
            if symbol:
                params["symbols"] = symbol
            
            async with self.alpaca_session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get open orders: {await response.text()}")
                    return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID"""
        try:
            url = f"{self.config.alpaca_base_url}/v2/orders/{order_id}"
            async with self.alpaca_session.delete(url) as response:
                if response.status == 200:
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True
                elif response.status == 422:
                    # Order might already be filled or cancelled
                    logger.info(f"Order {order_id} cannot be cancelled (likely already filled/cancelled)")
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to cancel order {order_id}: {response.status} - {error_text}")
                    return False
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def check_order_conflicts(self, symbol: str, side: str, order_type: str) -> bool:
        """Check for potential order conflicts and resolve them"""
        try:
            open_orders = await self.get_open_orders(symbol)
            
            # Check for conflicting orders
            for order in open_orders:
                if order.get("symbol") == symbol:
                    existing_side = order.get("side")
                    existing_type = order.get("order_type")
                    
                    # Check for wash trade potential or short selling restrictions
                    if existing_side != side:
                        logger.warning(f"Potential conflict detected: {existing_side} {existing_type} order exists for {symbol}")
                        
                        # Always cancel conflicting orders to prevent brokerage restrictions
                        logger.info(f"Cancelling conflicting {existing_side} order to allow {side} {order_type} order for {symbol}")
                        cancel_success = await self.cancel_order(order["id"])
                        if cancel_success:
                            logger.info(f"✅ Cancelled conflicting order {order['id']} to allow {side} order")
                        else:
                            logger.warning(f"⚠️ Could not cancel conflicting order {order['id']}")
                            # If we can't cancel, don't allow the new order
                            return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking order conflicts: {e}")
            return True  # Allow order if check fails
    
    async def submit_market_order(self, symbol: str, side: str, quantity: int) -> Optional[str]:
        """Submit a market order with conflict prevention"""
        # Auto-start if not running
        if not self.is_running:
            logger.info("Auto-starting HFT manager for market order submission")
            if not self.start():
                logger.error("Failed to auto-start HFT manager")
                return None
        
        try:
            # Check for order conflicts
            if not await self.check_order_conflicts(symbol, side, "market"):
                logger.error(f"Cannot submit {side} market order for {symbol}: conflicting orders exist")
                return None
            
            # Use WebSocket manager if available
            if self.use_websocket and self.websocket_manager:
                logger.info(f"Submitting market order via WebSocket: {side} {quantity} {symbol}")
                return await self.websocket_manager.submit_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity
                )
            
            # Fallback to REST API
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
                    
                    # Immediate confirmation
                    logger.info(f"🚀 ORDER SUBMITTED: {side.upper()} {quantity} {symbol}")
                    logger.info(f"   📋 Type: MARKET")
                    logger.info(f"   🆔 Order ID: {order_id}")
                    logger.info(f"   📊 Symbol: {symbol}")
                    logger.info(f"   📦 Quantity: {quantity}")
                    logger.info(f"   ⏰ Time-in-Force: {order_data.get('time_in_force', 'day')}")
                    logger.info(f"   ✅ Status: SUBMITTED TO ALPACA")
                    
                    # Start monitoring order status
                    asyncio.create_task(self._monitor_order_status(order_id, symbol, side, "market"))
                    
                    return order_id
                else:
                    error_data = await response.json()
                    logger.error(f"❌ ORDER REJECTED: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error submitting market order: {e}")
            return None
    
    async def submit_limit_order(self, symbol: str, side: str, quantity: int, price: float, time_in_force: Optional[str] = None) -> Optional[str]:
        """Submit a limit order with optional time-in-force and conflict prevention"""
        # Auto-start if not running
        if not self.is_running:
            logger.info("Auto-starting HFT manager for limit order submission")
            if not self.start():
                logger.error("Failed to auto-start HFT manager")
                return None
        
        try:
            # Check for order conflicts
            if not await self.check_order_conflicts(symbol, side, "limit"):
                logger.error(f"Cannot submit {side} limit order for {symbol}: conflicting orders exist")
                return None
            
            # Use WebSocket manager if available
            if self.use_websocket and self.websocket_manager:
                logger.info(f"Submitting limit order via WebSocket: {side} {quantity} {symbol} @ ${price}")
                return await self.websocket_manager.submit_limit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    limit_price=price,
                    time_in_force=time_in_force or "day"
                )
            
            # Fallback to REST API
            order_data = {
                "symbol": symbol,
                "qty": str(quantity),
                "side": side,
                "type": "limit",
                "limit_price": str(price),
                "time_in_force": (time_in_force or "day").lower()
            }
            
            url = f"{self.config.alpaca_base_url}/v2/orders"
            async with self.alpaca_session.post(url, json=order_data) as response:
                if response.status == 200:
                    data = await response.json()
                    order_id = data.get("id", "")
                    
                    # Immediate confirmation
                    logger.info(f"🚀 ORDER SUBMITTED: {side.upper()} {quantity} {symbol}")
                    logger.info(f"   📋 Type: LIMIT")
                    logger.info(f"   🆔 Order ID: {order_id}")
                    logger.info(f"   📊 Symbol: {symbol}")
                    logger.info(f"   📦 Quantity: {quantity}")
                    logger.info(f"   💰 Limit Price: ${price}")
                    logger.info(f"   ⏰ Time-in-Force: {time_in_force or 'day'}")
                    logger.info(f"   ✅ Status: SUBMITTED TO ALPACA")
                    
                    # Start monitoring order status
                    asyncio.create_task(self._monitor_order_status(order_id, symbol, side, "limit"))
                    
                    return order_id
                else:
                    error_data = await response.json()
                    logger.error(f"❌ ORDER REJECTED: {error_data}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error submitting limit order: {e}")
            return None
    
    async def _monitor_order_status(self, order_id: str, symbol: str, side: str, order_type: str):
        """Monitor order status and provide real-time updates with better confirmation"""
        try:
            max_attempts = 30  # Increased from 10 to 30 seconds
            attempt = 0
            initial_confirmation = False
            
            logger.info(f"🔍 Starting order monitoring for {order_id}")
            
            while attempt < max_attempts:
                await asyncio.sleep(1)  # Check every second
                
                try:
                    url = f"{self.config.alpaca_base_url}/v2/orders/{order_id}"
                    async with self.alpaca_session.get(url) as response:
                        if response.status == 200:
                            order_data = await response.json()
                            status = order_data.get("status", "unknown")
                            
                            # Initial confirmation
                            if not initial_confirmation and status in ["new", "accepted", "pending_new"]:
                                logger.info(f"✅ ORDER CONFIRMED: {order_id} accepted by Alpaca")
                                logger.info(f"   📊 Symbol: {symbol}")
                                logger.info(f"   📈 Side: {side.upper()}")
                                logger.info(f"   📋 Type: {order_type.upper()}")
                                logger.info(f"   📦 Quantity: {order_data.get('qty', 'N/A')}")
                                if order_data.get('limit_price'):
                                    logger.info(f"   💰 Limit Price: ${order_data.get('limit_price')}")
                                logger.info(f"   ⏰ Time-in-Force: {order_data.get('time_in_force', 'N/A')}")
                                initial_confirmation = True
                            
                            logger.info(f"📊 Order {order_id} status: {status}")
                            
                            if status in ["filled", "canceled", "rejected", "expired"]:
                                # Order is final
                                if status == "filled":
                                    filled_qty = order_data.get("filled_qty", "0")
                                    filled_avg_price = order_data.get("filled_avg_price", "0")
                                    logger.info(f"🎉 ORDER FILLED: {order_id}")
                                    logger.info(f"   ✅ Filled Quantity: {filled_qty}")
                                    logger.info(f"   💰 Average Price: ${filled_avg_price}")
                                    logger.info(f"   📊 Symbol: {symbol}")
                                    
                                    # Update performance metrics
                                    self.total_trades += 1
                                    
                                    # Trigger callbacks
                                    for callback in self.trade_callbacks:
                                        try:
                                            callback({
                                                "order_id": order_id,
                                                "symbol": symbol,
                                                "side": side,
                                                "order_type": order_type,
                                                "status": status,
                                                "filled_qty": filled_qty,
                                                "filled_avg_price": filled_avg_price
                                            })
                                        except Exception as e:
                                            logger.error(f"Error in trade callback: {e}")
                                
                                elif status in ["canceled", "rejected", "expired"]:
                                    logger.warning(f"❌ ORDER {status.upper()}: {order_id}")
                                    if order_data.get("reject_reason"):
                                        logger.warning(f"   Reason: {order_data.get('reject_reason')}")
                                
                                break
                            else:
                                # Order still pending - show progress
                                if status in ["new", "accepted", "pending_new"]:
                                    logger.info(f"⏳ Order {order_id} PENDING - waiting for execution")
                                elif status in ["partially_filled"]:
                                    filled_qty = order_data.get("filled_qty", "0")
                                    logger.info(f"🔄 Order {order_id} PARTIALLY FILLED: {filled_qty} shares")
                        
                        attempt += 1
                        
                except Exception as e:
                    logger.error(f"Error checking order status: {e}")
                    attempt += 1
            
            if attempt >= max_attempts:
                logger.warning(f"⏰ Order {order_id} monitoring timeout after {max_attempts} seconds")
                logger.info(f"   Order may still be active - check Alpaca dashboard")
                
        except Exception as e:
            logger.error(f"Error monitoring order {order_id}: {e}")
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders for a symbol or all symbols"""
        try:
            open_orders = await self.get_open_orders(symbol)
            cancelled_count = 0
            
            logger.info(f"Found {len(open_orders)} open orders for {symbol or 'all symbols'}")
            
            for order in open_orders:
                order_id = order["id"]
                order_symbol = order.get("symbol", "Unknown")
                order_side = order.get("side", "Unknown")
                order_type = order.get("order_type", "Unknown")
                
                logger.info(f"Cancelling {order_side} {order_type} order for {order_symbol} (ID: {order_id})")
                
                if await self.cancel_order(order_id):
                    cancelled_count += 1
                    logger.info(f"✅ Successfully cancelled order {order_id}")
                else:
                    logger.warning(f"⚠️ Failed to cancel order {order_id}")
            
            logger.info(f"Successfully cancelled {cancelled_count}/{len(open_orders)} orders for {symbol or 'all symbols'}")
            return cancelled_count
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return 0
    
    async def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get current status of an order"""
        try:
            url = f"{self.config.alpaca_base_url}/v2/orders/{order_id}"
            async with self.alpaca_session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get order status: {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
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