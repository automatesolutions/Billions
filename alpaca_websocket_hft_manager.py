import asyncio
import logging
import os
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json

# Try to import the C++ WebSocket client
try:
    import alpaca_websocket
    CPP_WEBSOCKET_AVAILABLE = True
except ImportError:
    CPP_WEBSOCKET_AVAILABLE = False
    logging.warning("C++ WebSocket client not available, falling back to REST API")

logger = logging.getLogger(__name__)

@dataclass
class OrderRequest:
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "limit", "market", "stop", "stop_limit"
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "day"  # "day", "gtc", "ioc", "fok"
    client_order_id: Optional[str] = None

@dataclass
class OrderResponse:
    order_id: str
    client_order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    limit_price: float
    stop_price: float
    time_in_force: str
    status: str
    created_at: str
    updated_at: str
    filled_avg_price: float
    filled_qty: int
    remaining_qty: int
    reject_reason: str

@dataclass
class FillNotification:
    order_id: str
    symbol: str
    side: str
    filled_qty: int
    filled_price: float
    filled_at: str
    trade_id: str

class AlpacaWebSocketHFTManager:
    """High-Frequency Trading Manager using C++ WebSocket client for Alpaca"""
    
    def __init__(self, api_key: str, secret_key: str, use_paper_trading: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.use_paper_trading = use_paper_trading
        
        self.client = None
        self.is_connected = False
        self.order_callbacks: Dict[str, Callable] = {}
        self.fill_callbacks: Dict[str, Callable] = {}
        
        # Performance tracking
        self.total_orders = 0
        self.filled_orders = 0
        self.total_pnl = 0.0
        self.order_latencies = []
        
        if CPP_WEBSOCKET_AVAILABLE:
            self._initialize_cpp_client()
        else:
            logger.warning("C++ WebSocket client not available")
    
    def _initialize_cpp_client(self):
        """Initialize the C++ WebSocket client"""
        try:
            self.client = alpaca_websocket.AlpacaWebSocketClient(
                self.api_key, 
                self.secret_key, 
                self.use_paper_trading
            )
            
            # Set up callbacks
            self.client.set_order_callback(self._on_order_update)
            self.client.set_fill_callback(self._on_fill_notification)
            self.client.set_error_callback(self._on_error)
            self.client.set_connection_callback(self._on_connection_change)
            
            logger.info("C++ WebSocket client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize C++ WebSocket client: {e}")
            self.client = None
    
    async def connect(self) -> bool:
        """Connect to Alpaca WebSocket"""
        if not self.client:
            logger.error("C++ WebSocket client not available")
            return False
        
        try:
            success = self.client.connect()
            if success:
                logger.info("Connected to Alpaca WebSocket")
                self.is_connected = True
            else:
                logger.error("Failed to connect to Alpaca WebSocket")
            return success
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca WebSocket"""
        if self.client:
            self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from Alpaca WebSocket")
    
    async def submit_limit_order(self, symbol: str, side: str, quantity: int, 
                               limit_price: float, time_in_force: str = "day") -> str:
        """Submit a limit order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca WebSocket")
        
        start_time = datetime.now()
        
        try:
            client_order_id = self.client.submit_order(
                symbol=symbol,
                side=side,
                order_type="limit",
                quantity=quantity,
                limit_price=limit_price,
                time_in_force=time_in_force
            )
            
            # Track latency
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.order_latencies.append(latency)
            self.total_orders += 1
            
            logger.info(f"Limit order submitted: {symbol} {side} {quantity} @ ${limit_price}")
            return client_order_id
            
        except Exception as e:
            logger.error(f"Failed to submit limit order: {e}")
            raise
    
    async def submit_market_order(self, symbol: str, side: str, quantity: int) -> str:
        """Submit a market order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca WebSocket")
        
        start_time = datetime.now()
        
        try:
            client_order_id = self.client.submit_order(
                symbol=symbol,
                side=side,
                order_type="market",
                quantity=quantity
            )
            
            # Track latency
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.order_latencies.append(latency)
            self.total_orders += 1
            
            logger.info(f"Market order submitted: {symbol} {side} {quantity}")
            return client_order_id
            
        except Exception as e:
            logger.error(f"Failed to submit market order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca WebSocket")
        
        try:
            success = self.client.cancel_order(order_id)
            if success:
                logger.info(f"Order cancellation requested: {order_id}")
            else:
                logger.error(f"Failed to cancel order: {order_id}")
            return success
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        if not self.is_connected:
            raise RuntimeError("Not connected to Alpaca WebSocket")
        
        try:
            success = self.client.cancel_all_orders()
            if success:
                logger.info("All orders cancellation requested")
            else:
                logger.error("Failed to cancel all orders")
            return success
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            return False
    
    def subscribe_to_trades(self, symbols: list) -> bool:
        """Subscribe to trade data for symbols"""
        if not self.is_connected:
            return False
        
        try:
            success = self.client.subscribe_to_trades(symbols)
            if success:
                logger.info(f"Subscribed to trades: {symbols}")
            return success
        except Exception as e:
            logger.error(f"Error subscribing to trades: {e}")
            return False
    
    def subscribe_to_quotes(self, symbols: list) -> bool:
        """Subscribe to quote data for symbols"""
        if not self.is_connected:
            return False
        
        try:
            success = self.client.subscribe_to_quotes(symbols)
            if success:
                logger.info(f"Subscribed to quotes: {symbols}")
            return success
        except Exception as e:
            logger.error(f"Error subscribing to quotes: {e}")
            return False
    
    def set_order_callback(self, callback: Callable[[OrderResponse], None]):
        """Set callback for order updates"""
        self.order_callbacks['default'] = callback
    
    def set_fill_callback(self, callback: Callable[[FillNotification], None]):
        """Set callback for fill notifications"""
        self.fill_callbacks['default'] = callback
    
    def _on_order_update(self, order_data):
        """Handle order update from C++ client"""
        try:
            order = OrderResponse(
                order_id=order_data.get('order_id', ''),
                client_order_id=order_data.get('client_order_id', ''),
                symbol=order_data.get('symbol', ''),
                side=order_data.get('side', ''),
                order_type=order_data.get('order_type', ''),
                quantity=order_data.get('quantity', 0),
                limit_price=order_data.get('limit_price', 0.0),
                stop_price=order_data.get('stop_price', 0.0),
                time_in_force=order_data.get('time_in_force', ''),
                status=order_data.get('status', ''),
                created_at=order_data.get('created_at', ''),
                updated_at=order_data.get('updated_at', ''),
                filled_avg_price=order_data.get('filled_avg_price', 0.0),
                filled_qty=order_data.get('filled_qty', 0),
                remaining_qty=order_data.get('remaining_qty', 0),
                reject_reason=order_data.get('reject_reason', '')
            )
            
            logger.info(f"Order update: {order.symbol} {order.side} {order.status}")
            
            # Call registered callbacks
            for callback in self.order_callbacks.values():
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in order callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing order update: {e}")
    
    def _on_fill_notification(self, fill_data):
        """Handle fill notification from C++ client"""
        try:
            fill = FillNotification(
                order_id=fill_data.get('order_id', ''),
                symbol=fill_data.get('symbol', ''),
                side=fill_data.get('side', ''),
                filled_qty=fill_data.get('filled_qty', 0),
                filled_price=fill_data.get('filled_price', 0.0),
                filled_at=fill_data.get('filled_at', ''),
                trade_id=fill_data.get('trade_id', '')
            )
            
            logger.info(f"Fill notification: {fill.symbol} {fill.side} {fill.filled_qty} @ ${fill.filled_price}")
            
            # Update performance metrics
            self.filled_orders += 1
            
            # Call registered callbacks
            for callback in self.fill_callbacks.values():
                try:
                    callback(fill)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing fill notification: {e}")
    
    def _on_error(self, error_msg: str):
        """Handle error from C++ client"""
        logger.error(f"Alpaca WebSocket error: {error_msg}")
    
    def _on_connection_change(self, connected: bool):
        """Handle connection status change"""
        self.is_connected = connected
        if connected:
            logger.info("Connected to Alpaca WebSocket")
        else:
            logger.warning("Disconnected from Alpaca WebSocket")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        win_rate = (self.filled_orders / self.total_orders * 100) if self.total_orders > 0 else 0
        avg_latency = sum(self.order_latencies) / len(self.order_latencies) if self.order_latencies else 0
        
        return {
            'total_trades': self.total_orders,
            'filled_trades': self.filled_orders,
            'win_rate': win_rate / 100,  # Convert to decimal
            'total_pnl': self.total_pnl,
            'avg_latency_ms': avg_latency,
            'is_connected': self.is_connected
        }

# Factory function to create HFT manager
def create_alpaca_hft_manager(api_key: str = None, secret_key: str = None, 
                            use_paper_trading: bool = True) -> AlpacaWebSocketHFTManager:
    """Create an Alpaca WebSocket HFT manager"""
    if not api_key:
        api_key = os.getenv('ALPACA_API_KEY')
    if not secret_key:
        secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError("Alpaca API credentials not provided")
    
    return AlpacaWebSocketHFTManager(api_key, secret_key, use_paper_trading)
