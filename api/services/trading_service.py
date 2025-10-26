import os
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from api.config import settings

logger = logging.getLogger(__name__)

# HFT Engine Integration (optional)
try:
    from hft_trading_manager import HFTTradingManager, HFTConfig
    HFT_AVAILABLE = True
    logger.info("HFT Trading Engine Python bindings loaded successfully")
except ImportError:
    try:
        from hft_trading_manager_simple import HFTTradingManager, HFTConfig
        HFT_AVAILABLE = True
        logger.info("Using simplified HFT Trading Manager (Python-only)")
    except ImportError:
        HFT_AVAILABLE = False
        logger.info("HFT Trading Engine not available. Using standard trading only.")

class PolygonService:
    """Polygon.io integration for real-time market data and orderbook"""
    
    def __init__(self):
        self.api_key = os.getenv('POLYGON_API_KEY', '')
        self.base_url = "https://api.polygon.io"
        self.ws_url = "wss://socket.polygon.io/stocks"
        self.session = None
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v2/last/trade/{symbol}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "symbol": symbol,
                        "price": data.get("results", {}).get("p", 0),
                        "timestamp": data.get("results", {}).get("t", 0),
                        "volume": data.get("results", {}).get("s", 0),
                        "status": "success"
                    }
                else:
                    logger.error(f"Polygon API error: {response.status}")
                    return {"symbol": symbol, "status": "error", "message": "API error"}
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "message": str(e)}
    
    async def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """Get orderbook data for a symbol"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    ticker_data = data.get("ticker", {})
                    
                    return {
                        "symbol": symbol,
                        "bid": ticker_data.get("bid", 0),
                        "ask": ticker_data.get("ask", 0),
                        "bid_size": ticker_data.get("bidSize", 0),
                        "ask_size": ticker_data.get("askSize", 0),
                        "last_price": ticker_data.get("lastTrade", {}).get("p", 0),
                        "volume": ticker_data.get("day", {}).get("v", 0),
                        "status": "success"
                    }
                else:
                    logger.error(f"Polygon orderbook error: {response.status}")
                    return {"symbol": symbol, "status": "error", "message": "API error"}
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {"symbol": symbol, "status": "error", "message": str(e)}
    
    async def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v1/marketstatus/now"
            params = {"apikey": self.api_key}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "market": data.get("market", "unknown"),
                        "serverTime": data.get("serverTime", ""),
                        "exchanges": data.get("exchanges", {}),
                        "currencies": data.get("currencies", {}),
                        "status": "success"
                    }
                else:
                    return {"status": "error", "message": "API error"}
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class AlpacaService:
    """Alpaca integration for paper trading"""
    
    def __init__(self):
        self.api_key = settings.ALPACA_API_KEY or os.getenv('ALPACA_API_KEY', '')
        self.secret_key = settings.ALPACA_SECRET_KEY or os.getenv('ALPACA_SECRET_KEY', '')
        self.base_url = "https://paper-api.alpaca.markets"  # Paper trading URL
        self.data_url = "https://data.alpaca.markets"
        self.session = None
        
    async def get_session(self):
        """Get or create aiohttp session with Alpaca headers"""
        if not self.session:
            # Validate API keys before creating session
            if not self.api_key or not self.secret_key:
                raise ValueError("Alpaca API keys are not configured")
            
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v2/account"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "account_id": data.get("id", ""),
                        "buying_power": float(data.get("buying_power", 0)),
                        "cash": float(data.get("cash", 0)),
                        "portfolio_value": float(data.get("portfolio_value", 0)),
                        "equity": float(data.get("equity", 0)),
                        "account_status": data.get("status", ""),
                        "currency": data.get("currency", "USD"),
                        "unrealized_pl": float(data.get("unrealized_pl", 0)),
                        "unrealized_plpc": float(data.get("unrealized_plpc", 0)),
                        "status": "success"
                    }
                else:
                    logger.error(f"Alpaca account error: {response.status}")
                    return {"status": "error", "message": "API error"}
        except Exception as e:
            logger.error(f"Error fetching account: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v2/positions"
            
            async with session.get(url) as response:
                if response.status == 200:
                    positions = await response.json()
                    return [
                        {
                            "symbol": pos.get("symbol", ""),
                            "qty": int(pos.get("qty", 0)),
                            "side": pos.get("side", ""),
                            "market_value": float(pos.get("market_value", 0)),
                            "cost_basis": float(pos.get("cost_basis", 0)),
                            "unrealized_pl": float(pos.get("unrealized_pl", 0)),
                            "unrealized_plpc": float(pos.get("unrealized_plpc", 0)),
                            "current_price": float(pos.get("current_price", 0)),
                            "status": "success"
                        }
                        for pos in positions
                    ]
                else:
                    logger.error(f"Alpaca positions error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def place_order(self, symbol: str, qty: int, side: str, order_type: str = "market") -> Dict[str, Any]:
        """Place a paper trading order"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v2/orders"
            
            order_data = {
                "symbol": symbol,
                "qty": str(qty),
                "side": side,  # "buy" or "sell"
                "type": order_type,  # "market", "limit", "stop", etc.
                "time_in_force": "day"
            }
            
            async with session.post(url, json=order_data) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "order_id": data.get("id", ""),
                        "symbol": data.get("symbol", ""),
                        "qty": data.get("qty", ""),
                        "side": data.get("side", ""),
                        "order_status": data.get("status", ""),
                        "submitted_at": data.get("submitted_at", ""),
                        "status": "success"
                    }
                else:
                    error_data = await response.json()
                    logger.error(f"Alpaca order error: {response.status} - {error_data}")
                    return {"status": "error", "message": error_data.get("message", "Order failed")}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_orders(self, status: str = "all") -> List[Dict[str, Any]]:
        """Get order history"""
        try:
            session = await self.get_session()
            url = f"{self.base_url}/v2/orders"
            params = {"status": status}
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    orders = await response.json()
                    return [
                        {
                            "id": order.get("id", ""),
                            "symbol": order.get("symbol", ""),
                            "qty": order.get("qty", ""),
                            "side": order.get("side", ""),
                            "order_status": order.get("status", ""),
                            "submitted_at": order.get("submitted_at", ""),
                            "filled_at": order.get("filled_at", ""),
                            "filled_qty": order.get("filled_qty", ""),
                            "filled_avg_price": order.get("filled_avg_price", ""),
                            "order_type": order.get("type", ""),
                            "status": "success"
                        }
                        for order in orders
                    ]
                else:
                    logger.error(f"Alpaca orders error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching orders: {e}")
            return []
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

class TradingService:
    """Main trading service that combines Polygon and Alpaca with optional HFT engine"""
    
    def __init__(self):
        self.polygon = PolygonService()
        self.alpaca = AlpacaService()
        self.is_connected = False
        
        # HFT Engine (optional)
        self.hft_manager = None
        self.hft_available = HFT_AVAILABLE
    
    async def initialize(self):
        """Initialize both services and optional HFT engine"""
        try:
            # Test Alpaca connection (required)
            account_info = await self.alpaca.get_account()
            
            if account_info.get("status") == "success":
                self.is_connected = True
                logger.info("Alpaca trading service initialized successfully")
                
                # Test Polygon connection (optional)
                if self.polygon.api_key:
                    try:
                        market_status = await self.polygon.get_market_status()
                        if market_status.get("status") == "success":
                            logger.info("Polygon market data service initialized successfully")
                        else:
                            logger.warning("Polygon market data service not available")
                    except Exception as e:
                        logger.warning(f"Polygon service not available: {e}")
                
                # Initialize HFT engine (optional)
                if self.hft_available:
                    await self._initialize_hft_engine()
                
                return True
            else:
                logger.error("Failed to initialize Alpaca trading service")
                return False
        except Exception as e:
            logger.error(f"Error initializing trading services: {e}")
            return False
    
    async def _initialize_hft_engine(self):
        """Initialize HFT engine if available"""
        try:
            # Get Polygon API key from environment
            polygon_api_key = os.getenv('POLYGON_API_KEY', '')
            
            # Create HFT configuration
            config = HFTConfig(
                polygon_api_key=polygon_api_key,
                alpaca_api_key=settings.ALPACA_API_KEY or os.getenv('ALPACA_API_KEY', ''),
                alpaca_secret_key=settings.ALPACA_SECRET_KEY or os.getenv('ALPACA_SECRET_KEY', ''),
                alpaca_base_url="https://paper-api.alpaca.markets",
                paper_trading=True,
                edge_threshold=0.001,  # 0.1%
                max_position_size=1000,
                max_daily_loss=5000.0,
                max_leverage=2.0,
                trading_symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            )
            
            # Create HFT manager
            self.hft_manager = HFTTradingManager(config)
            
            # Initialize engine
            if self.hft_manager.initialize():
                logger.info("HFT Trading Engine initialized successfully")
            else:
                logger.warning("Failed to initialize HFT Trading Engine")
                self.hft_manager = None
                
        except Exception as e:
            logger.warning(f"HFT Engine initialization failed: {e}")
            self.hft_manager = None
    
    async def get_portfolio_data(self) -> Dict[str, Any]:
        """Get combined portfolio data from Alpaca"""
        try:
            account = await self.alpaca.get_account()
            positions = await self.alpaca.get_positions()
            
            if account.get("status") == "success":
                return {
                    "account": account,
                    "positions": positions,
                    "total_positions": len(positions),
                    "status": "success"
                }
            else:
                return {"status": "error", "message": "Failed to fetch portfolio data"}
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return {"status": "error", "message": str(e)}
    
    async def execute_trade(self, symbol: str, qty: int, side: str, order_type: str = "market") -> Dict[str, Any]:
        """Execute a trade through Alpaca"""
        try:
            # Get real-time quote from Polygon first
            quote = await self.polygon.get_real_time_quote(symbol)
            
            if quote.get("status") == "success":
                # Place order through Alpaca
                order_result = await self.alpaca.place_order(symbol, qty, side, order_type)
                
                return {
                    "symbol": symbol,
                    "current_price": quote.get("price", 0),
                    "order_result": order_result,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            else:
                return {"status": "error", "message": "Failed to get real-time quote"}
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get market data for multiple symbols"""
        try:
            tasks = []
            for symbol in symbols:
                tasks.append(self.polygon.get_real_time_quote(symbol))
                tasks.append(self.polygon.get_orderbook(symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_data = {}
            for i in range(0, len(results), 2):
                symbol = symbols[i // 2]
                quote = results[i] if not isinstance(results[i], Exception) else {"status": "error"}
                orderbook = results[i + 1] if not isinstance(results[i + 1], Exception) else {"status": "error"}
                
                market_data[symbol] = {
                    "quote": quote,
                    "orderbook": orderbook
                }
            
            return {
                "market_data": market_data,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"status": "error", "message": str(e)}
    
    # HFT Engine Methods
    async def start_hft_engine(self) -> bool:
        """Start HFT engine if available"""
        if not self.hft_manager:
            logger.warning("HFT engine not available")
            return False
        
        try:
            return self.hft_manager.start()
        except Exception as e:
            logger.error(f"Failed to start HFT engine: {e}")
            return False
    
    async def stop_hft_engine(self):
        """Stop HFT engine if running"""
        if self.hft_manager:
            self.hft_manager.stop()
    
    async def submit_hft_order(self, order_type: str, symbol: str, side: str, 
                              quantity: int, **kwargs) -> Optional[str]:
        """Submit HFT order with advanced order types"""
        # Auto-start HFT engine if not running
        if not self.hft_manager:
            await self._initialize_hft_engine()
        
        if not self.hft_manager:
            raise Exception("Failed to initialize HFT engine")
        
        # Start engine if not running
        if not self.hft_manager.is_running:
            logger.info("Auto-starting HFT engine for order submission")
            await self.start_hft_engine()
        
        try:
            if order_type == "market":
                return await self.hft_manager.submit_market_order(symbol, side, quantity)
            elif order_type == "limit":
                price = kwargs.get('price', 0.0)
                time_in_force = kwargs.get('time_in_force', 'day')
                return await self.hft_manager.submit_limit_order(symbol, side, quantity, price, time_in_force)
            elif order_type == "twap":
                duration = kwargs.get('duration_minutes', 5)
                interval = kwargs.get('interval_seconds', 30)
                return await self.hft_manager.submit_twap_order(symbol, side, quantity, duration, interval)
            elif order_type == "vwap":
                volume_weight = kwargs.get('volume_weight', 0.1)
                return await self.hft_manager.submit_vwap_order(symbol, side, quantity, volume_weight)
            else:
                raise ValueError(f"Unsupported HFT order type: {order_type}")
        except Exception as e:
            logger.error(f"Failed to submit HFT order: {e}")
            raise
    
    def get_hft_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """Get HFT performance metrics"""
        if not self.hft_manager:
            return None
        
        try:
            metrics = self.hft_manager.get_performance_metrics()
            if not metrics:
                return None
            
            # Handle both dict and object returns
            if isinstance(metrics, dict):
                return metrics
            else:
                return {
                    "total_trades": metrics.total_trades,
                    "successful_trades": metrics.successful_trades,
                    "failed_trades": metrics.failed_trades,
                    "total_pnl": metrics.total_pnl,
                    "win_rate": metrics.win_rate,
                    "avg_execution_time_ms": metrics.avg_execution_time_ms,
                    "fill_rate": metrics.fill_rate,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown
                }
        except Exception as e:
            logger.error(f"Failed to get HFT performance metrics: {e}")
            return None
    
    def get_hft_status(self) -> Dict[str, Any]:
        """Get HFT engine status"""
        if not self.hft_manager:
            return {
                "available": False,
                "running": False,
                "error": "HFT engine not available"
            }
        
        return {
            "available": True,
            "running": self.hft_manager.is_running,
            "total_trades": self.hft_manager.total_trades,
            "total_pnl": self.hft_manager.total_pnl,
            "uptime_seconds": (datetime.now() - self.hft_manager.start_time).total_seconds() if self.hft_manager.start_time else 0
        }
    
    async def close(self):
        """Close all services"""
        await self.polygon.close()
        await self.alpaca.close()
        if self.hft_manager:
            self.hft_manager.stop()

# Global trading service instance
trading_service = TradingService()
