"""
Quick Start HFT Trading Integration for BILLIONS System
This script demonstrates how to integrate the HFT engine with your existing system
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, List

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BILLIONSHFTIntegration:
    """Integration class for BILLIONS system with HFT engine"""
    
    def __init__(self):
        self.hft_manager = None
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize HFT engine with BILLIONS configuration"""
        try:
            # Import HFT manager
            from hft_trading_manager import HFTTradingManager, HFTConfig
            
            # Get configuration from environment or use defaults
            config = HFTConfig(
                polygon_api_key=os.getenv('POLYGON_API_KEY', ''),
                alpaca_api_key=os.getenv('ALPACA_API_KEY', ''),
                alpaca_secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
                alpaca_base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
                paper_trading=True,  # Always use paper trading for safety
                edge_threshold=float(os.getenv('HFT_EDGE_THRESHOLD', '0.001')),
                max_position_size=int(os.getenv('HFT_MAX_POSITION_SIZE', '1000')),
                max_daily_loss=float(os.getenv('HFT_MAX_DAILY_LOSS', '5000.0')),
                max_leverage=float(os.getenv('HFT_MAX_LEVERAGE', '2.0')),
                trading_symbols=[
                    "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA",  # Tech stocks
                    "SPY", "QQQ", "IWM"  # ETFs
                ]
            )
            
            # Create HFT manager
            self.hft_manager = HFTTradingManager(config)
            
            # Add callbacks for integration
            self.hft_manager.add_trade_callback(self._on_trade_executed)
            self.hft_manager.add_error_callback(self._on_error)
            self.hft_manager.add_performance_callback(self._on_performance_update)
            
            # Initialize engine
            if not self.hft_manager.initialize():
                raise Exception("Failed to initialize HFT engine")
            
            self.is_initialized = True
            logger.info("HFT engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize HFT engine: {e}")
            return False
    
    async def start_trading(self):
        """Start HFT trading engine"""
        if not self.is_initialized:
            logger.error("HFT engine not initialized")
            return False
        
        try:
            if not self.hft_manager.start():
                raise Exception("Failed to start HFT engine")
            
            logger.info("HFT trading engine started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start HFT engine: {e}")
            return False
    
    async def stop_trading(self):
        """Stop HFT trading engine"""
        if self.hft_manager:
            self.hft_manager.stop()
            logger.info("HFT trading engine stopped")
    
    async def submit_order(self, order_type: str, symbol: str, side: str, 
                          quantity: int, **kwargs) -> str:
        """Submit HFT order"""
        if not self.hft_manager or not self.hft_manager.is_running:
            raise Exception("HFT engine not running")
        
        try:
            if order_type == "market":
                order_id = self.hft_manager.submit_market_order(symbol, side, quantity)
            elif order_type == "limit":
                price = kwargs.get('price', 0.0)
                order_id = self.hft_manager.submit_limit_order(symbol, side, quantity, price)
            elif order_type == "twap":
                duration = kwargs.get('duration_minutes', 5)
                interval = kwargs.get('interval_seconds', 30)
                order_id = self.hft_manager.submit_twap_order(symbol, side, quantity, duration, interval)
            elif order_type == "vwap":
                volume_weight = kwargs.get('volume_weight', 0.1)
                order_id = self.hft_manager.submit_vwap_order(symbol, side, quantity, volume_weight)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            logger.info(f"Order submitted: {order_type} {side} {quantity} {symbol} -> {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        if not self.hft_manager:
            return {}
        
        try:
            metrics = self.hft_manager.get_performance_metrics()
            if not metrics:
                return {}
            
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
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    def get_status(self) -> Dict:
        """Get HFT engine status"""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.hft_manager.is_running if self.hft_manager else False,
            "total_trades": self.hft_manager.total_trades if self.hft_manager else 0,
            "total_pnl": self.hft_manager.total_pnl if self.hft_manager else 0.0,
            "uptime_seconds": (datetime.now() - self.hft_manager.start_time).total_seconds() if self.hft_manager and self.hft_manager.start_time else 0
        }
    
    def _on_trade_executed(self, symbol: str, side: str, quantity: int, price: float):
        """Handle trade execution callback"""
        logger.info(f"Trade executed: {side} {quantity} {symbol} @ ${price:.2f}")
        
        # Here you can integrate with your existing database
        # For example, save trade to your database
        # await self.save_trade_to_database(symbol, side, quantity, price)
    
    def _on_error(self, error: str):
        """Handle error callback"""
        logger.error(f"HFT Engine error: {error}")
        
        # Here you can integrate with your existing error handling
        # For example, send alerts or notifications
    
    def _on_performance_update(self, metrics):
        """Handle performance update callback"""
        logger.info(f"Performance update: P&L=${metrics.total_pnl:.2f}, "
                   f"Trades={metrics.total_trades}, Win Rate={metrics.win_rate:.2%}")
        
        # Here you can integrate with your existing performance tracking
        # For example, update your dashboard or send reports

# Global instance for easy access
hft_integration = BILLIONSHFTIntegration()

async def demo_hft_trading():
    """Demonstrate HFT trading capabilities"""
    
    print("🚀 BILLIONS HFT Trading Demo")
    print("=" * 50)
    
    # Initialize HFT engine
    print("Initializing HFT engine...")
    if not await hft_integration.initialize():
        print("❌ Failed to initialize HFT engine")
        return
    
    # Start trading
    print("Starting HFT trading...")
    if not await hft_integration.start_trading():
        print("❌ Failed to start HFT trading")
        return
    
    print("✅ HFT engine running!")
    print()
    
    # Wait for engine to stabilize
    print("Waiting for engine to stabilize...")
    await asyncio.sleep(5)
    
    # Demo different order types
    print("📈 Submitting demo orders...")
    
    try:
        # Market order
        order_id = await hft_integration.submit_order("market", "AAPL", "buy", 10)
        print(f"✅ Market order submitted: {order_id}")
        
        # Limit order
        order_id = await hft_integration.submit_order("limit", "MSFT", "sell", 5, price=300.0)
        print(f"✅ Limit order submitted: {order_id}")
        
        # TWAP order
        order_id = await hft_integration.submit_order("twap", "GOOGL", "buy", 20, 
                                                     duration_minutes=2, interval_seconds=30)
        print(f"✅ TWAP order submitted: {order_id}")
        
        # VWAP order
        order_id = await hft_integration.submit_order("vwap", "TSLA", "sell", 15, volume_weight=0.1)
        print(f"✅ VWAP order submitted: {order_id}")
        
    except Exception as e:
        print(f"❌ Error submitting orders: {e}")
    
    print()
    
    # Monitor performance for a few minutes
    print("📊 Monitoring performance for 2 minutes...")
    start_time = datetime.now()
    
    while (datetime.now() - start_time).total_seconds() < 120:  # 2 minutes
        await asyncio.sleep(10)  # Check every 10 seconds
        
        status = hft_integration.get_status()
        metrics = hft_integration.get_performance_metrics()
        
        print(f"Status: Running={status['is_running']}, "
              f"Trades={status['total_trades']}, "
              f"P&L=${status['total_pnl']:.2f}")
        
        if metrics:
            print(f"Metrics: Win Rate={metrics['win_rate']:.2%}, "
                  f"Avg Execution={metrics['avg_execution_time_ms']:.1f}ms")
    
    # Final performance report
    print()
    print("📋 Final Performance Report")
    print("=" * 30)
    
    final_metrics = hft_integration.get_performance_metrics()
    if final_metrics:
        print(f"Total Trades: {final_metrics['total_trades']}")
        print(f"Successful Trades: {final_metrics['successful_trades']}")
        print(f"Total P&L: ${final_metrics['total_pnl']:.2f}")
        print(f"Win Rate: {final_metrics['win_rate']:.2%}")
        print(f"Avg Execution Time: {final_metrics['avg_execution_time_ms']:.2f}ms")
        print(f"Fill Rate: {final_metrics['fill_rate']:.2%}")
        print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.2f}")
    
    # Stop trading
    print()
    print("Stopping HFT engine...")
    await hft_integration.stop_trading()
    print("✅ HFT engine stopped")

def main():
    """Main function"""
    try:
        # Run the demo
        asyncio.run(demo_hft_trading())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    finally:
        print("👋 Demo completed")

if __name__ == "__main__":
    main()
