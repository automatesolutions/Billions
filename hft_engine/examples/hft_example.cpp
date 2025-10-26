#include "hft_engine.h"
#include <iostream>
#include <chrono>
#include <thread>

using namespace HFT;

int main() {
    std::cout << "HFT Trading Engine Example" << std::endl;
    
    // Configuration
    std::string polygon_api_key = "YOUR_POLYGON_API_KEY";
    
    AlpacaCredentials alpaca_creds;
    alpaca_creds.api_key = "YOUR_ALPACA_API_KEY";
    alpaca_creds.secret_key = "YOUR_ALPACA_SECRET_KEY";
    alpaca_creds.base_url = "https://paper-api.alpaca.markets";
    alpaca_creds.paper_trading = true;
    
    // Create HFT engine
    auto engine = create_hft_engine(polygon_api_key, alpaca_creds);
    
    if (!engine) {
        std::cerr << "Failed to create HFT engine" << std::endl;
        return 1;
    }
    
    // Set trading symbols
    std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "TSLA"};
    engine->set_trading_symbols(symbols);
    
    // Configure engine
    engine->set_edge_threshold(0.001); // 0.1% edge threshold
    engine->set_max_position_size(1000); // Max 1000 shares per position
    engine->set_risk_limits(5000.0, 2.0); // Max $5000 daily loss, 2x leverage
    
    // Set callbacks
    engine->set_trade_callback([](const std::string& symbol, const std::string& side, 
                                 int quantity, double price) {
        std::cout << "Trade executed: " << side << " " << quantity 
                  << " shares of " << symbol << " at $" << price << std::endl;
    });
    
    engine->set_error_callback([](const std::string& error) {
        std::cerr << "Error: " << error << std::endl;
    });
    
    engine->set_performance_callback([](const PerformanceMonitor::TradingMetrics& metrics) {
        std::cout << "Performance Update:" << std::endl;
        std::cout << "  Total P&L: $" << metrics.total_pnl << std::endl;
        std::cout << "  Win Rate: " << (metrics.win_rate * 100) << "%" << std::endl;
        std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
        std::cout << "  Avg Execution Time: " << metrics.avg_execution_time_ms << "ms" << std::endl;
    });
    
    // Initialize and start engine
    if (!engine->initialize()) {
        std::cerr << "Failed to initialize HFT engine" << std::endl;
        return 1;
    }
    
    std::cout << "Starting HFT engine..." << std::endl;
    engine->start();
    
    // Example: Submit different order types
    std::this_thread::sleep_for(std::chrono::seconds(5));
    
    // Market order
    std::string market_order_id = engine->submit_market_order("AAPL", "buy", 100);
    std::cout << "Submitted market order: " << market_order_id << std::endl;
    
    // Limit order
    std::string limit_order_id = engine->submit_limit_order("MSFT", "sell", 50, 300.0);
    std::cout << "Submitted limit order: " << limit_order_id << std::endl;
    
    // TWAP order (execute over 5 minutes with 30-second intervals)
    std::string twap_order_id = engine->submit_twap_order("GOOGL", "buy", 200, 
                                                         std::chrono::minutes(5), 
                                                         std::chrono::seconds(30));
    std::cout << "Submitted TWAP order: " << twap_order_id << std::endl;
    
    // VWAP order
    std::string vwap_order_id = engine->submit_vwap_order("TSLA", "sell", 150, 0.1);
    std::cout << "Submitted VWAP order: " << vwap_order_id << std::endl;
    
    // Run for 10 minutes
    std::cout << "Running for 10 minutes..." << std::endl;
    std::this_thread::sleep_for(std::chrono::minutes(10));
    
    // Get performance metrics
    auto metrics = engine->get_performance_metrics();
    std::cout << "\nFinal Performance Metrics:" << std::endl;
    std::cout << "  Total Trades: " << metrics.total_trades << std::endl;
    std::cout << "  Successful Trades: " << metrics.successful_trades << std::endl;
    std::cout << "  Total P&L: $" << metrics.total_pnl << std::endl;
    std::cout << "  Win Rate: " << (metrics.win_rate * 100) << "%" << std::endl;
    std::cout << "  Avg Execution Time: " << metrics.avg_execution_time_ms << "ms" << std::endl;
    std::cout << "  Fill Rate: " << (metrics.fill_rate * 100) << "%" << std::endl;
    
    // Stop engine
    std::cout << "Stopping HFT engine..." << std::endl;
    engine->stop();
    engine->shutdown();
    
    std::cout << "HFT engine stopped successfully" << std::endl;
    return 0;
}
