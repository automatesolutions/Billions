#pragma once

#include "market_data_ingestion.h"
#include "signal_processor.h"
#include "order_executor.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace HFT {

class HFTTradingEngine {
public:
    HFTTradingEngine(const std::string& polygon_api_key,
                     const AlpacaCredentials& alpaca_credentials);
    ~HFTTradingEngine();
    
    // Engine lifecycle
    bool initialize();
    void start();
    void stop();
    void shutdown();
    
    // Configuration
    void set_trading_symbols(const std::vector<std::string>& symbols);
    void set_edge_threshold(double threshold);
    void set_max_position_size(double max_size);
    void set_risk_limits(double max_daily_loss, double max_leverage);
    
    // Strategy management
    void enable_strategy(const std::string& strategy_name);
    void disable_strategy(const std::string& strategy_name);
    void set_strategy_parameters(const std::string& strategy_name, 
                                const std::unordered_map<std::string, double>& params);
    
    // Order management
    std::string submit_market_order(const std::string& symbol, 
                                   const std::string& side, 
                                   int quantity);
    
    std::string submit_limit_order(const std::string& symbol, 
                                  const std::string& side, 
                                  int quantity, 
                                  double price);
    
    std::string submit_twap_order(const std::string& symbol, 
                                 const std::string& side, 
                                 int quantity, 
                                 std::chrono::milliseconds duration,
                                 std::chrono::milliseconds interval);
    
    std::string submit_vwap_order(const std::string& symbol, 
                                 const std::string& side, 
                                 int quantity, 
                                 double target_volume_weight);
    
    // Status and monitoring
    bool is_running() const;
    PerformanceMonitor::TradingMetrics get_performance_metrics();
    AlpacaOrderExecutor::ExecutionMetrics get_execution_metrics();
    
    // Callbacks
    void set_trade_callback(std::function<void(const std::string&, const std::string&, int, double)> callback);
    void set_error_callback(std::function<void(const std::string&)> callback);
    void set_performance_callback(std::function<void(const PerformanceMonitor::TradingMetrics&)> callback);
    
private:
    // Core components
    std::unique_ptr<PolygonMarketDataIngestion> market_data_;
    std::unique_ptr<SignalProcessor> signal_processor_;
    std::unique_ptr<OrderManager> order_manager_;
    std::unique_ptr<AlpacaOrderExecutor> order_executor_;
    std::unique_ptr<PerformanceMonitor> performance_monitor_;
    
    // Configuration
    std::string polygon_api_key_;
    AlpacaCredentials alpaca_credentials_;
    std::vector<std::string> trading_symbols_;
    
    // Engine state
    std::atomic<bool> initialized_;
    std::atomic<bool> running_;
    std::atomic<bool> shutdown_requested_;
    
    // Threading
    std::thread main_thread_;
    std::mutex engine_mutex_;
    std::condition_variable engine_cv_;
    
    // Callbacks
    std::function<void(const std::string&, const std::string&, int, double)> trade_callback_;
    std::function<void(const std::string&)> error_callback_;
    std::function<void(const PerformanceMonitor::TradingMetrics&)> performance_callback_;
    
    // Main engine loop
    void main_loop();
    
    // Event handlers
    void on_market_data(const MarketData& data);
    void on_orderbook_update(const OrderBook& orderbook);
    void on_trade_update(const Trade& trade);
    void on_signal_generated(const Signal& signal);
    void on_order_filled(const Order& order, int filled_qty, double fill_price);
    void on_performance_update(const PerformanceMonitor::TradingMetrics& metrics);
    
    // Strategy execution
    void execute_strategy(const Signal& signal);
    void process_edge_calculation(const std::string& symbol, OrderSide side);
    
    // Risk management
    bool check_risk_limits(const std::string& symbol, const std::string& side, int quantity, double price);
    void update_position_tracking(const std::string& symbol, const std::string& side, int quantity, double price);
    
    // Utility methods
    std::string generate_order_id();
    void log_error(const std::string& error_message);
    void log_info(const std::string& info_message);
    
    // Configuration validation
    bool validate_configuration();
    bool test_connections();
};

// Factory function for easy instantiation
std::unique_ptr<HFTTradingEngine> create_hft_engine(
    const std::string& polygon_api_key,
    const AlpacaCredentials& alpaca_credentials);

} // namespace HFT
