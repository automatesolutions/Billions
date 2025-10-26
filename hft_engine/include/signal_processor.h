#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <chrono>

namespace HFT {

enum class OrderType {
    MARKET,
    LIMIT,
    LIMIT_EDGE,
    MARKET_EDGE,
    TWAP,
    TWAP_EDGE,
    VWAP
};

enum class OrderSide {
    BUY,
    SELL
};

enum class OrderStatus {
    PENDING,
    SUBMITTED,
    PARTIALLY_FILLED,
    FILLED,
    CANCELLED,
    REJECTED
};

struct Order {
    std::string order_id;
    std::string symbol;
    OrderType type;
    OrderSide side;
    int quantity;
    double price;
    double stop_price;
    std::chrono::milliseconds duration;  // For TWAP/VWAP
    std::chrono::milliseconds interval;  // For TWAP/VWAP
    OrderStatus status;
    int filled_quantity;
    double avg_fill_price;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    
    // Edge calculation parameters
    double edge_threshold;
    double volatility_factor;
    double liquidity_factor;
    
    // TWAP/VWAP specific
    std::vector<std::pair<std::chrono::milliseconds, int>> schedule;
    double target_volume_weight;
};

struct Signal {
    std::string symbol;
    double signal_strength;
    double confidence;
    std::chrono::system_clock::time_point timestamp;
    std::string signal_type;
    std::unordered_map<std::string, double> parameters;
};

class SignalProcessor {
public:
    using SignalCallback = std::function<void(const Signal&)>;
    
    SignalProcessor();
    ~SignalProcessor();
    
    // Signal processing methods
    void add_signal(const Signal& signal);
    void process_signals();
    
    // Edge calculation
    double calculate_market_edge(const std::string& symbol, OrderSide side);
    double calculate_limit_edge(const std::string& symbol, OrderSide side, double price);
    double calculate_twap_edge(const std::string& symbol, OrderSide side, 
                              std::chrono::milliseconds duration);
    double calculate_vwap_edge(const std::string& symbol, OrderSide side, 
                              double target_volume_weight);
    
    // Signal callbacks
    void set_signal_callback(SignalCallback callback);
    
    // Configuration
    void set_edge_threshold(double threshold);
    void set_volatility_factor(double factor);
    void set_liquidity_factor(double factor);
    
private:
    std::queue<Signal> signal_queue_;
    std::mutex signal_mutex_;
    std::condition_variable signal_cv_;
    std::thread processing_thread_;
    std::atomic<bool> running_;
    
    SignalCallback signal_callback_;
    
    // Edge calculation parameters
    double edge_threshold_;
    double volatility_factor_;
    double liquidity_factor_;
    
    // Market data cache for edge calculation
    std::unordered_map<std::string, double> latest_prices_;
    std::unordered_map<std::string, double> volatility_cache_;
    std::unordered_map<std::string, double> liquidity_cache_;
    
    std::mutex data_mutex_;
    
    // Processing methods
    void process_signal(const Signal& signal);
    void update_market_data_cache(const std::string& symbol);
    
    // Edge calculation helpers
    double calculate_volatility(const std::string& symbol);
    double calculate_liquidity(const std::string& symbol);
    double calculate_spread(const std::string& symbol);
};

class OrderManager {
public:
    using OrderCallback = std::function<void(const Order&)>;
    using FillCallback = std::function<void(const Order&, int filled_qty, double fill_price)>;
    
    OrderManager();
    ~OrderManager();
    
    // Order management
    std::string submit_order(const Order& order);
    void cancel_order(const std::string& order_id);
    void modify_order(const std::string& order_id, const Order& new_order);
    
    // Order status
    Order get_order(const std::string& order_id);
    std::vector<Order> get_active_orders();
    std::vector<Order> get_order_history();
    
    // TWAP/VWAP execution
    void execute_twap_order(const Order& order);
    void execute_vwap_order(const Order& order);
    
    // Callbacks
    void set_order_callback(OrderCallback callback);
    void set_fill_callback(FillCallback callback);
    
    // Performance metrics
    struct PerformanceMetrics {
        int total_orders;
        int filled_orders;
        int cancelled_orders;
        double total_pnl;
        double avg_fill_time_ms;
        double fill_rate;
        double slippage_avg;
    };
    
    PerformanceMetrics get_performance_metrics();
    
private:
    std::unordered_map<std::string, Order> orders_;
    std::mutex orders_mutex_;
    
    OrderCallback order_callback_;
    FillCallback fill_callback_;
    
    // TWAP/VWAP execution threads
    std::unordered_map<std::string, std::thread> execution_threads_;
    std::mutex execution_mutex_;
    
    // Performance tracking
    PerformanceMetrics metrics_;
    std::mutex metrics_mutex_;
    
    // Execution methods
    void execute_twap_thread(const Order& order);
    void execute_vwap_thread(const Order& order);
    void update_order_status(const std::string& order_id, OrderStatus status);
    void process_fill(const std::string& order_id, int filled_qty, double fill_price);
    
    // Performance calculation
    void update_performance_metrics(const Order& order);
};

} // namespace HFT
