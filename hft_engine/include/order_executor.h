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
#include <curl/curl.h>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>

namespace HFT {

struct AlpacaCredentials {
    std::string api_key;
    std::string secret_key;
    std::string base_url;
    bool paper_trading;
};

class AlpacaOrderExecutor {
public:
    using OrderStatusCallback = std::function<void(const std::string& order_id, 
                                                   const std::string& status)>;
    using FillCallback = std::function<void(const std::string& order_id, 
                                           int filled_qty, double fill_price)>;
    
    AlpacaOrderExecutor(const AlpacaCredentials& credentials);
    ~AlpacaOrderExecutor();
    
    // Order execution
    std::string submit_market_order(const std::string& symbol, 
                                   const std::string& side, 
                                   int quantity);
    
    std::string submit_limit_order(const std::string& symbol, 
                                  const std::string& side, 
                                  int quantity, 
                                  double price);
    
    std::string submit_stop_order(const std::string& symbol, 
                                 const std::string& side, 
                                 int quantity, 
                                 double stop_price);
    
    std::string submit_stop_limit_order(const std::string& symbol, 
                                       const std::string& side, 
                                       int quantity, 
                                       double limit_price, 
                                       double stop_price);
    
    // Order management
    void cancel_order(const std::string& order_id);
    void modify_order(const std::string& order_id, 
                     int quantity, 
                     double price);
    
    // Order status
    struct OrderStatus {
        std::string order_id;
        std::string symbol;
        std::string side;
        int quantity;
        int filled_quantity;
        double avg_fill_price;
        std::string status;
        std::string order_type;
        std::chrono::system_clock::time_point created_at;
        std::chrono::system_clock::time_point updated_at;
    };
    
    OrderStatus get_order_status(const std::string& order_id);
    std::vector<OrderStatus> get_active_orders();
    std::vector<OrderStatus> get_order_history();
    
    // Account information
    struct AccountInfo {
        std::string account_id;
        double buying_power;
        double cash;
        double portfolio_value;
        double equity;
        std::string status;
        std::string currency;
        double unrealized_pl;
        double unrealized_plpc;
    };
    
    AccountInfo get_account_info();
    
    // Positions
    struct Position {
        std::string symbol;
        int quantity;
        std::string side;
        double market_value;
        double cost_basis;
        double unrealized_pl;
        double unrealized_plpc;
        double current_price;
    };
    
    std::vector<Position> get_positions();
    Position get_position(const std::string& symbol);
    
    // WebSocket streaming
    void start_streaming();
    void stop_streaming();
    
    // Callbacks
    void set_order_status_callback(OrderStatusCallback callback);
    void set_fill_callback(FillCallback callback);
    
    // Performance metrics
    struct ExecutionMetrics {
        int total_orders;
        int successful_orders;
        int failed_orders;
        double avg_execution_time_ms;
        double total_slippage;
        double avg_slippage;
        double fill_rate;
    };
    
    ExecutionMetrics get_execution_metrics();
    
private:
    AlpacaCredentials credentials_;
    CURL* curl_;
    
    // WebSocket for real-time updates
    websocketpp::client<websocketpp::config::asio> ws_client_;
    websocketpp::connection_hdl ws_hdl_;
    std::thread ws_thread_;
    std::atomic<bool> ws_running_;
    
    // Callbacks
    OrderStatusCallback order_status_callback_;
    FillCallback fill_callback_;
    
    // Performance tracking
    ExecutionMetrics metrics_;
    std::mutex metrics_mutex_;
    
    // HTTP methods
    std::string make_http_request(const std::string& endpoint, 
                                 const std::string& method = "GET",
                                 const std::string& body = "");
    
    std::string build_auth_headers();
    std::string serialize_order_data(const std::string& symbol, 
                                     const std::string& side, 
                                     int quantity, 
                                     const std::string& type,
                                     double price = 0.0,
                                     double stop_price = 0.0);
    
    // WebSocket handlers
    void on_ws_open(websocketpp::connection_hdl hdl);
    void on_ws_message(websocketpp::connection_hdl hdl, 
                      websocketpp::config::asio_client::message_ptr msg);
    void on_ws_close(websocketpp::connection_hdl hdl);
    void on_ws_fail(websocketpp::connection_hdl hdl);
    
    // Message processing
    void process_ws_message(const std::string& message);
    void process_order_update(const nlohmann::json& data);
    void process_fill_update(const nlohmann::json& data);
    
    // WebSocket thread function
    void websocket_thread_func();
    
    // Performance tracking
    void update_execution_metrics(const std::string& order_id, 
                                 bool success, 
                                 double execution_time_ms,
                                 double slippage = 0.0);
    
    // Utility methods
    std::string generate_order_id();
    double get_current_timestamp();
};

class PerformanceMonitor {
public:
    struct TradingMetrics {
        // Execution metrics
        int total_trades;
        int successful_trades;
        int failed_trades;
        double total_pnl;
        double realized_pnl;
        double unrealized_pnl;
        
        // Performance metrics
        double win_rate;
        double avg_win;
        double avg_loss;
        double profit_factor;
        double sharpe_ratio;
        double max_drawdown;
        
        // Timing metrics
        double avg_execution_time_ms;
        double avg_fill_time_ms;
        double avg_slippage;
        
        // Volume metrics
        double total_volume_traded;
        double avg_trade_size;
        double daily_volume;
        
        // Risk metrics
        double var_95;
        double var_99;
        double max_position_size;
        double leverage_ratio;
    };
    
    PerformanceMonitor();
    ~PerformanceMonitor();
    
    // Start/Stop monitoring
    void start_monitoring();
    void stop_monitoring();
    
    // Trade tracking
    void record_trade(const std::string& symbol, 
                     const std::string& side, 
                     int quantity, 
                     double price, 
                     double execution_time_ms);
    
    void record_fill(const std::string& order_id, 
                    int filled_qty, 
                    double fill_price, 
                    double slippage);
    
    void record_pnl_update(double pnl);
    
    // Metrics retrieval
    TradingMetrics get_current_metrics();
    TradingMetrics get_metrics_for_period(std::chrono::system_clock::time_point start,
                                         std::chrono::system_clock::time_point end);
    
    // Real-time monitoring
    void set_metrics_callback(std::function<void(const TradingMetrics&)> callback);
    
    // Risk management
    bool check_risk_limits(const std::string& symbol, 
                          const std::string& side, 
                          int quantity, 
                          double price);
    
    void set_risk_limits(double max_position_size, 
                       double max_daily_loss, 
                       double max_leverage);
    
private:
    std::atomic<bool> monitoring_;
    std::thread monitoring_thread_;
    
    // Data storage
    std::vector<std::pair<std::chrono::system_clock::time_point, TradingMetrics>> metrics_history_;
    std::mutex metrics_mutex_;
    
    // Current metrics
    TradingMetrics current_metrics_;
    std::mutex current_metrics_mutex_;
    
    // Risk limits
    double max_position_size_;
    double max_daily_loss_;
    double max_leverage_;
    std::mutex risk_mutex_;
    
    // Callbacks
    std::function<void(const TradingMetrics&)> metrics_callback_;
    
    // Monitoring methods
    void monitoring_thread_func();
    void calculate_real_time_metrics();
    void update_risk_metrics();
    
    // Calculation helpers
    double calculate_sharpe_ratio();
    double calculate_max_drawdown();
    double calculate_var(double confidence_level);
    double calculate_profit_factor();
    
    // Risk management
    bool check_position_limits(const std::string& symbol, int quantity);
    bool check_daily_loss_limits();
    bool check_leverage_limits();
};

} // namespace HFT
