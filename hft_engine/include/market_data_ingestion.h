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
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>

namespace HFT {

struct MarketData {
    std::string symbol;
    double bid_price;
    double ask_price;
    int bid_size;
    int ask_size;
    double last_price;
    int last_size;
    uint64_t timestamp;
    std::string exchange;
};

struct OrderBook {
    std::string symbol;
    std::vector<std::pair<double, int>> bids;  // price, size
    std::vector<std::pair<double, int>> asks;  // price, size
    uint64_t timestamp;
};

struct Trade {
    std::string symbol;
    double price;
    int size;
    uint64_t timestamp;
    std::string exchange;
    bool is_buy;
};

class PolygonMarketDataIngestion {
public:
    using MarketDataCallback = std::function<void(const MarketData&)>;
    using OrderBookCallback = std::function<void(const OrderBook&)>;
    using TradeCallback = std::function<void(const Trade&)>;

    PolygonMarketDataIngestion(const std::string& api_key);
    ~PolygonMarketDataIngestion();

    // Start/Stop data ingestion
    bool start();
    void stop();

    // Subscribe to symbols
    void subscribe_to_quotes(const std::vector<std::string>& symbols);
    void subscribe_to_trades(const std::vector<std::string>& symbols);
    void subscribe_to_orderbook(const std::vector<std::string>& symbols);

    // Callbacks
    void set_market_data_callback(MarketDataCallback callback);
    void set_orderbook_callback(OrderBookCallback callback);
    void set_trade_callback(TradeCallback callback);

    // Ultra-fast data access
    MarketData get_latest_quote(const std::string& symbol);
    OrderBook get_latest_orderbook(const std::string& symbol);
    std::vector<Trade> get_recent_trades(const std::string& symbol, int count = 10);

private:
    std::string api_key_;
    std::atomic<bool> running_;
    std::thread ws_thread_;
    
    // WebSocket client
    websocketpp::client<websocketpp::config::asio> client_;
    websocketpp::connection_hdl hdl_;
    
    // Data storage for ultra-fast access
    std::unordered_map<std::string, MarketData> latest_quotes_;
    std::unordered_map<std::string, OrderBook> latest_orderbooks_;
    std::unordered_map<std::string, std::queue<Trade>> recent_trades_;
    
    // Thread-safe access
    std::mutex quotes_mutex_;
    std::mutex orderbook_mutex_;
    std::mutex trades_mutex_;
    
    // Callbacks
    MarketDataCallback market_data_callback_;
    OrderBookCallback orderbook_callback_;
    TradeCallback trade_callback_;
    
    // WebSocket handlers
    void on_open(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, 
                   websocketpp::config::asio_client::message_ptr msg);
    void on_close(websocketpp::connection_hdl hdl);
    void on_fail(websocketpp::connection_hdl hdl);
    
    // Message processing
    void process_message(const std::string& message);
    void process_quote_message(const nlohmann::json& data);
    void process_trade_message(const nlohmann::json& data);
    void process_orderbook_message(const nlohmann::json& data);
    
    // WebSocket thread function
    void websocket_thread_func();
};

} // namespace HFT
