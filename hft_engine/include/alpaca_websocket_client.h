#pragma once

#include <string>
#include <memory>
#include <functional>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <nlohmann/json.hpp>

namespace hft {

using json = nlohmann::json;
using websocket_client = websocketpp::client<websocketpp::config::asio_client>;

struct AlpacaConfig {
    std::string api_key;
    std::string secret_key;
    std::string base_url = "wss://stream.data.alpaca.markets/v2/iex";
    std::string paper_base_url = "wss://stream.data.alpaca.markets/v2/iex";
    bool use_paper_trading = true;
};

struct OrderRequest {
    std::string symbol;
    std::string side; // "buy" or "sell"
    std::string order_type; // "limit", "market", "stop", "stop_limit"
    int quantity;
    double limit_price = 0.0;
    double stop_price = 0.0;
    std::string time_in_force = "day"; // "day", "gtc", "ioc", "fok"
    std::string client_order_id;
};

struct OrderResponse {
    std::string order_id;
    std::string client_order_id;
    std::string symbol;
    std::string side;
    std::string order_type;
    int quantity;
    double limit_price = 0.0;
    double stop_price = 0.0;
    std::string time_in_force;
    std::string status; // "new", "partially_filled", "filled", "canceled", "rejected"
    std::string created_at;
    std::string updated_at;
    double filled_avg_price = 0.0;
    int filled_qty = 0;
    int remaining_qty = 0;
    std::string reject_reason;
};

struct FillNotification {
    std::string order_id;
    std::string symbol;
    std::string side;
    int filled_qty;
    double filled_price;
    std::string filled_at;
    std::string trade_id;
};

class AlpacaWebSocketClient {
public:
    using OrderCallback = std::function<void(const OrderResponse&)>;
    using FillCallback = std::function<void(const FillNotification&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    using ConnectionCallback = std::function<void(bool connected)>;

    AlpacaWebSocketClient(const AlpacaConfig& config);
    ~AlpacaWebSocketClient();

    // Connection management
    bool connect();
    void disconnect();
    bool is_connected() const;

    // Order management
    std::string submit_order(const OrderRequest& order);
    bool cancel_order(const std::string& order_id);
    bool cancel_all_orders();
    
    // Order status
    std::vector<OrderResponse> get_open_orders();
    OrderResponse get_order(const std::string& order_id);

    // Callbacks
    void set_order_callback(OrderCallback callback);
    void set_fill_callback(FillCallback callback);
    void set_error_callback(ErrorCallback callback);
    void set_connection_callback(ConnectionCallback callback);

    // Market data subscription
    bool subscribe_to_trades(const std::vector<std::string>& symbols);
    bool subscribe_to_quotes(const std::vector<std::string>& symbols);
    bool unsubscribe_from_trades(const std::vector<std::string>& symbols);
    bool unsubscribe_from_quotes(const std::vector<std::string>& symbols);

private:
    AlpacaConfig config_;
    std::unique_ptr<websocket_client> client_;
    websocketpp::connection_hdl connection_hdl_;
    
    std::atomic<bool> connected_;
    std::atomic<bool> authenticated_;
    
    std::thread io_thread_;
    std::mutex message_queue_mutex_;
    std::queue<std::string> message_queue_;
    
    // Callbacks
    OrderCallback order_callback_;
    FillCallback fill_callback_;
    ErrorCallback error_callback_;
    ConnectionCallback connection_callback_;

    // Message handling
    void on_open(websocketpp::connection_hdl hdl);
    void on_close(websocketpp::connection_hdl hdl);
    void on_message(websocketpp::connection_hdl hdl, 
                   websocketpp::config::asio_client::message_ptr msg);
    void on_fail(websocketpp::connection_hdl hdl);

    // Authentication
    void authenticate();
    void handle_auth_response(const json& response);

    // Order handling
    void handle_order_update(const json& order_data);
    void handle_fill_notification(const json& fill_data);
    void handle_error(const json& error_data);

    // Message processing
    void process_message(const std::string& message);
    void send_message(const json& message);

    // Utility functions
    std::string generate_client_order_id();
    std::string get_current_timestamp();
};

} // namespace hft
