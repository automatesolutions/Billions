#include "alpaca_websocket_client.h"
#include <iostream>
#include <chrono>
#include <random>
#include <sstream>
#include <iomanip>
#include <openssl/hmac.h>
#include <openssl/sha.h>
#include <base64.h>

namespace hft {

AlpacaWebSocketClient::AlpacaWebSocketClient(const AlpacaConfig& config)
    : config_(config), connected_(false), authenticated_(false) {
    
    client_ = std::make_unique<websocket_client>();
    
    // Set up logging
    client_->set_access_channels(websocketpp::log::alevel::all);
    client_->clear_access_channels(websocketpp::log::alevel::frame_payload);
    
    // Initialize ASIO
    client_->init_asio();
    
    // Set up handlers
    client_->set_open_handler([this](websocketpp::connection_hdl hdl) {
        on_open(hdl);
    });
    
    client_->set_close_handler([this](websocketpp::connection_hdl hdl) {
        on_close(hdl);
    });
    
    client_->set_message_handler([this](websocketpp::connection_hdl hdl, 
                                      websocketpp::config::asio_client::message_ptr msg) {
        on_message(hdl, msg);
    });
    
    client_->set_fail_handler([this](websocketpp::connection_hdl hdl) {
        on_fail(hdl);
    });
}

AlpacaWebSocketClient::~AlpacaWebSocketClient() {
    disconnect();
}

bool AlpacaWebSocketClient::connect() {
    try {
        websocketpp::lib::error_code ec;
        
        std::string url = config_.use_paper_trading ? 
                         config_.paper_base_url : config_.base_url;
        
        auto con = client_->get_connection(url, ec);
        if (ec) {
            std::cerr << "Failed to create connection: " << ec.message() << std::endl;
            return false;
        }
        
        connection_hdl_ = con->get_handle();
        client_->connect(con);
        
        // Start the IO thread
        io_thread_ = std::thread([this]() {
            client_->run();
        });
        
        // Wait for connection
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        return connected_.load();
    } catch (const std::exception& e) {
        std::cerr << "Connection error: " << e.what() << std::endl;
        return false;
    }
}

void AlpacaWebSocketClient::disconnect() {
    if (connected_.load()) {
        websocketpp::lib::error_code ec;
        client_->close(connection_hdl_, websocketpp::close::status::normal, "", ec);
        
        if (io_thread_.joinable()) {
            io_thread_.join();
        }
        
        connected_.store(false);
        authenticated_.store(false);
    }
}

bool AlpacaWebSocketClient::is_connected() const {
    return connected_.load() && authenticated_.load();
}

std::string AlpacaWebSocketClient::submit_order(const OrderRequest& order) {
    if (!is_connected()) {
        throw std::runtime_error("Not connected to Alpaca");
    }
    
    json order_msg = {
        {"action", "order:create"},
        {"data", {
            {"symbol", order.symbol},
            {"side", order.side},
            {"type", order.order_type},
            {"qty", std::to_string(order.quantity)},
            {"time_in_force", order.time_in_force}
        }}
    };
    
    if (order.order_type == "limit" && order.limit_price > 0) {
        order_msg["data"]["limit_price"] = std::to_string(order.limit_price);
    }
    
    if (order.order_type == "stop" && order.stop_price > 0) {
        order_msg["data"]["stop_price"] = std::to_string(order.stop_price);
    }
    
    if (order.order_type == "stop_limit" && order.limit_price > 0 && order.stop_price > 0) {
        order_msg["data"]["limit_price"] = std::to_string(order.limit_price);
        order_msg["data"]["stop_price"] = std::to_string(order.stop_price);
    }
    
    if (!order.client_order_id.empty()) {
        order_msg["data"]["client_order_id"] = order.client_order_id;
    } else {
        order_msg["data"]["client_order_id"] = generate_client_order_id();
    }
    
    send_message(order_msg);
    return order_msg["data"]["client_order_id"];
}

bool AlpacaWebSocketClient::cancel_order(const std::string& order_id) {
    if (!is_connected()) {
        return false;
    }
    
    json cancel_msg = {
        {"action", "order:cancel"},
        {"data", {
            {"order_id", order_id}
        }}
    };
    
    send_message(cancel_msg);
    return true;
}

bool AlpacaWebSocketClient::cancel_all_orders() {
    if (!is_connected()) {
        return false;
    }
    
    json cancel_all_msg = {
        {"action", "order:cancel_all"}
    };
    
    send_message(cancel_all_msg);
    return true;
}

void AlpacaWebSocketClient::set_order_callback(OrderCallback callback) {
    order_callback_ = callback;
}

void AlpacaWebSocketClient::set_fill_callback(FillCallback callback) {
    fill_callback_ = callback;
}

void AlpacaWebSocketClient::set_error_callback(ErrorCallback callback) {
    error_callback_ = callback;
}

void AlpacaWebSocketClient::set_connection_callback(ConnectionCallback callback) {
    connection_callback_ = callback;
}

bool AlpacaWebSocketClient::subscribe_to_trades(const std::vector<std::string>& symbols) {
    if (!is_connected()) {
        return false;
    }
    
    json subscribe_msg = {
        {"action", "subscribe"},
        {"trades", symbols}
    };
    
    send_message(subscribe_msg);
    return true;
}

bool AlpacaWebSocketClient::subscribe_to_quotes(const std::vector<std::string>& symbols) {
    if (!is_connected()) {
        return false;
    }
    
    json subscribe_msg = {
        {"action", "subscribe"},
        {"quotes", symbols}
    };
    
    send_message(subscribe_msg);
    return true;
}

bool AlpacaWebSocketClient::unsubscribe_from_trades(const std::vector<std::string>& symbols) {
    if (!is_connected()) {
        return false;
    }
    
    json unsubscribe_msg = {
        {"action", "unsubscribe"},
        {"trades", symbols}
    };
    
    send_message(unsubscribe_msg);
    return true;
}

bool AlpacaWebSocketClient::unsubscribe_from_quotes(const std::vector<std::string>& symbols) {
    if (!is_connected()) {
        return false;
    }
    
    json unsubscribe_msg = {
        {"action", "unsubscribe"},
        {"quotes", symbols}
    };
    
    send_message(unsubscribe_msg);
    return true;
}

void AlpacaWebSocketClient::on_open(websocketpp::connection_hdl hdl) {
    std::cout << "WebSocket connection opened" << std::endl;
    connected_.store(true);
    
    if (connection_callback_) {
        connection_callback_(true);
    }
    
    // Authenticate immediately after connection
    authenticate();
}

void AlpacaWebSocketClient::on_close(websocketpp::connection_hdl hdl) {
    std::cout << "WebSocket connection closed" << std::endl;
    connected_.store(false);
    authenticated_.store(false);
    
    if (connection_callback_) {
        connection_callback_(false);
    }
}

void AlpacaWebSocketClient::on_message(websocketpp::connection_hdl hdl, 
                                      websocketpp::config::asio_client::message_ptr msg) {
    std::string message = msg->get_payload();
    process_message(message);
}

void AlpacaWebSocketClient::on_fail(websocketpp::connection_hdl hdl) {
    std::cout << "WebSocket connection failed" << std::endl;
    connected_.store(false);
    authenticated_.store(false);
    
    if (connection_callback_) {
        connection_callback_(false);
    }
}

void AlpacaWebSocketClient::authenticate() {
    std::string timestamp = get_current_timestamp();
    
    // Create authentication message
    json auth_msg = {
        {"action", "auth"},
        {"key", config_.api_key},
        {"secret", config_.secret_key}
    };
    
    send_message(auth_msg);
}

void AlpacaWebSocketClient::handle_auth_response(const json& response) {
    if (response.contains("T") && response["T"] == "success") {
        std::cout << "Authentication successful" << std::endl;
        authenticated_.store(true);
    } else {
        std::cerr << "Authentication failed: " << response.dump() << std::endl;
        if (error_callback_) {
            error_callback_("Authentication failed");
        }
    }
}

void AlpacaWebSocketClient::handle_order_update(const json& order_data) {
    OrderResponse order;
    
    order.order_id = order_data.value("i", "");
    order.client_order_id = order_data.value("c", "");
    order.symbol = order_data.value("S", "");
    order.side = order_data.value("s", "");
    order.order_type = order_data.value("ot", "");
    order.quantity = std::stoi(order_data.value("q", "0"));
    order.limit_price = std::stod(order_data.value("lp", "0"));
    order.stop_price = std::stod(order_data.value("sp", "0"));
    order.time_in_force = order_data.value("tif", "");
    order.status = order_data.value("s", "");
    order.created_at = order_data.value("t", "");
    order.updated_at = order_data.value("u", "");
    order.filled_avg_price = std::stod(order_data.value("ap", "0"));
    order.filled_qty = std::stoi(order_data.value("f", "0"));
    order.remaining_qty = std::stoi(order_data.value("r", "0"));
    order.reject_reason = order_data.value("rr", "");
    
    if (order_callback_) {
        order_callback_(order);
    }
}

void AlpacaWebSocketClient::handle_fill_notification(const json& fill_data) {
    FillNotification fill;
    
    fill.order_id = fill_data.value("i", "");
    fill.symbol = fill_data.value("S", "");
    fill.side = fill_data.value("s", "");
    fill.filled_qty = std::stoi(fill_data.value("q", "0"));
    fill.filled_price = std::stod(fill_data.value("p", "0"));
    fill.filled_at = fill_data.value("t", "");
    fill.trade_id = fill_data.value("T", "");
    
    if (fill_callback_) {
        fill_callback_(fill);
    }
}

void AlpacaWebSocketClient::handle_error(const json& error_data) {
    std::string error_msg = error_data.value("msg", "Unknown error");
    std::cerr << "Alpaca error: " << error_msg << std::endl;
    
    if (error_callback_) {
        error_callback_(error_msg);
    }
}

void AlpacaWebSocketClient::process_message(const std::string& message) {
    try {
        json data = json::parse(message);
        
        if (data.contains("T")) {
            std::string msg_type = data["T"];
            
            if (msg_type == "success" || msg_type == "error") {
                handle_auth_response(data);
            } else if (msg_type == "order_update") {
                handle_order_update(data);
            } else if (msg_type == "fill") {
                handle_fill_notification(data);
            } else if (msg_type == "error") {
                handle_error(data);
            }
        }
    } catch (const json::exception& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    }
}

void AlpacaWebSocketClient::send_message(const json& message) {
    try {
        websocketpp::lib::error_code ec;
        client_->send(connection_hdl_, message.dump(), websocketpp::frame::opcode::text, ec);
        
        if (ec) {
            std::cerr << "Send error: " << ec.message() << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Send exception: " << e.what() << std::endl;
    }
}

std::string AlpacaWebSocketClient::generate_client_order_id() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1000, 9999);
    
    return "HFT_" + std::to_string(timestamp) + "_" + std::to_string(dis(gen));
}

std::string AlpacaWebSocketClient::get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    
    return ss.str();
}

} // namespace hft
