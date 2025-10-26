#include "market_data_ingestion.h"
#include <iostream>
#include <sstream>
#include <iomanip>

namespace HFT {

PolygonMarketDataIngestion::PolygonMarketDataIngestion(const std::string& api_key)
    : api_key_(api_key), running_(false) {
    // Initialize WebSocket client
    client_.clear_access_channels(websocketpp::log::alevel::all);
    client_.clear_error_channels(websocketpp::log::elevel::all);
    
    client_.init_asio();
    
    // Set handlers
    client_.set_open_handler([this](websocketpp::connection_hdl hdl) {
        on_open(hdl);
    });
    
    client_.set_message_handler([this](websocketpp::connection_hdl hdl, 
                                     websocketpp::config::asio_client::message_ptr msg) {
        on_message(hdl, msg);
    });
    
    client_.set_close_handler([this](websocketpp::connection_hdl hdl) {
        on_close(hdl);
    });
    
    client_.set_fail_handler([this](websocketpp::connection_hdl hdl) {
        on_fail(hdl);
    });
}

PolygonMarketDataIngestion::~PolygonMarketDataIngestion() {
    stop();
}

bool PolygonMarketDataIngestion::start() {
    if (running_) {
        return true;
    }
    
    try {
        // Connect to Polygon WebSocket
        websocketpp::lib::error_code ec;
        auto con = client_.get_connection("wss://socket.polygon.io/stocks", ec);
        
        if (ec) {
            std::cerr << "Failed to create connection: " << ec.message() << std::endl;
            return false;
        }
        
        hdl_ = con->get_handle();
        client_.connect(con);
        
        running_ = true;
        ws_thread_ = std::thread(&PolygonMarketDataIngestion::websocket_thread_func, this);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to start market data ingestion: " << e.what() << std::endl;
        return false;
    }
}

void PolygonMarketDataIngestion::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Close WebSocket connection
    try {
        client_.close(hdl_, websocketpp::close::status::normal, "Shutdown");
    } catch (const std::exception& e) {
        std::cerr << "Error closing WebSocket: " << e.what() << std::endl;
    }
    
    // Wait for thread to finish
    if (ws_thread_.joinable()) {
        ws_thread_.join();
    }
}

void PolygonMarketDataIngestion::subscribe_to_quotes(const std::vector<std::string>& symbols) {
    if (!running_) {
        return;
    }
    
    try {
        nlohmann::json subscribe_msg;
        subscribe_msg["action"] = "subscribe";
        subscribe_msg["params"] = "Q." + symbols[0]; // Start with first symbol
        
        // Add additional symbols
        for (size_t i = 1; i < symbols.size(); ++i) {
            subscribe_msg["params"] += "," + symbols[i];
        }
        
        std::string message = subscribe_msg.dump();
        client_.send(hdl_, message, websocketpp::frame::opcode::text);
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to subscribe to quotes: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::subscribe_to_trades(const std::vector<std::string>& symbols) {
    if (!running_) {
        return;
    }
    
    try {
        nlohmann::json subscribe_msg;
        subscribe_msg["action"] = "subscribe";
        subscribe_msg["params"] = "T." + symbols[0]; // Start with first symbol
        
        // Add additional symbols
        for (size_t i = 1; i < symbols.size(); ++i) {
            subscribe_msg["params"] += "," + symbols[i];
        }
        
        std::string message = subscribe_msg.dump();
        client_.send(hdl_, message, websocketpp::frame::opcode::text);
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to subscribe to trades: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::subscribe_to_orderbook(const std::vector<std::string>& symbols) {
    if (!running_) {
        return;
    }
    
    try {
        nlohmann::json subscribe_msg;
        subscribe_msg["action"] = "subscribe";
        subscribe_msg["params"] = "L." + symbols[0]; // Start with first symbol
        
        // Add additional symbols
        for (size_t i = 1; i < symbols.size(); ++i) {
            subscribe_msg["params"] += "," + symbols[i];
        }
        
        std::string message = subscribe_msg.dump();
        client_.send(hdl_, message, websocketpp::frame::opcode::text);
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to subscribe to orderbook: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::set_market_data_callback(MarketDataCallback callback) {
    market_data_callback_ = callback;
}

void PolygonMarketDataIngestion::set_orderbook_callback(OrderBookCallback callback) {
    orderbook_callback_ = callback;
}

void PolygonMarketDataIngestion::set_trade_callback(TradeCallback callback) {
    trade_callback_ = callback;
}

MarketData PolygonMarketDataIngestion::get_latest_quote(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(quotes_mutex_);
    auto it = latest_quotes_.find(symbol);
    if (it != latest_quotes_.end()) {
        return it->second;
    }
    return MarketData{}; // Return empty MarketData if not found
}

OrderBook PolygonMarketDataIngestion::get_latest_orderbook(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(orderbook_mutex_);
    auto it = latest_orderbooks_.find(symbol);
    if (it != latest_orderbooks_.end()) {
        return it->second;
    }
    return OrderBook{}; // Return empty OrderBook if not found
}

std::vector<Trade> PolygonMarketDataIngestion::get_recent_trades(const std::string& symbol, int count) {
    std::lock_guard<std::mutex> lock(trades_mutex_);
    auto it = recent_trades_.find(symbol);
    if (it == recent_trades_.end()) {
        return {};
    }
    
    std::vector<Trade> result;
    auto& trade_queue = it->second;
    
    // Get the most recent trades
    int collected = 0;
    while (!trade_queue.empty() && collected < count) {
        result.push_back(trade_queue.front());
        trade_queue.pop();
        collected++;
    }
    
    return result;
}

void PolygonMarketDataIngestion::on_open(websocketpp::connection_hdl hdl) {
    std::cout << "Connected to Polygon WebSocket" << std::endl;
    
    // Authenticate with API key
    nlohmann::json auth_msg;
    auth_msg["action"] = "auth";
    auth_msg["params"] = api_key_;
    
    std::string message = auth_msg.dump();
    client_.send(hdl, message, websocketpp::frame::opcode::text);
}

void PolygonMarketDataIngestion::on_message(websocketpp::connection_hdl hdl, 
                                           websocketpp::config::asio_client::message_ptr msg) {
    try {
        std::string message = msg->get_payload();
        process_message(message);
    } catch (const std::exception& e) {
        std::cerr << "Error processing message: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::on_close(websocketpp::connection_hdl hdl) {
    std::cout << "WebSocket connection closed" << std::endl;
    running_ = false;
}

void PolygonMarketDataIngestion::on_fail(websocketpp::connection_hdl hdl) {
    std::cerr << "WebSocket connection failed" << std::endl;
    running_ = false;
}

void PolygonMarketDataIngestion::process_message(const std::string& message) {
    try {
        auto data = nlohmann::json::parse(message);
        
        // Check message type
        if (data.contains("ev")) {
            std::string event_type = data["ev"];
            
            if (event_type == "Q") {
                process_quote_message(data);
            } else if (event_type == "T") {
                process_trade_message(data);
            } else if (event_type == "L") {
                process_orderbook_message(data);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing message: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::process_quote_message(const nlohmann::json& data) {
    try {
        MarketData quote;
        quote.symbol = data.value("sym", "");
        quote.bid_price = data.value("bp", 0.0);
        quote.ask_price = data.value("ap", 0.0);
        quote.bid_size = data.value("bs", 0);
        quote.ask_size = data.value("as", 0);
        quote.last_price = data.value("p", 0.0);
        quote.last_size = data.value("s", 0);
        quote.timestamp = data.value("t", 0ULL);
        quote.exchange = data.value("x", "");
        
        // Update cache
        {
            std::lock_guard<std::mutex> lock(quotes_mutex_);
            latest_quotes_[quote.symbol] = quote;
        }
        
        // Call callback
        if (market_data_callback_) {
            market_data_callback_(quote);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing quote message: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::process_trade_message(const nlohmann::json& data) {
    try {
        Trade trade;
        trade.symbol = data.value("sym", "");
        trade.price = data.value("p", 0.0);
        trade.size = data.value("s", 0);
        trade.timestamp = data.value("t", 0ULL);
        trade.exchange = data.value("x", "");
        trade.is_buy = data.value("c", std::vector<int>{}).empty() ? false : true;
        
        // Update cache
        {
            std::lock_guard<std::mutex> lock(trades_mutex_);
            recent_trades_[trade.symbol].push(trade);
            
            // Keep only recent trades (limit to 1000)
            auto& trade_queue = recent_trades_[trade.symbol];
            while (trade_queue.size() > 1000) {
                trade_queue.pop();
            }
        }
        
        // Call callback
        if (trade_callback_) {
            trade_callback_(trade);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing trade message: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::process_orderbook_message(const nlohmann::json& data) {
    try {
        OrderBook orderbook;
        orderbook.symbol = data.value("sym", "");
        orderbook.timestamp = data.value("t", 0ULL);
        
        // Process bids
        if (data.contains("b")) {
            for (const auto& bid : data["b"]) {
                double price = bid[0].get<double>();
                int size = bid[1].get<int>();
                orderbook.bids.push_back({price, size});
            }
        }
        
        // Process asks
        if (data.contains("a")) {
            for (const auto& ask : data["a"]) {
                double price = ask[0].get<double>();
                int size = ask[1].get<int>();
                orderbook.asks.push_back({price, size});
            }
        }
        
        // Update cache
        {
            std::lock_guard<std::mutex> lock(orderbook_mutex_);
            latest_orderbooks_[orderbook.symbol] = orderbook;
        }
        
        // Call callback
        if (orderbook_callback_) {
            orderbook_callback_(orderbook);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error processing orderbook message: " << e.what() << std::endl;
    }
}

void PolygonMarketDataIngestion::websocket_thread_func() {
    try {
        client_.run();
    } catch (const std::exception& e) {
        std::cerr << "WebSocket thread error: " << e.what() << std::endl;
    }
}

} // namespace HFT
