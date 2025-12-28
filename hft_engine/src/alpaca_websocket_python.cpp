#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "alpaca_websocket_client.h"

namespace py = pybind11;

class PyAlpacaWebSocketClient {
public:
    PyAlpacaWebSocketClient(const std::string& api_key, 
                           const std::string& secret_key,
                           bool use_paper_trading = true) {
        hft::AlpacaConfig config;
        config.api_key = api_key;
        config.secret_key = secret_key;
        config.use_paper_trading = use_paper_trading;
        
        client_ = std::make_unique<hft::AlpacaWebSocketClient>(config);
        
        // Set up callbacks
        client_->set_order_callback([this](const hft::OrderResponse& order) {
            if (order_callback_) {
                py::gil_scoped_acquire acquire;
                order_callback_(order);
            }
        });
        
        client_->set_fill_callback([this](const hft::FillNotification& fill) {
            if (fill_callback_) {
                py::gil_scoped_acquire acquire;
                fill_callback_(fill);
            }
        });
        
        client_->set_error_callback([this](const std::string& error) {
            if (error_callback_) {
                py::gil_scoped_acquire acquire;
                error_callback_(error);
            }
        });
        
        client_->set_connection_callback([this](bool connected) {
            if (connection_callback_) {
                py::gil_scoped_acquire acquire;
                connection_callback_(connected);
            }
        });
    }
    
    bool connect() {
        return client_->connect();
    }
    
    void disconnect() {
        client_->disconnect();
    }
    
    bool is_connected() const {
        return client_->is_connected();
    }
    
    std::string submit_order(const std::string& symbol,
                           const std::string& side,
                           const std::string& order_type,
                           int quantity,
                           double limit_price = 0.0,
                           double stop_price = 0.0,
                           const std::string& time_in_force = "day",
                           const std::string& client_order_id = "") {
        hft::OrderRequest order;
        order.symbol = symbol;
        order.side = side;
        order.order_type = order_type;
        order.quantity = quantity;
        order.limit_price = limit_price;
        order.stop_price = stop_price;
        order.time_in_force = time_in_force;
        order.client_order_id = client_order_id;
        
        return client_->submit_order(order);
    }
    
    bool cancel_order(const std::string& order_id) {
        return client_->cancel_order(order_id);
    }
    
    bool cancel_all_orders() {
        return client_->cancel_all_orders();
    }
    
    bool subscribe_to_trades(const std::vector<std::string>& symbols) {
        return client_->subscribe_to_trades(symbols);
    }
    
    bool subscribe_to_quotes(const std::vector<std::string>& symbols) {
        return client_->subscribe_to_quotes(symbols);
    }
    
    bool unsubscribe_from_trades(const std::vector<std::string>& symbols) {
        return client_->unsubscribe_from_trades(symbols);
    }
    
    bool unsubscribe_from_quotes(const std::vector<std::string>& symbols) {
        return client_->unsubscribe_from_quotes(symbols);
    }
    
    // Callback setters
    void set_order_callback(py::function callback) {
        order_callback_ = callback;
    }
    
    void set_fill_callback(py::function callback) {
        fill_callback_ = callback;
    }
    
    void set_error_callback(py::function callback) {
        error_callback_ = callback;
    }
    
    void set_connection_callback(py::function callback) {
        connection_callback_ = callback;
    }

private:
    std::unique_ptr<hft::AlpacaWebSocketClient> client_;
    py::function order_callback_;
    py::function fill_callback_;
    py::function error_callback_;
    py::function connection_callback_;
};

PYBIND11_MODULE(alpaca_websocket, m) {
    m.doc() = "Alpaca WebSocket client for HFT trading";
    
    py::class_<PyAlpacaWebSocketClient>(m, "AlpacaWebSocketClient")
        .def(py::init<const std::string&, const std::string&, bool>(),
             py::arg("api_key"), py::arg("secret_key"), py::arg("use_paper_trading") = true)
        .def("connect", &PyAlpacaWebSocketClient::connect)
        .def("disconnect", &PyAlpacaWebSocketClient::disconnect)
        .def("is_connected", &PyAlpacaWebSocketClient::is_connected)
        .def("submit_order", &PyAlpacaWebSocketClient::submit_order,
             py::arg("symbol"), py::arg("side"), py::arg("order_type"), py::arg("quantity"),
             py::arg("limit_price") = 0.0, py::arg("stop_price") = 0.0,
             py::arg("time_in_force") = "day", py::arg("client_order_id") = "")
        .def("cancel_order", &PyAlpacaWebSocketClient::cancel_order)
        .def("cancel_all_orders", &PyAlpacaWebSocketClient::cancel_all_orders)
        .def("subscribe_to_trades", &PyAlpacaWebSocketClient::subscribe_to_trades)
        .def("subscribe_to_quotes", &PyAlpacaWebSocketClient::subscribe_to_quotes)
        .def("unsubscribe_from_trades", &PyAlpacaWebSocketClient::unsubscribe_from_trades)
        .def("unsubscribe_from_quotes", &PyAlpacaWebSocketClient::unsubscribe_from_quotes)
        .def("set_order_callback", &PyAlpacaWebSocketClient::set_order_callback)
        .def("set_fill_callback", &PyAlpacaWebSocketClient::set_fill_callback)
        .def("set_error_callback", &PyAlpacaWebSocketClient::set_error_callback)
        .def("set_connection_callback", &PyAlpacaWebSocketClient::set_connection_callback);
}
