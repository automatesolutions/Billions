#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "hft_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(hft_python_bindings, m) {
    m.doc() = "HFT Trading Engine Python Bindings";
    
    // Enums
    py::enum_<HFT::OrderType>(m, "OrderType")
        .value("MARKET", HFT::OrderType::MARKET)
        .value("LIMIT", HFT::OrderType::LIMIT)
        .value("LIMIT_EDGE", HFT::OrderType::LIMIT_EDGE)
        .value("MARKET_EDGE", HFT::OrderType::MARKET_EDGE)
        .value("TWAP", HFT::OrderType::TWAP)
        .value("TWAP_EDGE", HFT::OrderType::TWAP_EDGE)
        .value("VWAP", HFT::OrderType::VWAP);
    
    py::enum_<HFT::OrderSide>(m, "OrderSide")
        .value("BUY", HFT::OrderSide::BUY)
        .value("SELL", HFT::OrderSide::SELL);
    
    py::enum_<HFT::OrderStatus>(m, "OrderStatus")
        .value("PENDING", HFT::OrderStatus::PENDING)
        .value("SUBMITTED", HFT::OrderStatus::SUBMITTED)
        .value("PARTIALLY_FILLED", HFT::OrderStatus::PARTIALLY_FILLED)
        .value("FILLED", HFT::OrderStatus::FILLED)
        .value("CANCELLED", HFT::OrderStatus::CANCELLED)
        .value("REJECTED", HFT::OrderStatus::REJECTED);
    
    // Data structures
    py::class_<HFT::MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("symbol", &HFT::MarketData::symbol)
        .def_readwrite("bid_price", &HFT::MarketData::bid_price)
        .def_readwrite("ask_price", &HFT::MarketData::ask_price)
        .def_readwrite("bid_size", &HFT::MarketData::bid_size)
        .def_readwrite("ask_size", &HFT::MarketData::ask_size)
        .def_readwrite("last_price", &HFT::MarketData::last_price)
        .def_readwrite("last_size", &HFT::MarketData::last_size)
        .def_readwrite("timestamp", &HFT::MarketData::timestamp)
        .def_readwrite("exchange", &HFT::MarketData::exchange);
    
    py::class_<HFT::OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def_readwrite("symbol", &HFT::OrderBook::symbol)
        .def_readwrite("bids", &HFT::OrderBook::bids)
        .def_readwrite("asks", &HFT::OrderBook::asks)
        .def_readwrite("timestamp", &HFT::OrderBook::timestamp);
    
    py::class_<HFT::Trade>(m, "Trade")
        .def(py::init<>())
        .def_readwrite("symbol", &HFT::Trade::symbol)
        .def_readwrite("price", &HFT::Trade::price)
        .def_readwrite("size", &HFT::Trade::size)
        .def_readwrite("timestamp", &HFT::Trade::timestamp)
        .def_readwrite("exchange", &HFT::Trade::exchange)
        .def_readwrite("is_buy", &HFT::Trade::is_buy);
    
    py::class_<HFT::Signal>(m, "Signal")
        .def(py::init<>())
        .def_readwrite("symbol", &HFT::Signal::symbol)
        .def_readwrite("signal_strength", &HFT::Signal::signal_strength)
        .def_readwrite("confidence", &HFT::Signal::confidence)
        .def_readwrite("timestamp", &HFT::Signal::timestamp)
        .def_readwrite("signal_type", &HFT::Signal::signal_type)
        .def_readwrite("parameters", &HFT::Signal::parameters);
    
    py::class_<HFT::Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("order_id", &HFT::Order::order_id)
        .def_readwrite("symbol", &HFT::Order::symbol)
        .def_readwrite("type", &HFT::Order::type)
        .def_readwrite("side", &HFT::Order::side)
        .def_readwrite("quantity", &HFT::Order::quantity)
        .def_readwrite("price", &HFT::Order::price)
        .def_readwrite("stop_price", &HFT::Order::stop_price)
        .def_readwrite("duration", &HFT::Order::duration)
        .def_readwrite("interval", &HFT::Order::interval)
        .def_readwrite("status", &HFT::Order::status)
        .def_readwrite("filled_quantity", &HFT::Order::filled_quantity)
        .def_readwrite("avg_fill_price", &HFT::Order::avg_fill_price)
        .def_readwrite("created_at", &HFT::Order::created_at)
        .def_readwrite("updated_at", &HFT::Order::updated_at)
        .def_readwrite("edge_threshold", &HFT::Order::edge_threshold)
        .def_readwrite("volatility_factor", &HFT::Order::volatility_factor)
        .def_readwrite("liquidity_factor", &HFT::Order::liquidity_factor)
        .def_readwrite("schedule", &HFT::Order::schedule)
        .def_readwrite("target_volume_weight", &HFT::Order::target_volume_weight);
    
    py::class_<HFT::AlpacaCredentials>(m, "AlpacaCredentials")
        .def(py::init<>())
        .def_readwrite("api_key", &HFT::AlpacaCredentials::api_key)
        .def_readwrite("secret_key", &HFT::AlpacaCredentials::secret_key)
        .def_readwrite("base_url", &HFT::AlpacaCredentials::base_url)
        .def_readwrite("paper_trading", &HFT::AlpacaCredentials::paper_trading);
    
    py::class_<HFT::PerformanceMonitor::TradingMetrics>(m, "TradingMetrics")
        .def(py::init<>())
        .def_readwrite("total_trades", &HFT::PerformanceMonitor::TradingMetrics::total_trades)
        .def_readwrite("successful_trades", &HFT::PerformanceMonitor::TradingMetrics::successful_trades)
        .def_readwrite("failed_trades", &HFT::PerformanceMonitor::TradingMetrics::failed_trades)
        .def_readwrite("total_pnl", &HFT::PerformanceMonitor::TradingMetrics::total_pnl)
        .def_readwrite("realized_pnl", &HFT::PerformanceMonitor::TradingMetrics::realized_pnl)
        .def_readwrite("unrealized_pnl", &HFT::PerformanceMonitor::TradingMetrics::unrealized_pnl)
        .def_readwrite("win_rate", &HFT::PerformanceMonitor::TradingMetrics::win_rate)
        .def_readwrite("avg_win", &HFT::PerformanceMonitor::TradingMetrics::avg_win)
        .def_readwrite("avg_loss", &HFT::PerformanceMonitor::TradingMetrics::avg_loss)
        .def_readwrite("profit_factor", &HFT::PerformanceMonitor::TradingMetrics::profit_factor)
        .def_readwrite("sharpe_ratio", &HFT::PerformanceMonitor::TradingMetrics::sharpe_ratio)
        .def_readwrite("max_drawdown", &HFT::PerformanceMonitor::TradingMetrics::max_drawdown)
        .def_readwrite("avg_execution_time_ms", &HFT::PerformanceMonitor::TradingMetrics::avg_execution_time_ms)
        .def_readwrite("avg_fill_time_ms", &HFT::PerformanceMonitor::TradingMetrics::avg_fill_time_ms)
        .def_readwrite("avg_slippage", &HFT::PerformanceMonitor::TradingMetrics::avg_slippage)
        .def_readwrite("total_volume_traded", &HFT::PerformanceMonitor::TradingMetrics::total_volume_traded)
        .def_readwrite("avg_trade_size", &HFT::PerformanceMonitor::TradingMetrics::avg_trade_size)
        .def_readwrite("daily_volume", &HFT::PerformanceMonitor::TradingMetrics::daily_volume)
        .def_readwrite("var_95", &HFT::PerformanceMonitor::TradingMetrics::var_95)
        .def_readwrite("var_99", &HFT::PerformanceMonitor::TradingMetrics::var_99)
        .def_readwrite("max_position_size", &HFT::PerformanceMonitor::TradingMetrics::max_position_size)
        .def_readwrite("leverage_ratio", &HFT::PerformanceMonitor::TradingMetrics::leverage_ratio);
    
    // Main HFT Engine class
    py::class_<HFT::HFTTradingEngine>(m, "HFTTradingEngine")
        .def(py::init<const std::string&, const HFT::AlpacaCredentials&>())
        .def("initialize", &HFT::HFTTradingEngine::initialize)
        .def("start", &HFT::HFTTradingEngine::start)
        .def("stop", &HFT::HFTTradingEngine::stop)
        .def("shutdown", &HFT::HFTTradingEngine::shutdown)
        .def("set_trading_symbols", &HFT::HFTTradingEngine::set_trading_symbols)
        .def("set_edge_threshold", &HFT::HFTTradingEngine::set_edge_threshold)
        .def("set_max_position_size", &HFT::HFTTradingEngine::set_max_position_size)
        .def("set_risk_limits", &HFT::HFTTradingEngine::set_risk_limits)
        .def("enable_strategy", &HFT::HFTTradingEngine::enable_strategy)
        .def("disable_strategy", &HFT::HFTTradingEngine::disable_strategy)
        .def("set_strategy_parameters", &HFT::HFTTradingEngine::set_strategy_parameters)
        .def("submit_market_order", &HFT::HFTTradingEngine::submit_market_order)
        .def("submit_limit_order", &HFT::HFTTradingEngine::submit_limit_order)
        .def("submit_twap_order", &HFT::HFTTradingEngine::submit_twap_order)
        .def("submit_vwap_order", &HFT::HFTTradingEngine::submit_vwap_order)
        .def("is_running", &HFT::HFTTradingEngine::is_running)
        .def("get_performance_metrics", &HFT::HFTTradingEngine::get_performance_metrics)
        .def("get_execution_metrics", &HFT::HFTTradingEngine::get_execution_metrics)
        .def("set_trade_callback", [](HFT::HFTTradingEngine& engine, py::function callback) {
            engine.set_trade_callback([callback](const std::string& symbol, const std::string& side, 
                                                int quantity, double price) {
                callback(symbol, side, quantity, price);
            });
        })
        .def("set_error_callback", [](HFT::HFTTradingEngine& engine, py::function callback) {
            engine.set_error_callback([callback](const std::string& error) {
                callback(error);
            });
        })
        .def("set_performance_callback", [](HFT::HFTTradingEngine& engine, py::function callback) {
            engine.set_performance_callback([callback](const HFT::PerformanceMonitor::TradingMetrics& metrics) {
                callback(metrics);
            });
        });
    
    // Factory function
    m.def("create_hft_engine", &HFT::create_hft_engine, 
          "Create HFT Trading Engine instance");
}
