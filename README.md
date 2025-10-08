<div align="center">

# 💰 BILLIONS ML PREDICTION SYSTEM

<img src="funda/assets/logo.png" alt="Billions Logo" width="200"/>

### *Advanced Stock Market Prediction & Outlier Detection Platform*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Dash](https://img.shields.io/badge/Dash-Plotly-purple.svg)](https://dash.plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

<img src="funda/assets/nanakorobi_yaoki.png" alt="七転び八起き" width="150"/>

*七転び八起き - Fall seven times, stand up eight*

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 🎯 Overview

**BILLIONS** is a sophisticated machine learning platform designed for stock market prediction and outlier detection. It combines advanced LSTM neural networks, comprehensive technical analysis, and real-time data processing to provide actionable trading insights across multiple timeframes.

### Why BILLIONS?

- 🧠 **Advanced ML Models**: LSTM-based predictions with enhanced feature engineering
- 📊 **Multi-Strategy Analysis**: Scalp, Swing, and Long-term trading strategies
- 🎯 **Outlier Detection**: Identify high-potential stocks before the market
- 📈 **Real-time Dashboard**: Interactive Dash/Plotly visualization
- 🔄 **Continuous Learning**: Automated data refresh and model updates
- 💾 **Persistent Storage**: SQLite database for performance tracking

---

## ✨ Features

### 🤖 Machine Learning & Predictions

- **LSTM Neural Networks**: Multi-layer LSTM architecture for time-series prediction
- **Enhanced Feature Engineering**: 50+ technical indicators and custom features
- **Ensemble Predictions**: Combine multiple models for robust forecasts
- **30-Day Forecasting**: Extended prediction horizons with confidence scoring
- **Institutional Flow Analysis**: Track smart money movements

### 📊 Technical Analysis

- **Advanced Indicators**: RSI, MACD, Bollinger Bands, Stochastic, ADX, and more
- **Volume Analysis**: Institutional flow, volume patterns, and accumulation/distribution
- **Momentum Indicators**: Rate of change, momentum oscillators, trend strength
- **Volatility Metrics**: ATR, historical volatility, Keltner channels
- **Sector Correlation**: Multi-sector comparative analysis with SPY and sector ETFs

### 🎯 Outlier Detection Engine

Three distinct trading strategies with customizable parameters:

| Strategy | Timeframe | Period | Analysis Window | Min Market Cap |
|----------|-----------|--------|-----------------|----------------|
| **Scalp** | 1 minute | 1 week | 21 days | $1B |
| **Swing** | 3 months | 1 month | 63 days | $2B |
| **Long-term** | 1 year | 6 months | 252 days | $10B |

### 🖥️ Interactive Dashboard

- **Real-time Charts**: Candlestick, volume, and indicator overlays
- **Prediction Visualization**: LSTM forecasts with confidence intervals
- **Performance Metrics**: Win rate, accuracy, Sharpe ratio, max drawdown
- **Outlier Explorer**: Interactive scatter plots with Z-score analysis
- **Multi-ticker Comparison**: Side-by-side analysis of multiple stocks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BILLIONS ML PREDICTION SYSTEM                │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  USER INTERFACE  │    │    ML MODELS     │    │   DATA LAYER     │
│                  │    │                  │    │                  │
│   SPS.py (Dash)  │◄──►│  LSTM Training   │◄──►│  SQLite DB       │
│   Interactive    │    │  Prediction      │    │  Performance     │
│   Dashboard      │    │  Ensemble        │    │  Metrics         │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                        │
         └───────────────┬───────┴────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌──────────────────┐            ┌──────────────────┐
│ FEATURE ENGINE   │            │ OUTLIER ENGINE   │
│                  │            │                  │
│ • Technical      │            │ • Z-Score        │
│ • Fundamental    │            │ • Multi-Strategy │
│ • Sentiment      │            │ • Real-time      │
│ • Sector         │            │ • Auto-refresh   │
└──────────────────┘            └──────────────────┘
```

### Core Components

```
billions/
├── 📱 funda/                      # Main application
│   ├── SPS.py                     # Dashboard & prediction system
│   ├── train_lstm_model.py        # LSTM model training
│   ├── enhanced_features.py       # Feature engineering
│   ├── outlier_engine.py          # Outlier detection logic
│   ├── refresh_outliers.py        # Background refresh thread
│   ├── fine_tuning_strategy.py    # Strategy optimization
│   └── model_diagnostics.py       # Model analysis tools
│
├── 💾 db/                         # Database layer
│   ├── core.py                    # SQLAlchemy setup
│   ├── models.py                  # Database models
│   └── __init__.py
│
├── 🎯 outlier/                    # Strategy modules
│   ├── Outlier_Nasdaq_Scalp.py
│   ├── Outlier_Nasdaq_Swing.py
│   └── Outlier_Nasdaq_Longterm.py
│
├── 📊 Data Storage
│   ├── funda/cache/               # Historical price data
│   ├── funda/model/               # Trained LSTM models
│   ├── outlier/cache/             # Sector ETF data
│   └── billions.db                # Performance metrics
│
└── 🎨 Assets
    └── funda/assets/              # Logos, fonts, UI assets
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- Alpha Vantage API key (free at [alphavantage.co](https://www.alphavantage.co/))
- FRED API key (optional, for economic data)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Billions.git
cd Billions
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file in the root directory
touch .env

# Add your API keys
echo "ALPHA_VANTAGE_API_KEY=your_api_key_here" >> .env
echo "FRED_API_KEY=your_fred_key_here" >> .env  # Optional
```

5. **Initialize database**
```bash
python -c "from db.core import engine, Base; from db.models import PerfMetric; Base.metadata.create_all(bind=engine)"
```

6. **Run the application**
```bash
cd funda
python SPS.py
```

7. **Open your browser**
Navigate to `http://127.0.0.1:8050/`

---

## 📖 Usage

### Running Predictions

1. **Launch the Dashboard**
```bash
cd funda
python SPS.py
```

2. **Enter a Ticker Symbol**
   - Type any stock ticker (e.g., TSLA, NVDA, AAPL)
   - Click "🚀 Run Prediction"

3. **Explore Results**
   - View LSTM predictions
   - Analyze technical indicators
   - Check confidence scores
   - Review historical performance

### Training Custom Models

```bash
cd funda
python train_lstm_model.py
```

This will:
- Fetch multi-ticker data from Yahoo Finance
- Apply enhanced feature engineering
- Train LSTM model with validation
- Save model to `funda/model/lstm_daily_model.pt`

### Running Outlier Detection

```python
from funda.outlier_engine import run_outlier_strategy

# Run specific strategy
run_outlier_strategy("scalp")    # For day trading
run_outlier_strategy("swing")    # For swing trading  
run_outlier_strategy("longterm") # For position trading
```

### Refreshing Data

The system includes automatic background refresh, or manually:

```python
from funda.refresh_outliers import start_refresh_thread

# Start background refresh thread
start_refresh_thread()
```

---

## 🧪 Example Predictions

### LSTM Prediction Output

```
📊 TESLA (TSLA) - 30-Day Forecast
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current Price: $242.50
Predicted (Day 1): $245.30 (+1.15%)
Predicted (Day 7): $251.20 (+3.59%)
Predicted (Day 30): $268.80 (+10.86%)

Confidence Score: 78.5%
Trend: BULLISH 📈
Risk Level: MODERATE
```

### Outlier Detection Results

```
🎯 Top 5 Outliers - Swing Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NVTS - Z-Score: 3.24 | Performance: +45.2% (63d)
2. RGTI - Z-Score: 2.89 | Performance: +38.7% (63d)
3. SMMT - Z-Score: 2.71 | Performance: +34.1% (63d)
4. RKLB - Z-Score: 2.45 | Performance: +29.8% (63d)
5. MSTR - Z-Score: 2.38 | Performance: +28.3% (63d)
```

---

## 🔧 Configuration

### Strategy Parameters

Edit `funda/outlier_engine.py`:

```python
STRATEGIES = {
    "scalp":   ("1m", "1w", 21, 5, 1e9),      # (period, window, days, lookback, min_market_cap)
    "swing":   ("3m", "1m", 63, 21, 2e9),
    "longterm":("1y", "6m", 252, 126, 10e9),
}
```

### LSTM Hyperparameters

Modify in `funda/train_lstm_model.py`:

```python
# Model architecture
hidden_layer_size = 100
num_layers = 2
dropout = 0.2

# Training parameters
batch_size = 32
num_epochs = 100
learning_rate = 0.001
```

---

## 📊 Technical Indicators

The system computes 50+ technical indicators including:

### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- Rate of Change (ROC)
- Momentum

### Trend Indicators
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- ADX (Average Directional Index)
- Parabolic SAR
- Ichimoku Cloud

### Volatility Indicators
- Bollinger Bands
- ATR (Average True Range)
- Keltner Channels
- Standard Deviation
- Historical Volatility

### Volume Indicators
- OBV (On-Balance Volume)
- Volume SMA/EMA
- Volume Rate of Change
- Accumulation/Distribution
- Institutional Flow Score

---

## 🎨 Dashboard Features

### Main Dashboard Sections

1. **Prediction Panel**
   - 30-day LSTM forecast
   - Confidence intervals
   - Ensemble predictions
   - Risk assessment

2. **Technical Analysis**
   - Interactive candlestick charts
   - Indicator overlays
   - Volume analysis
   - Support/resistance levels

3. **Outlier Explorer**
   - Multi-strategy scatter plots
   - Z-score heatmaps
   - Performance metrics
   - Real-time updates

4. **Performance Tracker**
   - Historical accuracy
   - Win/loss ratios
   - Sharpe ratio
   - Maximum drawdown
   - Cumulative returns

---

## 🗄️ Database Schema

```sql
CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy VARCHAR(16),      -- scalp, swing, longterm
    symbol VARCHAR(10),        -- Stock ticker
    metric_x NUMERIC,          -- Performance metric
    metric_y NUMERIC,          -- Comparison metric
    z_x NUMERIC,              -- Z-score X
    z_y NUMERIC,              -- Z-score Y
    is_outlier BOOLEAN,       -- Outlier flag
    inserted TIMESTAMP        -- Creation timestamp
);
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Collection
```python
# Multi-source data fetching
├── Yahoo Finance (OHLCV data)
├── Alpha Vantage (Fundamentals)
├── FRED API (Economic indicators)
└── Sector ETFs (Market correlation)
```

### 2. Feature Engineering
```python
# Enhanced feature pipeline
├── Technical Indicators (50+)
├── Price Transformations
├── Volume Analysis
├── Momentum Metrics
├── Volatility Measures
└── Sector Correlations
```

### 3. Model Training
```python
# LSTM Architecture
Input Layer → LSTM Layer(100) → Dropout(0.2) 
           → LSTM Layer(100) → Dropout(0.2)
           → Dense Layer → Output
```

### 4. Prediction & Evaluation
```python
# Multi-horizon forecasting
├── 1-day ahead
├── 7-day ahead
├── 30-day ahead
└── Confidence scoring
```

---

## 🔬 Performance Metrics

The system tracks comprehensive performance metrics:

- **Accuracy**: Directional prediction accuracy
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable predictions
- **Alpha**: Excess returns vs. benchmark
- **Beta**: Market correlation

---

## 🛠️ Development

### Project Structure Philosophy

Each module follows the **Single Responsibility Principle**:

- `SPS.py`: Dashboard orchestration
- `enhanced_features.py`: Feature engineering only
- `outlier_engine.py`: Outlier detection logic
- `train_lstm_model.py`: Model training pipeline
- `db/`: Data persistence layer

### Adding New Features

1. **New Technical Indicator**
```python
# In enhanced_features.py
def compute_custom_indicator(df):
    """Your custom indicator logic"""
    return df
```

2. **New Trading Strategy**
```python
# In outlier_engine.py
STRATEGIES["custom"] = ("period", "window", days, lookback, min_cap)
```

3. **New Prediction Model**
```python
# In train_lstm_model.py
class CustomModel(nn.Module):
    """Your custom model architecture"""
    pass
```

---

## 📚 Documentation

For detailed documentation, see:

- [SYSTEM_FLOWCHART.md](SYSTEM_FLOWCHART.md) - Complete system architecture
- [Database Documentation](db/README.md) - Database schema and operations
- [API Documentation](docs/API.md) - Function references (coming soon)

---

## 🐛 Troubleshooting

### Common Issues

**1. API Rate Limits**
```
Solution: The system implements automatic rate limiting and caching.
Default cache duration: 24 hours for daily data.
```

**2. Missing Dependencies**
```bash
pip install --upgrade -r requirements.txt
```

**3. Database Lock Errors**
```python
# Increase timeout in db/core.py
engine = create_engine('sqlite:///billions.db', 
                       connect_args={'timeout': 30})
```

**4. CUDA/PyTorch Issues**
```bash
# CPU-only installation
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- **NOT FINANCIAL ADVICE**: This tool does not provide financial, investment, or trading advice
- **USE AT YOUR OWN RISK**: Past performance does not guarantee future results
- **NO WARRANTIES**: The software is provided "as is" without warranties of any kind
- **LOSSES**: You may lose money trading stocks - only invest what you can afford to lose
- **DO YOUR RESEARCH**: Always conduct your own research before making investment decisions
- **CONSULT PROFESSIONALS**: Speak with a licensed financial advisor for personalized advice

The developers and contributors are not responsible for any financial losses incurred from using this software.

---

## 🙏 Acknowledgments

- **Yahoo Finance** - Historical stock data
- **Alpha Vantage** - Fundamental data and NASDAQ listings
- **FRED** - Economic indicators
- **PyTorch** - Deep learning framework
- **Plotly/Dash** - Interactive visualization
- **scikit-learn** - Machine learning utilities

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/Billions/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Billions/discussions)
- **Email**: kumpooniapp@gmail.com

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

<div align="center">

### 💎 Built with passion for the markets

**七転び八起き**

*Made with ❤️ by traders, for traders*

[Back to Top](#-billions-ml-prediction-system)

</div>

