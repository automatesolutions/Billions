# 🚀 Quick Start Guide

Get BILLIONS up and running in 5 minutes!

## Prerequisites

- Python 3.8+ installed
- Alpha Vantage API key ([Get it free here](https://www.alphavantage.co/support/#api-key))

## Installation Steps

### 1️⃣ Clone and Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Billions.git
cd Billions

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure API Keys

Create a `.env` file in the project root:

```bash
# Windows (PowerShell)
echo "ALPHA_VANTAGE_API_KEY=your_key_here" > .env

# Linux/Mac
echo "ALPHA_VANTAGE_API_KEY=your_key_here" > .env
```

Replace `your_key_here` with your actual Alpha Vantage API key.

### 3️⃣ Initialize Database

```bash
python -c "from db.core import engine, Base; from db.models import PerfMetric; Base.metadata.create_all(bind=engine)"
```

### 4️⃣ Launch the Application

```bash
cd funda
python SPS.py
```

### 5️⃣ Open Your Browser

Navigate to: **http://127.0.0.1:8050/**

## 🎯 First Prediction

1. **Enter a ticker symbol** (e.g., `TSLA`, `NVDA`, `AAPL`)
2. **Click "🚀 Run Prediction"**
3. **Wait 10-30 seconds** for data fetching and analysis
4. **Explore the results!**

## 📊 Understanding the Dashboard

### Main Sections

1. **Top Panel**: Input ticker and run predictions
2. **Price Chart**: Candlestick chart with technical indicators
3. **Predictions Table**: LSTM forecasts for next 30 days
4. **Technical Analysis**: RSI, MACD, Bollinger Bands
5. **Outlier Detection**: High-potential stock identification

### Key Metrics

- **Current Price**: Latest closing price
- **Predicted Price**: LSTM model forecast
- **Confidence Score**: Model confidence (0-100%)
- **Trend**: Direction (Bullish/Bearish/Neutral)
- **Z-Score**: Statistical outlier measurement

## 🎓 Common Use Cases

### Case 1: Daily Trading Signals

```bash
# Launch the app
python funda/SPS.py

# In the dashboard:
1. Enter ticker (e.g., TSLA)
2. Check technical indicators (RSI, MACD)
3. Review LSTM prediction for tomorrow
4. Use confidence score to gauge reliability
```

### Case 2: Finding Outlier Stocks

```python
# Run outlier detection
from funda.outlier_engine import run_outlier_strategy

# For swing trading (3-month window)
run_outlier_strategy("swing")

# Check results in dashboard under "Outlier Explorer"
```

### Case 3: Training Custom Models

```bash
# Train on latest data
cd funda
python train_lstm_model.py

# Model will be saved to: funda/model/lstm_daily_model.pt
```

## 🔧 Troubleshooting

### Issue: "No module named 'dash'"

```bash
pip install --upgrade -r requirements.txt
```

### Issue: "API key not found"

Check your `.env` file exists in the root directory and contains:
```
ALPHA_VANTAGE_API_KEY=your_actual_key
```

### Issue: "Database locked"

Close any other instances of the application and try again.

### Issue: "CUDA not available"

This is normal if you don't have an NVIDIA GPU. The system will use CPU automatically.

## 📈 Example Workflows

### Morning Routine: Check Top Movers

```python
# Run outlier detection for scalp strategy
from funda.outlier_engine import run_outlier_strategy
run_outlier_strategy("scalp")

# View results in dashboard
# Launch: python funda/SPS.py
```

### Weekly Analysis: Long-term Positions

```python
# Run long-term outlier detection
from funda.outlier_engine import run_outlier_strategy
run_outlier_strategy("longterm")

# Analyze top 5 outliers in dashboard
```

### Custom Analysis: Specific Stock

1. Open dashboard: `python funda/SPS.py`
2. Enter ticker: `NVDA`
3. Click "Run Prediction"
4. Review:
   - 30-day price forecast
   - Technical indicators
   - Confidence scores
   - Risk assessment

## 🎨 Customization

### Change Prediction Horizon

Edit `funda/SPS.py`:
```python
# Find this line
prediction_days = 30  # Change to desired number of days
```

### Adjust LSTM Model

Edit `funda/train_lstm_model.py`:
```python
# Model parameters
hidden_layer_size = 100  # Increase for more complex patterns
num_layers = 2           # Add more layers for deeper learning
dropout = 0.2           # Adjust to prevent overfitting
```

### Modify Outlier Strategies

Edit `funda/outlier_engine.py`:
```python
STRATEGIES = {
    "scalp":   ("1m", "1w", 21, 5, 1e9),   # Adjust these values
    "swing":   ("3m", "1m", 63, 21, 2e9),
    "longterm":("1y", "6m", 252, 126, 10e9),
}
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [SYSTEM_FLOWCHART.md](SYSTEM_FLOWCHART.md) for architecture details
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute to the project
- Join discussions for questions and tips

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/Billions/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Billions/discussions)
- **Documentation**: [Full README](README.md)

## ⚠️ Important Reminder

**This is not financial advice!** Always:
- Do your own research
- Never invest more than you can afford to lose
- Consult with licensed financial advisors
- Use this tool for educational purposes

---

**Ready to explore the markets? Happy trading! 📊**

[Back to Main README](README.md)

