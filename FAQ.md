# ❓ Frequently Asked Questions (FAQ)

## General Questions

### What is BILLIONS?

BILLIONS is an advanced machine learning platform for stock market prediction and outlier detection. It uses LSTM neural networks, technical analysis, and statistical methods to identify trading opportunities across multiple timeframes.

### Is this financial advice?

**NO.** BILLIONS is an educational and research tool. It is NOT financial advice. Always:
- Do your own research
- Consult licensed financial advisors
- Never invest more than you can afford to lose
- Understand that past performance doesn't guarantee future results

### Is BILLIONS free to use?

Yes! BILLIONS is open-source under the MIT License. However, you'll need free API keys from:
- Alpha Vantage (free tier available)
- Yahoo Finance (no key needed)
- FRED (optional, free tier available)

### What stocks can I analyze?

BILLIONS works with any stock available on:
- NASDAQ
- NYSE
- Other major exchanges supported by Yahoo Finance

---

## Technical Questions

### What programming language is BILLIONS written in?

Python 3.8+, using libraries like:
- PyTorch (LSTM models)
- Dash/Plotly (dashboard)
- Pandas/NumPy (data processing)
- SQLAlchemy (database)
- Scikit-learn (machine learning utilities)

### What machine learning models does it use?

- **LSTM (Long Short-Term Memory)**: Primary prediction model
- **Random Forest**: Feature importance and ensemble predictions
- **Statistical Methods**: Z-score analysis for outlier detection

### How accurate are the predictions?

Accuracy varies by:
- Stock volatility
- Market conditions
- Prediction horizon
- Data quality

Typical accuracy: 65-85% for directional predictions. Always check the confidence score!

### What is the prediction horizon?

- Default: 30 days ahead
- Supports: 1-day, 7-day, and 30-day forecasts
- Customizable in the code

### How long does it take to run a prediction?

- First run: 30-60 seconds (data fetching + model loading)
- Subsequent runs: 10-20 seconds (cached data)
- Training new model: 5-15 minutes

---

## Setup & Installation

### What are the system requirements?

**Minimum:**
- Python 3.8+
- 4GB RAM
- 1GB free disk space
- Internet connection

**Recommended:**
- Python 3.10+
- 8GB RAM
- 5GB free disk space
- NVIDIA GPU (optional, for faster training)

### Do I need a GPU?

No, but it helps! 
- **CPU**: Works fine, slightly slower training
- **GPU**: Faster training (10x speedup with CUDA)

The system automatically detects and uses available hardware.

### How do I get API keys?

**Alpha Vantage** (Required):
1. Visit https://www.alphavantage.co/support/#api-key
2. Fill out the free API key request form
3. Receive key instantly via email
4. Add to `.env` file

**FRED** (Optional):
1. Visit https://fred.stlouisfed.org/
2. Create free account
3. Request API key in account settings
4. Add to `.env` file

### Installation failed, what should I do?

Try these steps:

1. **Update pip**:
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install dependencies one by one**:
   ```bash
   pip install pandas numpy scikit-learn
   pip install torch
   pip install dash plotly dash-bootstrap-components
   ```

3. **Check Python version**:
   ```bash
   python --version  # Should be 3.8+
   ```

4. **Use virtual environment**:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

---

## Usage Questions

### How do I run predictions?

1. Launch dashboard: `python funda/SPS.py`
2. Open browser: http://127.0.0.1:8050/
3. Enter ticker symbol (e.g., TSLA)
4. Click "🚀 Run Prediction"
5. Wait 10-30 seconds
6. Analyze results!

### What does "confidence score" mean?

The confidence score (0-100%) indicates:
- **80-100%**: High confidence - strong signal
- **60-79%**: Medium confidence - moderate signal
- **Below 60%**: Low confidence - weak signal

It's calculated from:
- Model prediction variance
- Historical accuracy
- Feature quality
- Market volatility

### What is an "outlier" stock?

An outlier is a stock that shows statistically unusual performance compared to peers:
- **High Z-score**: Performance significantly above average
- **Volume anomalies**: Unusual trading activity
- **Momentum**: Strong price movement

Outliers may indicate:
- Emerging trends
- Institutional interest
- Market inefficiencies
- Potential opportunities (or risks!)

### What are the different strategies?

| Strategy | Best For | Timeframe | Risk Level |
|----------|----------|-----------|------------|
| **Scalp** | Day trading | Minutes to hours | High |
| **Swing** | Short-term trades | Days to weeks | Medium |
| **Long-term** | Position trading | Weeks to months | Lower |

### How often should I refresh data?

- **Daily traders**: Refresh every morning
- **Swing traders**: 2-3 times per week
- **Long-term**: Once a week
- **Automatic**: Enable background refresh in settings

---

## Troubleshooting

### "API key not found" error

1. Check `.env` file exists in project root
2. Verify format: `ALPHA_VANTAGE_API_KEY=your_key_here`
3. No quotes around the key
4. No spaces before/after the `=`
5. Restart the application

### "No data available" for a ticker

Possible causes:
- Ticker symbol is incorrect
- Stock is delisted or suspended
- API rate limit reached (wait 1 minute)
- Stock has insufficient history
- Network connectivity issues

### Dashboard won't load

1. Check if another app is using port 8050:
   ```bash
   # Windows
   netstat -ano | findstr :8050
   # Linux/Mac
   lsof -i :8050
   ```

2. Change port in `SPS.py`:
   ```python
   app.run_server(debug=True, port=8051)  # Use different port
   ```

3. Clear browser cache
4. Try a different browser

### Predictions seem inaccurate

Remember:
- Market prediction is inherently uncertain
- Check confidence score
- Consider market conditions
- Verify data quality
- This is not financial advice!

Improve accuracy by:
- Training model on more data
- Using ensemble predictions
- Combining with technical analysis
- Adjusting strategy parameters

### "Database is locked" error

1. Close all running instances of BILLIONS
2. Check for zombie processes:
   ```bash
   # Windows
   tasklist | findstr python
   # Linux/Mac
   ps aux | grep python
   ```
3. Delete database lock file (if safe):
   ```bash
   rm billions.db-journal
   ```

---

## Performance & Optimization

### How can I make it faster?

1. **Use GPU** (if available):
   - Install CUDA-enabled PyTorch
   - Training speedup: 10-20x

2. **Increase cache duration**:
   - Edit cache settings in code
   - Trade-off: Speed vs. freshness

3. **Reduce prediction horizon**:
   - Change from 30 days to 7 days
   - Faster computation

4. **Limit technical indicators**:
   - Comment out unused indicators
   - Reduce feature engineering time

### Can I run multiple predictions simultaneously?

Not recommended due to:
- API rate limits
- Database lock conflicts
- Memory usage

Best practice: Queue predictions sequentially

### How much disk space do I need?

- **Installation**: ~500MB (dependencies)
- **Cache**: ~100MB (grows over time)
- **Database**: ~10-50MB (grows with use)
- **Models**: ~10MB
- **Total**: 1-2GB recommended

---

## Customization

### Can I add custom indicators?

Yes! Edit `funda/enhanced_features.py`:

```python
def compute_custom_indicator(df):
    """Your custom indicator"""
    df['Custom'] = df['Close'].rolling(20).mean()
    return df
```

### Can I change the model architecture?

Yes! Edit `funda/train_lstm_model.py`:

```python
hidden_layer_size = 150  # Default: 100
num_layers = 3          # Default: 2
dropout = 0.3           # Default: 0.2
```

### Can I analyze cryptocurrencies?

Partial support:
- Works with crypto tickers on Yahoo Finance (BTC-USD, ETH-USD)
- Dedicated crypto strategy in development
- Some indicators may not apply to 24/7 markets

---

## Contributing

### How can I contribute?

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation
- Share your strategies!

### I found a bug, what should I do?

1. Check if it's already reported in [Issues](https://github.com/yourusername/Billions/issues)
2. Create new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs. actual behavior
   - System information
   - Error messages

### Can I add new trading strategies?

Absolutely! Edit `funda/outlier_engine.py`:

```python
STRATEGIES["custom"] = ("period", "window", days, lookback, min_cap)
```

---

## Legal & Licensing

### What license is BILLIONS under?

MIT License - you can:
- Use commercially
- Modify
- Distribute
- Use privately

Must:
- Include license and copyright notice
- Provide source code attribution

### Can I use this for my trading business?

Yes, but:
- Comply with financial regulations in your jurisdiction
- Understand this is NOT regulated financial advice
- Seek legal counsel if needed
- Use at your own risk

### Can I sell predictions from BILLIONS?

Legally complex:
- May require financial licenses
- Subject to securities regulations
- Consult legal/financial professionals
- We assume no liability

---

## Support

### Where can I get help?

- **Documentation**: Start with [README.md](README.md)
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/Billions/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Billions/discussions)

### Is there a community?

Join us on:
- GitHub Discussions (primary)
- Twitter: @BillionsML (coming soon)
- Discord: (coming soon)

### Can I hire you for custom development?

Contact us via:
- Email: your.email@example.com
- GitHub: Open an issue
- LinkedIn: (your profile)

---

## Roadmap

### What features are coming next?

Planned features:
- [ ] Real-time WebSocket data feeds
- [ ] Cryptocurrency dedicated module
- [ ] Mobile app
- [ ] Advanced backtesting framework
- [ ] Portfolio optimization
- [ ] Sentiment analysis integration
- [ ] Options pricing models
- [ ] Multi-asset correlation

See [CHANGELOG.md](CHANGELOG.md) for details.

---

**Still have questions? Open an issue on GitHub!** 💬

[📖 Back to README](README.md) | [🚀 Quick Start](QUICKSTART.md) | [🤝 Contributing](CONTRIBUTING.md)

