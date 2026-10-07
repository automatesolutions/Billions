# Changelog

All notable changes to BILLIONS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2.0.0 "outliers only"

BILLIONS is now a read-only outlier intelligence tool. It never places, routes or simulates trades.

### Removed
- Trading, HFT, Alpaca, portfolio, behavioral, capitulation and valuation features (frontend pages, API routers, services, C++ HFT engine and root scripts).
- Google OAuth, NextAuth, JWT and user/watchlist/alert models. The app is public with no accounts.
- News and hype detection, LSTM and Markov forecasts, the legacy Dash app (`funda/SPS.py`), standalone outlier scripts and committed data, model and media files.
- Stale docs (architecture flowcharts, plan, API test results, screenshots, FAQ, quickstart).

### Changed
- Backend is one package (`api/`). Outlier engine moved to `api/services/outlier_engine.py`.
- Configuration comes from environment variables (`CORS_ORIGINS`, `DATABASE_URL`). See `.env.example`.
- One command (`pnpm dev`) starts the backend and frontend together.
- Deploy configs: Railway and Render build `api/Dockerfile`; Vercel builds `web/`.
- One CI workflow (`.github/workflows/ci.yml`).

### Security
- Removed `create-env.bat`, which contained Alpaca API keys. Those keys must be revoked; they remain in git history.

## [1.0.0] - 2025-10-08

### Added
- Initial release of BILLIONS ML Prediction System
- LSTM-based stock price prediction with 30-day forecasts
- Enhanced feature engineering with 50+ technical indicators
- Multi-strategy outlier detection (Scalp, Swing, Long-term)
- Interactive Dash/Plotly dashboard
- Real-time data fetching from Yahoo Finance and Alpha Vantage
- SQLite database for performance metrics storage
- Institutional flow analysis
- Confidence scoring system
- Sector correlation analysis with SPY and sector ETFs
- Automated background data refresh
- Model diagnostics and feature importance analysis
- Comprehensive technical indicators:
  - Momentum: RSI, MACD, Stochastic, ROC
  - Trend: SMA, EMA, ADX, Parabolic SAR
  - Volatility: Bollinger Bands, ATR, Keltner Channels
  - Volume: OBV, Volume patterns, Accumulation/Distribution

### Core Modules
- `SPS.py` - Main dashboard application
- `train_lstm_model.py` - LSTM model training pipeline
- `enhanced_features.py` - Advanced feature engineering
- `outlier_engine.py` - Outlier detection engine
- `refresh_outliers.py` - Background data refresh
- `fine_tuning_strategy.py` - Strategy optimization
- `model_diagnostics.py` - Model analysis tools
- Database layer (`db/core.py`, `db/models.py`)
- Strategy modules (Scalp, Swing, Long-term)

### Documentation
- Comprehensive README.md with installation guide
- QUICKSTART.md for rapid setup
- SYSTEM_FLOWCHART.md with architecture diagrams
- CONTRIBUTING.md with contribution guidelines
- MIT License
- GitHub Actions CI/CD workflow

### Infrastructure
- SQLAlchemy-based database management
- PyTorch LSTM model architecture
- Multi-ticker data caching system
- Error handling and logging
- API rate limiting

## [0.9.0] - Development Phase

### Added
- Prototype LSTM models
- Basic technical indicators
- Initial outlier detection logic
- Database schema design
- Core prediction algorithms

### Changed
- Migrated from simple moving averages to enhanced features
- Improved model accuracy with additional layers
- Optimized data fetching and caching

### Fixed
- NaN value handling in feature engineering
- Database connection timeout issues
- Cache invalidation bugs

---

## Version History Legend

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security vulnerability fixes

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this changelog.

## Support

For questions or issues, please visit our [GitHub Issues](https://github.com/yourusername/Billions/issues).

---

*Keep building, keep improving! 🚀*

