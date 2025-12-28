# 🚀 BILLIONS - ML-Powered Stock Forecasting Platform

<div align="center">

![BILLIONS Logo](web/public/logo.png)

**Advanced LSTM-based stock market forecasting and outlier detection**

[![Next.js](https://img.shields.io/badge/Next.js-15.5-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.118-green)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue)](https://www.typescriptlang.org/)
[![Tests](https://img.shields.io/badge/Tests-89%20passing-brightgreen)](.)
[![Coverage](https://img.shields.io/badge/Coverage-85%25-brightgreen)](.)

</div>

---

## 📊 Project Status

**Current Version**: v2.0 Web App (Phases 1-5 Complete)  
**Progress**: **71.9%** | 5.75/8 phases complete  
**Status**: ✅ **MVP READY FOR DEPLOYMENT**

### Phase Completion
- ✅ **Phase 0**: Foundation & Analysis (100%)
- ✅ **Phase 1**: Infrastructure Setup (100%)
- ✅ **Phase 2**: Testing Infrastructure (100%)
- ✅ **Phase 3**: Authentication & User Management (100%)
- ✅ **Phase 4**: ML Backend Migration (100%)
- ✅ **Phase 5**: Frontend Development MVP (100%)
- 🔄 **Phase 6**: Deployment & Monitoring (75%)
- ⏳ **Phase 7**: Data Migration (0%)
- ⏳ **Phase 8**: Launch (0%)

### Latest Updates (2025)
- ✅ **System Architecture Documentation** - Complete flowchart with file references
- ✅ **Interactive HTML Visualization** - Visual architecture explorer
- ✅ **Enhanced API Documentation** - 30+ endpoints documented
- ✅ **Communication Flow Diagrams** - Frontend ↔ Backend flows
- ✅ **File Structure Mapping** - Complete file organization guide

---

## ✨ Features

### 🤖 Machine Learning
- **LSTM Neural Networks** - Multi-timeframe stock predictions
- **Outlier Detection** - 3 strategies (scalp, swing, long-term)
- **Sentiment Analysis** - Real-time news sentiment scoring
- **Technical Analysis** - 20+ indicators and metrics

### 📈 Market Intelligence
- **Stock Forecasting** - 30-day predictions with confidence bands
- **Outlier Visualization** - Scatter plots for market anomalies
- **News Aggregation** - Real-time market news and analysis
- **Performance Metrics** - ROI, Sharpe ratio, volatility analysis

### 👤 User Features
- **Google OAuth** - Secure authentication
- **User Dashboards** - Personalized stock tracking
- **Watchlists** - Save favorite tickers
- **Alerts** - Price and prediction notifications
- **Auto-refresh** - Real-time data updates (5-min intervals)

### 🎨 User Interface
- **Dark Mode** - CLI-inspired mysterious theme
- **Custom Charts** - SVG-based prediction & scatter plots
- **Mobile Responsive** - Works on all devices
- **Toast Notifications** - Real-time user feedback

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────┐
│           Next.js Frontend                 │
│  - 8+ pages (login, dashboard, analyze)   │
│  - 30+ components                          │
│  - Custom SVG charts                       │
│  - Port: 3000                              │
└────────────────────────────────────────────┘
                    │
                    │ REST API (30+ endpoints)
                    │ WebSocket (HFT trading)
                    │
┌────────────────────────────────────────────┐
│          FastAPI Backend                   │
│  - ML predictions (LSTM)                   │
│  - Outlier detection                       │
│  - News & sentiment                        │
│  - User management                         │
│  - HFT trading engine                      │
│  - Portfolio management                    │
│  - Port: 8000                              │
└────────────────────────────────────────────┘
                    │
                    │
┌────────────────────────────────────────────┐
│      SQLite Database                       │
│  - User data                               │
│  - Predictions                             │
│  - Market data cache                       │
│  - Performance metrics                     │
└────────────────────────────────────────────┘
```

**📖 For detailed architecture documentation with file references and communication flows, see:**
- **[SYSTEM_ARCHITECTURE_FLOWCHART.md](SYSTEM_ARCHITECTURE_FLOWCHART.md)** - Complete architecture guide
- **[SYSTEM_ARCHITECTURE_FLOWCHART.html](SYSTEM_ARCHITECTURE_FLOWCHART.html)** - Interactive HTML visualization

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Node.js 20+**
- **pnpm 9+**
- **Google OAuth credentials**

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/billions.git
cd billions
```

### 2. Backend Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r api/requirements.txt
pip install -r api/requirements-dev.txt

# Start backend
python -m uvicorn api.main:app --reload
# Backend runs at http://localhost:8000
```

### 3. Frontend Setup
```bash
cd web

# Install dependencies
pnpm install

# Setup environment
cp .env.example .env.local
# Edit .env.local with your Google OAuth credentials

# Start frontend
pnpm dev
# Frontend runs at http://localhost:3000
```

### 4. Setup Google OAuth
See [GOOGLE_OAUTH_SETUP.md](GOOGLE_OAUTH_SETUP.md) for detailed instructions.

---

## 🧪 Testing

```bash
# Backend tests (pytest)
pytest                      # Run all backend tests
pytest --cov               # With coverage report

# Frontend tests (Vitest)
cd web
pnpm test                  # Run component tests
pnpm test:watch           # Watch mode

# E2E tests (Playwright)
cd web
pnpm test:e2e             # Run E2E tests
pnpm test:e2e:ui          # Interactive UI mode
```

**Test Statistics:**
- **89 total tests** ✅
- **Backend**: 57 pytest tests (85% coverage)
- **Frontend**: 20 component tests
- **E2E**: 12 Playwright tests

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PLAN.md](PLAN.md) | Complete project roadmap (602 lines) |
| [SYSTEM_ARCHITECTURE_FLOWCHART.md](SYSTEM_ARCHITECTURE_FLOWCHART.md) | **NEW** - Complete system architecture with file references |
| [SYSTEM_ARCHITECTURE_FLOWCHART.html](SYSTEM_ARCHITECTURE_FLOWCHART.html) | **NEW** - Interactive HTML flowchart visualization |
| [CHANGELOG.md](CHANGELOG.md) | Version history and changes |
| [FAQ.md](FAQ.md) | Frequently asked questions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [API_TESTING_RESULTS.md](API_TESTING_RESULTS.md) | API endpoint testing results |

---

## 🛠️ Tech Stack

### Frontend
- **Framework**: Next.js 15.5.4 (App Router)
- **Language**: TypeScript 5.9
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui
- **Auth**: NextAuth.js
- **Testing**: Vitest + Playwright

### Backend
- **Framework**: FastAPI 0.118
- **Language**: Python 3.12
- **ORM**: SQLAlchemy 2.0
- **ML**: PyTorch 2.4, TensorFlow 2.19
- **Testing**: pytest 8.4
- **Coverage**: 85%

### Infrastructure
- **Database**: SQLite (MVP), PostgreSQL (future)
- **CI/CD**: GitHub Actions
- **Frontend Deploy**: Vercel (configured)
- **Backend Deploy**: Railway/Render (configured)
- **Monitoring**: Sentry (ready to integrate)

---

## 🌐 API Endpoints

### Predictions (`/api/v1/predictions`)
- `GET /api/v1/predictions/{ticker}` - Get ML predictions
- `GET /api/v1/predictions/info/{ticker}` - Get ticker info
- `GET /api/v1/predictions/search` - Search tickers

### Market Data (`/api/v1/market`)
- `GET /api/v1/market/outliers/{strategy}` - Get outlier stocks
- `GET /api/v1/market/performance/{strategy}` - Get performance metrics
- `GET /api/v1/{ticker}/historical` - Historical price data

### Outliers (`/api/v1/outliers`)
- `GET /api/v1/outliers/{strategy}` - Get outlier data
- `GET /api/v1/outliers/strategies` - List available strategies
- `POST /api/v1/outliers/{strategy}/refresh` - Refresh outlier cache

### News (`/api/v1/news`)
- `GET /api/v1/news/{ticker}` - Get ticker news with sentiment
- `GET /api/v1/nasdaq-news/latest` - Latest NASDAQ news
- `GET /api/v1/nasdaq-news/urgent` - Urgent news alerts

### Trading (`/api/v1/trading`)
- `GET /api/v1/trading/status` - Trading account status
- `POST /api/v1/trading/execute` - Execute trade
- `GET /api/v1/trading/positions` - Current positions
- `POST /api/v1/trading/quote/{symbol}` - Real-time quote
- `GET /api/v1/trading/orders` - Order history

### HFT (`/api/v1/hft`)
- `GET /api/v1/hft/status` - HFT engine status
- `POST /api/v1/hft/start` - Start HFT engine
- `POST /api/v1/hft/stop` - Stop HFT engine
- `POST /api/v1/hft/orders` - Submit HFT order
- `GET /api/v1/hft/performance` - Performance metrics

### Portfolio (`/api/v1/portfolio`)
- `POST /api/v1/portfolio/calculate-metrics` - Calculate portfolio metrics
- `GET /api/v1/portfolio/risk-analysis/{ticker}` - Risk analysis
- `POST /api/v1/portfolio/calculate-allocation` - Optimal allocation

### Valuation (`/api/v1/valuation`)
- `GET /api/v1/valuation/{ticker}` - Stock valuation
- `GET /api/v1/valuation/{ticker}/fair-value` - Black-Scholes fair value

### Users (`/api/v1/users`)
- `POST /api/v1/users/` - Create user
- `GET /api/v1/users/{user_id}` - Get user profile
- `PUT /api/v1/users/{user_id}/preferences` - Update preferences
- `GET /api/v1/users/{user_id}/watchlist` - Get watchlist
- `POST /api/v1/users/{user_id}/watchlist` - Add to watchlist

### Behavioral (`/api/v1/behavioral`)
- `POST /api/v1/behavioral/rationale` - Add trade rationale
- `GET /api/v1/behavioral/insights` - Behavioral insights
- `GET /api/v1/behavioral/performance-analysis` - Performance analysis

### Capitulation (`/api/v1/capitulation`)
- `GET /api/v1/capitulation/screen` - Screen for capitulation
- `GET /api/v1/capitulation/analyze/{symbol}` - Analyze stock

**📖 See [SYSTEM_ARCHITECTURE_FLOWCHART.md](SYSTEM_ARCHITECTURE_FLOWCHART.md) for detailed endpoint documentation with file references.**

---

## 📁 Project Structure

```
Billions/
├── web/                     # Next.js Frontend
│   ├── app/                # Pages (App Router)
│   │   ├── login/         # Authentication
│   │   ├── dashboard/     # User dashboard
│   │   ├── analyze/       # Stock analysis
│   │   └── outliers/      # Outlier detection
│   ├── components/        # UI components
│   │   ├── charts/        # Custom SVG charts
│   │   ├── ui/            # shadcn/ui components
│   │   └── ...
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # API client & utilities
│   ├── __tests__/         # Component tests (20)
│   └── e2e/               # E2E tests (12)
│
├── api/                   # FastAPI Backend
│   ├── routers/           # API routes
│   │   ├── predictions.py
│   │   ├── outliers.py
│   │   ├── news.py
│   │   └── users.py
│   ├── services/          # Business logic
│   ├── tests/             # Backend tests (57)
│   └── main.py            # FastAPI app
│
├── db/                    # Database
│   ├── models.py          # SQLAlchemy models
│   └── models_auth.py     # User models
│
├── funda/                 # ML Models (legacy)
│   ├── SPS.py             # News & sentiment
│   ├── train_lstm_model.py
│   └── outlier_engine.py
│
├── .github/
│   └── workflows/         # CI/CD
│       ├── test.yml       # Test pipeline
│       ├── lint.yml       # Linting
│       └── deploy.yml     # Deployment
│
├── vercel.json            # Vercel config
├── railway.json           # Railway config
├── render.yaml            # Render config
└── docker-compose.yml     # Dev environment
```

---

## 🎯 Key Statistics

- **Files Created**: 200+ files
- **Lines of Code**: 10,000+ lines
- **Documentation**: 8,000+ lines
- **API Endpoints**: 30+ endpoints
- **Frontend Pages**: 8+ pages
- **Components**: 30+ components
- **Backend Routers**: 13 routers
- **Backend Services**: 12 services
- **Tests**: 89 tests passing
- **Test Coverage**: 85% (backend)
- **Architecture Docs**: Complete flowchart with file references

---

## 🚀 Deployment

The application is **ready to deploy**! Configuration files are in place for:

1. **Frontend (Vercel)** - `vercel.json` configured
2. **Backend (Railway or Render)** - `railway.json` / `render.yaml` configured
3. **CI/CD (GitHub Actions)** - Automated testing & deployment

**To deploy**, follow the step-by-step guide in [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

---

## 🔒 Security

- **Google OAuth** - Secure authentication via NextAuth.js
- **JWT Sessions** - Stateless authentication
- **CORS Protection** - Configured for production
- **Environment Variables** - Secrets management
- **Rate Limiting** - API throttling (future)
- **SQL Injection Protection** - SQLAlchemy parameterized queries

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

## 📞 Support

For questions or issues:
1. Check the [FAQ.md](FAQ.md)
2. Review [DEVELOPMENT.md](DEVELOPMENT.md)
3. Open a GitHub issue

---

## 🎉 Acknowledgments

Built with modern best practices:
- Test-Driven Development (TDD)
- Continuous Integration/Deployment (CI/CD)
- Comprehensive documentation
- Clean architecture

---

<div align="center">

**BILLIONS** - Machine Learning for Trading Intelligence

Made with ❤️ and ☕

[Website](#) | [Docs](PLAN.md) | [API Docs](http://localhost:8000/docs)

</div>
