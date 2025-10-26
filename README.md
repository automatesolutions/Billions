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
│  - 5 pages (login, dashboard, analyze)    │
│  - 20+ components                          │
│  - Custom SVG charts                       │
└────────────────────────────────────────────┘
                    │
                    │ REST API (21 endpoints)
                    │
┌────────────────────────────────────────────┐
│          FastAPI Backend                   │
│  - ML predictions (LSTM)                   │
│  - Outlier detection                       │
│  - News & sentiment                        │
│  - User management                         │
└────────────────────────────────────────────┘
                    │
                    │
┌────────────────────────────────────────────┐
│      SQLite Database                       │
│  - User data                               │
│  - Predictions                             │
│  - Market data cache                       │
└────────────────────────────────────────────┘
```

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
| [STATUS.md](STATUS.md) | Current project status & metrics |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Development setup guide |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Production deployment steps |
| [READY_TO_DEPLOY.md](READY_TO_DEPLOY.md) | Deployment checklist |
| [GOOGLE_OAUTH_SETUP.md](GOOGLE_OAUTH_SETUP.md) | OAuth configuration |

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

### Predictions (`/api/predictions`)
- `GET /api/predictions/{ticker}` - Get ML predictions
- `GET /api/predictions/info/{ticker}` - Get ticker info
- `POST /api/predictions/train` - Train new model

### Outliers (`/api/outliers`)
- `GET /api/outliers/{strategy}` - Get outlier data
- `GET /api/outliers/strategies` - List available strategies
- `POST /api/outliers/refresh` - Refresh outlier cache
- `GET /api/outliers/cache/{strategy}` - Get cached data

### News (`/api/news`)
- `GET /api/news/{ticker}` - Get ticker news
- `GET /api/news/{ticker}/sentiment` - Get sentiment analysis

### Users (`/api/users`)
- `POST /api/users` - Create user
- `GET /api/users/{user_id}` - Get user profile
- `PUT /api/users/{user_id}/preferences` - Update preferences
- `GET /api/users/{user_id}/watchlist` - Get watchlist
- `POST /api/users/{user_id}/watchlist` - Add to watchlist
- `GET /api/users/{user_id}/alerts` - Get alerts

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

- **Files Created**: 150+ files
- **Lines of Code**: 7,500+ lines
- **Documentation**: 5,000+ lines
- **API Endpoints**: 21 endpoints
- **Frontend Pages**: 5 pages
- **Components**: 20+ components
- **Tests**: 89 tests passing
- **Test Coverage**: 85% (backend)

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
