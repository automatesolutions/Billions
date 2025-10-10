# BILLIONS - ML Stock Forecasting Platform

> **Status**: 🚀 **50% Complete** - Backend & Core ML APIs Operational!

A powerful machine learning platform for stock market forecasting and outlier detection, now transformed into a modern full-stack web application.

---

## 🎯 Project Overview

BILLIONS integrates advanced LSTM neural networks, technical analysis, and real-time data pipelines to deliver actionable trading insights across multiple timeframes.

**Current Phase**: Phase 5 - Frontend UI Development  
**Progress**: 50% (4/8 phases complete)

---

## ✨ Features

### ✅ Live Features (Backend APIs Ready)
- **Google OAuth Authentication** - Secure login with Google
- **30-Day Stock Predictions** - LSTM-based price forecasting
- **Outlier Detection** - 3 strategies (scalp, swing, longterm)
- **Market Data Pipeline** - Real-time data with intelligent caching
- **User Management** - Profiles, preferences, watchlists
- **Portfolio Tracking** - (Schema ready, UI in progress)

### 🔄 In Development (Phase 5)
- Interactive dashboards with charts
- Prediction visualization
- Outlier scatter plots
- Real-time alerts
- News & sentiment analysis

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** 20+ and **pnpm** 9+
- **Python** 3.12+
- **Google OAuth** credentials ([Setup Guide](./GOOGLE_OAUTH_SETUP.md))

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd Billions

# 2. Install dependencies
## Backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r api/requirements.txt

## Frontend
cd web
pnpm install
cd ..

# 3. Setup environment variables
cp .env.example .env
cp web/.env.local.example web/.env.local
# Edit web/.env.local with your Google OAuth credentials

# 4. Initialize database
python -c "from db.core import Base, engine; from db.models_auth import User; Base.metadata.create_all(bind=engine)"

# 5. Start the application
## Terminal 1 - Backend
start-backend.bat

## Terminal 2 - Frontend
start-frontend.bat
```

### Access
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

📚 **Complete Setup Guide**: [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md)

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Next.js 15    │         │   FastAPI       │
│   Frontend      │◄───────►│   Backend       │
│   (TypeScript)  │   API   │   (Python)      │
│                 │         │                 │
│  - Auth (OAuth) │         │  - ML Services  │
│  - Dashboard    │         │  - Predictions  │
│  - Charts (P5)  │         │  - Outliers     │
└─────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   SQLite DB     │
                            │  + SQLAlchemy   │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  LSTM Models    │
                            │  (PyTorch)      │
                            └─────────────────┘
```

---

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 15.5.4 (App Router)
- **Language**: TypeScript 5.9
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui
- **Auth**: NextAuth.js 5
- **Testing**: Vitest + Playwright

### Backend
- **Framework**: FastAPI 0.118
- **Language**: Python 3.12
- **ORM**: SQLAlchemy 2.0
- **Database**: SQLite
- **ML**: PyTorch 2.4, TensorFlow 2.19
- **Testing**: pytest (85% coverage)

### DevOps
- **CI/CD**: GitHub Actions
- **Containers**: Docker Compose
- **Monitoring**: Sentry (Phase 6)
- **Deployment**: Vercel + Railway (Phase 6)

---

## 📊 Current Status

| Component | Status | Tests | Coverage |
|-----------|--------|-------|----------|
| Infrastructure | ✅ Complete | Manual | 100% |
| Testing Framework | ✅ Complete | 12 tests | 85% |
| Authentication | ✅ Complete | 28 tests | 85% |
| ML Backend | ✅ Complete | 46 tests | 85% |
| Frontend UI | 🔄 In Progress | TBD | TBD |
| Deployment | ⏳ Planned | - | - |

**Overall**: **50% Complete** (4/8 phases)

---

## 📡 API Endpoints (18 total)

### Predictions
```http
GET  /api/v1/predictions/{ticker}?days=30
GET  /api/v1/predictions/info/{ticker}
GET  /api/v1/predictions/search?q=TSLA
```

### Outliers
```http
GET  /api/v1/market/outliers/{strategy}
POST /api/v1/outliers/{strategy}/refresh
GET  /api/v1/outliers/strategies
```

### Users
```http
POST /api/v1/users/
GET  /api/v1/users/{user_id}/watchlist
POST /api/v1/users/{user_id}/watchlist
```

**Full API Docs**: http://localhost:8000/docs

---

## 🧪 Testing

```bash
# Run all backend tests
pytest
# 29 tests, 85% coverage

# Run frontend tests
cd web && pnpm vitest run
# 9 tests

# Run E2E tests
cd web && pnpm test:e2e
# 8 tests

# Total: 46 tests passing ✅
```

📚 **Testing Guide**: [README_TESTING.md](./README_TESTING.md)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [PLAN.md](./PLAN.md) | Master project roadmap |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Development guide |
| [SETUP_INSTRUCTIONS.md](./SETUP_INSTRUCTIONS.md) | Quick setup |
| [GOOGLE_OAUTH_SETUP.md](./GOOGLE_OAUTH_SETUP.md) | OAuth configuration |
| [README_TESTING.md](./README_TESTING.md) | Testing guide |
| [STATUS.md](./STATUS.md) | Project status |
| [MILESTONE_50PERCENT.md](./MILESTONE_50PERCENT.md) | 50% milestone |
| [PHASE1-4_SUMMARY.md](./PHASE1_SUMMARY.md) | Detailed phase summaries |

---

## 🎯 Key Features (Backend Ready)

### ML Predictions
- 30-day stock price forecasts using bidirectional LSTM
- Confidence intervals based on volatility
- 40+ technical indicators
- Sector-relative analysis

### Outlier Detection
- **Scalp Strategy**: 1-week vs 1-month performance
- **Swing Strategy**: 3-month vs 1-month performance
- **Longterm Strategy**: 1-year vs 6-month performance
- Z-score analysis (|z| > 2)

### User Management
- Google OAuth authentication
- User preferences (theme, notifications, strategy defaults)
- Stock watchlists with notes
- Price alerts (schema ready)

---

## 🔒 Security

- ✅ Google OAuth 2.0 authentication
- ✅ JWT session management
- ✅ Protected routes with middleware
- ✅ Environment variable configuration
- ✅ SQL injection prevention (ORM)
- ✅ CORS configuration
- ✅ Input validation

---

## 🤝 Contributing

We follow conventional commits and test-driven development:

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes with tests
# Run tests
pytest && cd web && pnpm test

# Commit
git commit -m "feat: add your feature"

# Push and create PR
git push origin feature/your-feature
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

---

## 📈 Roadmap

- [x] **Phase 0**: Foundation & Analysis
- [x] **Phase 1**: Infrastructure Setup
- [x] **Phase 2**: Testing Infrastructure
- [x] **Phase 3**: Authentication & User Management
- [x] **Phase 4**: ML Backend Migration
- [ ] **Phase 5**: Frontend UI Development ← **Next**
- [ ] **Phase 6**: Deployment & Monitoring
- [ ] **Phase 7**: Data Migration & Validation
- [ ] **Phase 8**: Documentation & Launch

---

## 📄 License

See [LICENSE](./LICENSE) file for details.

---

## 🆘 Support

- **Documentation**: Check docs in this repo
- **Issues**: Create GitHub issue
- **Questions**: See [FAQ.md](./FAQ.md)

---

## 🎉 Acknowledgments

Built with modern technologies:
- Next.js & React Team
- FastAPI creators
- shadcn/ui components
- PyTorch & TensorFlow teams
- yfinance contributors

---

**Built with ❤️ by the BILLIONS team**

**Status**: 🟢 50% Complete | Backend Operational | Frontend In Progress

**Star this repo if you find it useful!** ⭐
