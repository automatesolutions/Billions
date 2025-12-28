# BILLIONS System Architecture Flowchart

**Documentation Date:** Generated automatically  
**Version:** 1.0.0  
**Purpose:** Visual representation of frontend-backend communication flow

---

## System Overview

The BILLIONS platform consists of:
- **Frontend**: Next.js 15 application (TypeScript/React)
- **Backend**: FastAPI application (Python)
- **Database**: SQLite with SQLAlchemy ORM
- **Communication**: REST API (JSON) + WebSocket (HFT)

---

## Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Next.js)                                 │
│                         Port: 3000 (localhost)                              │
│                    File: web/ (Next.js application)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
        ┌───────────▼──────────┐      ┌───────────▼──────────┐
        │   Pages (Routes)      │      │   Components         │
        │   web/app/             │      │   web/components/    │
        │                       │      │                      │
        │ • / (Home)            │      │ • Charts             │
        │   page.tsx            │      │   charts/            │
        │ • /login              │      │ • Forms              │
        │   login/page.tsx      │      │ • Cards              │
        │ • /dashboard          │      │ • Navigation         │
        │   dashboard/page.tsx  │      │   nav-menu.tsx       │
        │ • /analyze/[ticker]   │      │ • Search             │
        │   analyze/[ticker]/   │      │   ticker-search.tsx  │
        │ • /outliers           │      │                      │
        │   outliers/page.tsx  │      │                      │
        │ • /portfolio          │      │                      │
        │   portfolio/page.tsx  │      │                      │
        │ • /trading/hft        │      │                      │
        │   trading/hft/page.tsx│      │                      │
        │ • /capitulation       │      │                      │
        │   capitulation/page.tsx│     │                      │
        └───────────┬───────────┘      └───────────┬──────────┘
                    │                               │
                    └───────────┬───────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   API Client          │
                    │   File: web/lib/api.ts│
                    │                       │
                    │ • ApiClient class     │
                    │ • HTTP methods        │
                    │ • Error handling      │
                    │ • Base URL:           │
                    │   localhost:8000      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Custom Hooks        │
                    │   web/hooks/          │
                    │                       │
                    │ • usePrediction()     │
                    │   use-prediction.ts   │
                    │ • useOutliers()       │
                    │   use-outliers.ts     │
                    │ • useValuation()      │
                    │   use-valuation.ts    │
                    │ • useHftQuotes()      │
                    │   use-hft-quotes.ts   │
                    │ • useAutoRefresh()    │
                    │   use-auto-refresh.ts │
                    └───────────┬───────────┘
                                │
                                │ HTTP Requests
                                │ (GET, POST, PUT, DELETE)
                                │ JSON Payloads
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI)                                        │
│                    Port: 8000 (localhost)                                   │
│                    File: api/ (Python application)                          │
│                    CORS: Enabled for localhost:3000                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   main.py             │
                    │   File: api/main.py   │
                    │   FastAPI App         │
                    │                       │
                    │ • CORS Middleware     │
                    │ • Lifespan Events     │
                    │ • Router Registration │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
┌───────▼────────┐   ┌──────────▼──────────┐  ┌────────▼─────────┐
│   Routers      │   │   Services          │  │   Database       │
│   api/routers/ │   │   api/services/     │  │   db/            │
│                │   │                     │  │                  │
│ • market.py    │   │ • predictions.py    │  │ • SQLite DB      │
│ • predictions  │   │ • outlier_detection │  │   billions.db    │
│ • outliers     │   │ • market_data.py    │  │ • SQLAlchemy     │
│ • news         │   │ • black_scholes.py  │  │   db/core.py     │
│ • trading      │   │ • news_service.py   │  │ • Models:        │
│ • hft          │   │ • trading_service   │  │   db/models.py   │
│ • portfolio    │   │ • behavioral.py     │  │   - User         │
│ • valuation    │   │ • capitulation_*    │  │   - PerfMetric   │
│ • capitulation │   │ • nasdaq_news_*     │  │   - Watchlist    │
│ • behavioral   │   │                     │  │   - Alert        │
│ • nasdaq_news  │   │                     │  │   db/models_auth.py│
│ • users        │   │                     │  │                  │
│ • historical   │   │                     │  │                  │
└───────┬────────┘   └──────────┬──────────┘  └────────┬─────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   External APIs       │
                    │   (via services)      │
                    │                       │
                    │ • Alpha Vantage       │
                    │ • Polygon.io          │
                    │ • FRED (Federal Res)  │
                    │ • News API            │
                    │ • OpenAI/Anthropic    │
                    │ • Alpaca Trading API  │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   ML Models           │
                    │   funda/              │
                    │                       │
                    │ • LSTM Models         │
                    │   funda/model/*.pt    │
                    │ • Outlier Engine      │
                    │   funda/outlier_*.py  │
                    │ • Cache System        │
                    │   funda/cache/*.csv   │
                    └───────────────────────┘
```

---

## Detailed Communication Flows

### 1. User Authentication Flow

**Files Involved:**
- Frontend: `web/app/login/page.tsx`
- Frontend: `web/auth.ts` (NextAuth configuration)
- Frontend: `web/app/api/auth/[...nextauth]/route.ts`
- Backend: `api/routers/users.py`

**Flow:**
```
User → /login → NextAuth → Google OAuth → Session → /dashboard
```

**Code Path:**
1. User clicks login → `web/app/login/page.tsx` → `signIn("google")`
2. NextAuth handles OAuth → `web/auth.ts`
3. Session created → Redirect to `/dashboard`
4. Dashboard checks session → `web/app/dashboard/page.tsx`

---

### 2. Stock Analysis Flow

**Files Involved:**
- Frontend: `web/components/analyze-stock-search.tsx`
- Frontend: `web/app/analyze/[ticker]/page.tsx`
- Frontend: `web/app/analyze/[ticker]/client-page.tsx`
- Frontend: `web/hooks/use-prediction.ts`
- Frontend: `web/lib/api.ts` → `getPrediction()`
- Backend: `api/routers/predictions.py`
- Backend: `api/services/predictions.py`
- ML: `funda/train_lstm_model.py` (model training)
- ML: `funda/model/*.pt` (trained models)

**Flow:**
```
User searches ticker
    ↓
Frontend: AnalyzeStockSearch component
    ↓
API Call: api.getPrediction(ticker)
    ↓
Backend: /api/v1/predictions/{ticker}
    ↓
Service: predictions.py → LSTM Model
    ↓
Response: {predictions, confidence, current_price}
    ↓
Frontend: PredictionChart component renders
```

**Code Path:**
1. User input → `web/components/analyze-stock-search.tsx`
2. Navigation → `web/app/analyze/[ticker]/page.tsx`
3. Data fetch → `web/app/analyze/[ticker]/client-page.tsx`
4. Hook → `web/hooks/use-prediction.ts`
5. API call → `web/lib/api.ts` → `GET /api/v1/predictions/{ticker}`
6. Backend router → `api/routers/predictions.py`
7. Service → `api/services/predictions.py`
8. ML model → Load `funda/model/*.pt`
9. Response → JSON → Frontend renders chart

---

### 3. Outliers Detection Flow

**Files Involved:**
- Frontend: `web/app/outliers/page.tsx`
- Frontend: `web/app/outliers/client-page.tsx`
- Frontend: `web/hooks/use-outliers.ts`
- Frontend: `web/lib/api.ts` → `getOutliers()`
- Backend: `api/routers/market.py` or `api/routers/outliers.py`
- Backend: `api/services/outlier_detection.py`
- Backend: `api/database.py` → `get_db()`
- Database: `billions.db` → `PerfMetric` table
- ML: `funda/outlier_engine.py`
- ML: `outlier/Outlier_Nasdaq_*.py`

**Flow:**
```
User visits /outliers
    ↓
Frontend: useOutliers hook (auto-refresh every 5min)
    ↓
API Call: api.getOutliers(strategy)
    ↓
Backend: /api/v1/market/outliers/{strategy}
    ↓
Service: outlier_detection.py → Outlier Engine
    ↓
Database: Query PerfMetric table
    ↓
Response: {outliers: [...], metrics: [...]}
    ↓
Frontend: ScatterPlot component visualizes
```

**Code Path:**
1. Page load → `web/app/outliers/page.tsx`
2. Client component → `web/app/outliers/client-page.tsx`
3. Hook → `web/hooks/use-outliers.ts` (auto-refresh: 5min)
4. API call → `web/lib/api.ts` → `GET /api/v1/market/outliers/{strategy}`
5. Backend router → `api/routers/market.py` → `/outliers/{strategy}`
6. Service → `api/services/outlier_detection.py`
7. Database query → `api/database.py` → `billions.db` → `PerfMetric`
8. Outlier calculation → `funda/outlier_engine.py`
9. Response → JSON → Frontend → `web/components/charts/scatter-plot.tsx`

---

### 4. HFT Trading Flow

**Files Involved:**
- Frontend: `web/app/trading/hft/page.tsx`
- Frontend: `web/hooks/use-hft-quotes.ts`
- Frontend: `web/hooks/use-orderbook.ts`
- Frontend: `web/lib/api.ts` → `hftStatus()`, `hftSubmitOrder()`
- Backend: `api/routers/hft.py`
- Backend: `api/services/trading_service.py`
- External: Alpaca WebSocket → `alpaca_websocket_hft_manager.py`
- External: Alpaca Trading API

**Flow:**
```
User on /trading/hft page
    ↓
Frontend: HftTradingPage component
    ↓
API Calls:
  - api.hftStatus() → GET /api/v1/hft/status
  - api.hftPerformance() → GET /api/v1/hft/performance
  - api.hftSubmitOrder() → POST /api/v1/hft/orders
    ↓
Backend: hft.py router
    ↓
Service: HFT Manager → Alpaca WebSocket
    ↓
External: Alpaca Trading API
    ↓
Response: Order status, quotes, positions
    ↓
Frontend: Real-time updates via WebSocket hooks
```

**Code Path:**
1. Page load → `web/app/trading/hft/page.tsx`
2. Status check → `web/lib/api.ts` → `GET /api/v1/hft/status`
3. Backend → `api/routers/hft.py` → `/status`
4. WebSocket manager → `alpaca_websocket_hft_manager.py`
5. Alpaca API → Real-time quotes
6. Order submission → `POST /api/v1/hft/orders`
7. Backend → `api/routers/hft.py` → `/orders`
8. Trading service → `api/services/trading_service.py`
9. Alpaca execution → Order placed
10. Real-time updates → `web/hooks/use-hft-quotes.ts` (WebSocket)

---

### 5. News & Sentiment Flow

**Files Involved:**
- Frontend: `web/app/analyze/[ticker]/news-section.tsx`
- Frontend: `web/components/hype-warning-card.tsx`
- Frontend: `web/lib/api.ts` → `getNews()`
- Backend: `api/routers/news.py`
- Backend: `api/services/enhanced_news_service.py`
- Backend: `api/services/advanced_hype_detector.py`
- External: News API, OpenAI/Anthropic APIs

**Flow:**
```
User views stock analysis
    ↓
Frontend: NewsSection component
    ↓
API Call: api.getNews(ticker)
    ↓
Backend: /api/v1/news/{ticker}
    ↓
Service: news_service.py
    ↓
External: News API + Sentiment Analysis
    ↓
Response: {articles: [...], sentiment: {...}}
    ↓
Frontend: HypeWarningCard displays alerts
```

**Code Path:**
1. Component render → `web/app/analyze/[ticker]/news-section.tsx`
2. API call → `web/lib/api.ts` → `GET /api/v1/news/{ticker}`
3. Backend router → `api/routers/news.py` → `/news/{ticker}`
4. Service → `api/services/enhanced_news_service.py`
5. External API → News API (fetch articles)
6. Sentiment analysis → `api/services/advanced_hype_detector.py`
7. Hype detection → Analyze text for hype indicators
8. Response → JSON with articles + sentiment scores
9. Frontend → `web/components/hype-warning-card.tsx` displays warnings

---

### 6. Portfolio Management Flow

**Files Involved:**
- Frontend: `web/app/portfolio/page.tsx`
- Frontend: `web/app/portfolio/portfolio-dashboard.tsx`
- Frontend: `web/app/portfolio/portfolio-setup.tsx`
- Frontend: `web/lib/api.ts` → `calculatePortfolioMetrics()`, `getRiskAnalysis()`
- Backend: `api/routers/portfolio.py`
- Backend: `api/services/black_scholes.py` (valuation)
- Database: `billions.db` → User holdings, preferences

**Flow:**
```
User on /portfolio page
    ↓
Frontend: PortfolioDashboard component
    ↓
API Calls:
  - api.calculatePortfolioMetrics()
  - api.getRiskAnalysis()
  - api.getPositions()
    ↓
Backend: portfolio.py router
    ↓
Service: Portfolio calculations
    ↓
Database: User holdings, preferences
    ↓
Response: Portfolio metrics, allocations
    ↓
Frontend: Dashboard displays charts & metrics
```

**Code Path:**
1. Page load → `web/app/portfolio/page.tsx`
2. Dashboard component → `web/app/portfolio/portfolio-dashboard.tsx`
3. API calls → `web/lib/api.ts`:
   - `POST /api/v1/portfolio/calculate-metrics`
   - `GET /api/v1/portfolio/risk-analysis/{ticker}`
   - `GET /api/v1/trading/positions`
4. Backend routers:
   - `api/routers/portfolio.py` → `/calculate-metrics`
   - `api/routers/trading.py` → `/positions`
5. Services:
   - `api/services/black_scholes.py` (valuation)
   - Portfolio calculations
6. Database → `billions.db` → Query user holdings
7. Response → JSON with metrics, allocations, risk analysis
8. Frontend → Render charts and metrics

---

## API Endpoint Categories with File References

### Market Data
- `GET /api/v1/{ticker}/historical`
  - Backend: `api/routers/historical.py`
  - Service: `api/services/market_data.py`
  - Frontend: `web/components/charts/candlestick-prediction-chart.tsx`

- `GET /api/v1/market/outliers/{strategy}`
  - Backend: `api/routers/market.py`
  - Service: `api/services/outlier_detection.py`
  - Frontend: `web/hooks/use-outliers.ts`

- `GET /api/v1/market/performance/{strategy}`
  - Backend: `api/routers/market.py`
  - Service: `api/services/outlier_detection.py`
  - Frontend: `web/hooks/use-performance-metrics.ts`

### Predictions
- `GET /api/v1/predictions/{ticker}`
  - Backend: `api/routers/predictions.py`
  - Service: `api/services/predictions.py`
  - ML Models: `funda/model/*.pt`
  - Frontend: `web/hooks/use-prediction.ts`

- `GET /api/v1/predictions/info/{ticker}`
  - Backend: `api/routers/predictions.py`
  - Service: `api/services/market_data.py`
  - Frontend: `web/hooks/use-ticker-info.ts`

- `GET /api/v1/predictions/search`
  - Backend: `api/routers/predictions.py`
  - Frontend: `web/components/ticker-search.tsx`

### Trading
- `GET /api/v1/trading/status`
  - Backend: `api/routers/trading.py`
  - Service: `api/services/trading_service.py`
  - Frontend: `web/app/trading/hft/page.tsx`

- `POST /api/v1/trading/execute`
  - Backend: `api/routers/trading.py`
  - Service: `api/services/trading_service.py`
  - External: Alpaca Trading API

- `GET /api/v1/trading/positions`
  - Backend: `api/routers/trading.py`
  - Service: `api/services/trading_service.py`
  - Frontend: `web/app/portfolio/portfolio-dashboard.tsx`

- `POST /api/v1/trading/quote/{symbol}`
  - Backend: `api/routers/trading.py`
  - Service: `api/services/trading_service.py`
  - Frontend: `web/hooks/use-hft-quotes.ts`

### HFT
- `GET /api/v1/hft/status`
  - Backend: `api/routers/hft.py`
  - Manager: `alpaca_websocket_hft_manager.py`
  - Frontend: `web/app/trading/hft/page.tsx`

- `POST /api/v1/hft/start`
  - Backend: `api/routers/hft.py`
  - Manager: `alpaca_websocket_hft_manager.py`

- `POST /api/v1/hft/orders`
  - Backend: `api/routers/hft.py`
  - Service: `api/services/trading_service.py`
  - Frontend: `web/app/trading/hft/page.tsx`

- `GET /api/v1/hft/performance`
  - Backend: `api/routers/hft.py`
  - Frontend: `web/app/trading/hft/page.tsx`

### News & Analysis
- `GET /api/v1/news/{ticker}`
  - Backend: `api/routers/news.py`
  - Service: `api/services/enhanced_news_service.py`
  - Frontend: `web/app/analyze/[ticker]/news-section.tsx`

- `GET /api/v1/nasdaq-news/latest`
  - Backend: `api/routers/nasdaq_news.py`
  - Service: `api/services/nasdaq_news_service.py`
  - Frontend: `web/components/nasdaq-news-section.tsx`

- `GET /api/v1/valuation/{ticker}`
  - Backend: `api/routers/valuation.py`
  - Service: `api/services/black_scholes.py`
  - Frontend: `web/components/fair-value-card.tsx`

### User Management
- `POST /api/v1/users/`
  - Backend: `api/routers/users.py`
  - Database: `db/models_auth.py` → `User` model

- `GET /api/v1/users/{id}/watchlist`
  - Backend: `api/routers/users.py`
  - Database: `db/models_auth.py` → `Watchlist` model

- `POST /api/v1/users/{id}/watchlist`
  - Backend: `api/routers/users.py`
  - Database: `db/models_auth.py` → `Watchlist` model

---

## Data Flow Summary

1. **User Interaction** → Frontend component (`web/app/` or `web/components/`)
2. **Component** → API client (`web/lib/api.ts`)
3. **API Client** → HTTP request to backend (`http://localhost:8000`)
4. **Backend Router** → Service layer (`api/routers/` → `api/services/`)
5. **Service** → Database query (`api/database.py` → `billions.db`) OR External API OR ML model (`funda/`)
6. **Response** → Backend router → JSON response
7. **Frontend** → Update UI with data (React state/hooks)

---

## Key File Locations

### Frontend Core Files
- **Entry Point**: `web/app/layout.tsx`
- **API Client**: `web/lib/api.ts`
- **Authentication**: `web/auth.ts`
- **Main Pages**: `web/app/*/page.tsx`
- **Components**: `web/components/`
- **Hooks**: `web/hooks/`
- **Types**: `web/types/`

### Backend Core Files
- **Entry Point**: `api/main.py`
- **Configuration**: `api/config.py`
- **Database**: `api/database.py`
- **Routers**: `api/routers/*.py`
- **Services**: `api/services/*.py`
- **Models**: `db/models.py`, `db/models_auth.py`

### ML & Data Files
- **LSTM Models**: `funda/model/*.pt`
- **Outlier Engines**: `funda/outlier_engine.py`, `outlier/Outlier_Nasdaq_*.py`
- **Training**: `funda/train_lstm_model.py`
- **Cache**: `funda/cache/*.csv`

### Trading Files
- **HFT Manager**: `alpaca_websocket_hft_manager.py`
- **Trading Manager**: `hft_trading_manager.py`
- **Trading Service**: `api/services/trading_service.py`

---

## Key Technologies

**Frontend:**
- Next.js 15 (React framework) - `web/package.json`
- TypeScript - `web/tsconfig.json`
- Tailwind CSS - `web/tailwind.config.ts`
- NextAuth (authentication) - `web/auth.ts`
- Custom hooks for data fetching - `web/hooks/`

**Backend:**
- FastAPI (Python web framework) - `api/main.py`
- SQLAlchemy (ORM) - `db/core.py`
- SQLite (database) - `billions.db`
- LSTM models (PyTorch) - `funda/model/`
- WebSocket support (Alpaca) - `alpaca_websocket_hft_manager.py`

**Communication:**
- REST API (JSON) - `web/lib/api.ts` ↔ `api/routers/`
- CORS enabled - `api/main.py` → CORS Middleware
- WebSocket (for real-time HFT data) - `web/hooks/use-hft-quotes.ts`

---

## Port Configuration

- **Frontend**: `http://localhost:3000` (Next.js dev server)
- **Backend**: `http://localhost:8000` (FastAPI/Uvicorn)
- **Database**: `billions.db` (SQLite file)

---

## Environment Variables

**Frontend** (`web/.env.local`):
- `NEXT_PUBLIC_API_URL=http://localhost:8000`

**Backend** (`.env`):
- `DATABASE_URL=sqlite:///./billions.db`
- `ALPACA_API_KEY=...`
- `ALPACA_SECRET_KEY=...`
- `NEWS_API_KEY=...`
- `OPENAI_API_KEY=...`
- `ALPHA_VANTAGE_API_KEY=...`
- `POLYGON_API_KEY=...`

---

This architecture supports real-time stock analysis, ML predictions, trading, and portfolio management with a clear separation between frontend and backend.

