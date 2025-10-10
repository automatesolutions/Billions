# BILLIONS Web App

> Modern web application for stock market forecasting and outlier detection, powered by machine learning.

## 🚀 Quick Start

### Prerequisites
- Node.js 20+ and pnpm
- Python 3.12+
- Git

### Installation

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd Billions

# Install frontend dependencies
cd web
pnpm install
cd ..

# Create Python virtual environment
python -m venv venv
```

2. **Setup environment variables:**
```bash
# Copy example files
cp .env.example .env
cp api/.env.example api/.env
cp web/.env.local.example web/.env.local

# Edit the files with your API keys
```

3. **Start the application:**

**Option A: Using startup scripts (easiest)**
```bash
# Terminal 1 - Backend
start-backend.bat  # Windows
./start-backend.sh # macOS/Linux

# Terminal 2 - Frontend
start-frontend.bat  # Windows
./start-frontend.sh # macOS/Linux
```

**Option B: Manual start**
```bash
# Terminal 1 - Backend
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
python -m uvicorn api.main:app --reload

# Terminal 2 - Frontend
cd web
pnpm dev
```

**Option C: Docker Compose**
```bash
docker-compose up --build
```

### Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📚 Documentation

- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Complete development guide
- **[PLAN.md](PLAN.md)** - Project roadmap and architecture
- **[PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)** - Phase 1 completion summary

## 🏗️ Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Next.js       │         │   FastAPI       │
│   Frontend      │◄───────►│   Backend       │
│   Port 3000     │   API   │   Port 8000     │
└─────────────────┘         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │   SQLite DB     │
                            │   billions.db   │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │  ML Models      │
                            │  (PyTorch LSTM) │
                            └─────────────────┘
```

## 🎯 Features

### Current (Phase 1)
- ✅ Modern Next.js frontend with TypeScript
- ✅ FastAPI backend with auto-generated documentation
- ✅ Market data API endpoints (outliers, performance metrics)
- ✅ Health check and monitoring
- ✅ shadcn/ui component library
- ✅ Hot reload for development

### Coming Soon
- 🔄 Testing infrastructure (Phase 2)
- 🔄 Google OAuth authentication (Phase 3)
- 🔄 30-day ML predictions API (Phase 4)
- 🔄 Interactive dashboards and charts (Phase 5)
- 🔄 Deployment to Vercel (Phase 6)

## 🧪 Testing

Testing infrastructure will be set up in Phase 2. Once complete:

```bash
# Backend tests
pytest
pytest --cov=api

# Frontend tests  
cd web
pnpm test
pnpm test:e2e
```

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS v4
- **Components**: shadcn/ui
- **Build**: Turbopack

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.12
- **ORM**: SQLAlchemy
- **Database**: SQLite
- **ML**: PyTorch, TensorFlow, scikit-learn
- **Data**: yfinance, pandas, numpy

### DevOps
- **Package Manager**: pnpm (frontend), pip (backend)
- **Containerization**: Docker & Docker Compose
- **CI/CD**: GitHub Actions (Phase 6)
- **Deployment**: Vercel (frontend), Railway/Render (backend)
- **Monitoring**: Sentry (Phase 6)

## 📁 Project Structure

```
Billions/
├── web/              # Next.js frontend
├── api/              # FastAPI backend
├── db/               # Database models
├── funda/            # ML models and features
├── venv/             # Python virtual environment
├── billions.db       # SQLite database
└── docker-compose.yml
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

See [LICENSE](LICENSE) file.

## 🔗 Links

- [Original BILLIONS Project](../README.md)
- [Development Guide](DEVELOPMENT.md)
- [Project Plan](PLAN.md)

---

**Status**: Phase 1 Complete ✅ | Phase 2 In Progress 🔄

Built with ❤️ using Next.js and FastAPI

