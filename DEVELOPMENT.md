# BILLIONS Development Guide

## Prerequisites

- **Node.js** 20+ and **pnpm** 9+
- **Python** 3.12+
- **Git**
- **Docker** (optional, for containerized development)

## Project Structure

```
Billions/
├── web/                    # Next.js frontend
│   ├── app/               # App router pages
│   ├── components/        # React components
│   ├── lib/               # Utilities and API client
│   ├── types/             # TypeScript types
│   └── public/            # Static assets
├── api/                   # FastAPI backend
│   ├── routers/           # API route handlers
│   ├── main.py            # FastAPI app entry
│   ├── config.py          # Configuration
│   ├── database.py        # Database setup
│   └── requirements.txt   # Python dependencies
├── db/                    # Database models (existing)
├── funda/                 # ML models and features (existing)
├── billions.db            # SQLite database
└── docker-compose.yml     # Docker setup
```

## Local Development Setup

### Option 1: Manual Setup (Recommended for Development)

#### 1. Setup Frontend

```bash
cd web
pnpm install
cp .env.local.example .env.local
pnpm dev
```

Frontend will be available at **http://localhost:3000**

#### 2. Setup Backend

```bash
# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt

# Create .env file
cp api/.env.example api/.env

# Run the backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend API will be available at **http://localhost:8000**
- API docs: **http://localhost:8000/docs**
- Health check: **http://localhost:8000/health**

### Option 2: Docker Compose (Simplified)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Environment Variables

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-here
```

### Backend (api/.env)
```env
DEBUG=true
DATABASE_URL=sqlite:///./billions.db
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
SECRET_KEY=your-secret-key
```

## Available Commands

### Frontend (web/)
```bash
pnpm dev          # Start development server
pnpm build        # Build for production
pnpm start        # Start production server
pnpm lint         # Run ESLint
pnpm lint:fix     # Fix ESLint errors
```

### Backend (api/)
```bash
# Development
uvicorn api.main:app --reload

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Run tests (Phase 2)
pytest
pytest --cov=api
```

## Testing the Setup

### 1. Check Frontend
- Navigate to http://localhost:3000
- You should see the BILLIONS homepage
- API status should show "Connected ✓"

### 2. Check Backend
- Navigate to http://localhost:8000/docs
- You should see the interactive API documentation
- Try the `/health` endpoint - should return `{"status": "healthy"}`

### 3. Test API Connection
```bash
# From the command line
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/ping
```

## Database

The application uses **SQLite** for development. The database file `billions.db` is located in the project root.

### Accessing Database
```bash
# Using SQLite CLI
sqlite3 billions.db

# List tables
.tables

# Query performance metrics
SELECT * FROM performance_metrics LIMIT 10;
```

### Database Models
Models are defined in `db/models.py` (existing) and reused by the API layer.

## Development Workflow

### 1. Create a New Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Frontend: Edit files in `web/`
- Backend: Edit files in `api/`
- Both auto-reload on file changes

### 3. Commit with Conventional Commits
```bash
git commit -m "feat: add user authentication"
git commit -m "fix: resolve API connection issue"
git commit -m "test: add unit tests for predictions"
```

### 4. Push and Create PR
```bash
git push origin feature/your-feature-name
```

## Troubleshooting

### Frontend Issues

**Problem**: `pnpm install` fails
```bash
# Clear pnpm cache
pnpm store prune
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

**Problem**: Port 3000 already in use
```bash
# Kill process on port 3000 (Windows)
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:3000 | xargs kill -9
```

### Backend Issues

**Problem**: Module not found errors
```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r api/requirements.txt
```

**Problem**: Database locked
```bash
# Stop all running instances
# Delete database lock file
rm billions.db-journal
```

**Problem**: Port 8000 already in use
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

### Database Issues

**Problem**: Tables don't exist
```bash
# Tables are auto-created on startup
# Restart the backend API
```

## Next Steps

- [ ] **Phase 2**: Setup testing infrastructure (pytest, Vitest, Playwright)
- [ ] **Phase 3**: Implement Google OAuth authentication
- [ ] **Phase 4**: Migrate ML prediction APIs
- [ ] **Phase 5**: Build frontend UI components

## Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [shadcn/ui Components](https://ui.shadcn.com/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [SQLAlchemy](https://docs.sqlalchemy.org/)

## Support

For questions or issues:
1. Check existing documentation
2. Review the PLAN.md for architectural decisions
3. Check GitHub Issues
4. Create a new issue with detailed information

---

**Happy Coding! 🚀**

