# BILLIONS Web App - Complete Setup Instructions

## 🎯 Current Status: Phase 3 Complete!

**Progress**: 37.5% (3/8 phases complete)
- ✅ Phase 0: Foundation & Analysis
- ✅ Phase 1: Infrastructure Setup
- ✅ Phase 2: Testing Infrastructure
- ✅ Phase 3: Authentication & User Management
- ⏳ Phase 4-8: In Progress

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Backend dependencies (Python)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r api/requirements.txt
pip install -r api/requirements-dev.txt

# Frontend dependencies (Node.js)
cd web
pnpm install
cd ..
```

### Step 2: Setup Google OAuth (REQUIRED)

**Follow the complete guide**: [GOOGLE_OAUTH_SETUP.md](./GOOGLE_OAUTH_SETUP.md)

**Quick Steps**:
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project → Enable Google+ API
3. Create OAuth credentials
4. Add redirect URI: `http://localhost:3000/api/auth/callback/google`
5. Copy Client ID and Secret

### Step 3: Configure Environment Variables

```bash
# Copy templates
cp .env.example .env
cp api/.env.example api/.env
cp web/.env.local.example web/.env.local

# Edit web/.env.local with your Google OAuth credentials:
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=<generate-with-openssl-rand-base64-32>
GOOGLE_CLIENT_ID=<your-client-id>.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=<your-client-secret>
```

**Generate NEXTAUTH_SECRET:**
```powershell
# Windows PowerShell
[Convert]::ToBase64String((1..32 | ForEach-Object { Get-Random -Minimum 0 -Maximum 256 }))
```

### Step 4: Initialize Database

```bash
# Create auth tables
python -c "from db.core import Base, engine; from db.models_auth import User; Base.metadata.create_all(bind=engine); print('✅ Database ready!')"
```

### Step 5: Start the Application

**Terminal 1 - Backend:**
```bash
start-backend.bat      # Windows
# ./start-backend.sh   # macOS/Linux
```

**Terminal 2 - Frontend:**
```bash
start-frontend.bat     # Windows
# ./start-frontend.sh  # macOS/Linux
```

### Step 6: Test the Application

1. **Open**: http://localhost:3000
2. **Click**: "Sign In" button
3. **Expected**: Redirects to `/login`
4. **Click**: "Sign in with Google"
5. **Expected**: Google OAuth consent screen
6. **Grant Permission**
7. **Expected**: Redirects to `/dashboard`
8. **Verify**: You see your name and profile picture

## ✅ Verification Checklist

### Backend Verification
- [ ] Backend starts on http://localhost:8000
- [ ] API docs accessible at http://localhost:8000/docs
- [ ] Health check returns 200: http://localhost:8000/health
- [ ] All tests passing: `pytest` (19 tests)

### Frontend Verification
- [ ] Frontend starts on http://localhost:3000
- [ ] Homepage shows "Sign In" button
- [ ] Login page accessible at /login
- [ ] All tests passing: `cd web && pnpm vitest run` (9 tests)

### Authentication Verification
- [ ] Can access /login page
- [ ] Google OAuth button visible
- [ ] Protected routes redirect to /login
- [ ] Can sign in with Google (after OAuth setup)
- [ ] Dashboard accessible after login
- [ ] User profile displays correctly
- [ ] Can sign out successfully

## 🧪 Run Tests

```bash
# Backend tests
pytest -v
# Expected: 19 passed, 85% coverage

# Frontend unit tests
cd web && pnpm vitest run
# Expected: 9 passed

# E2E tests (requires app running)
cd web && pnpm test:e2e
# Expected: 8 tests configured

# All tests
pytest && cd web && pnpm vitest run
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [PLAN.md](./PLAN.md) | Master project plan |
| [DEVELOPMENT.md](./DEVELOPMENT.md) | Development guide |
| [README_TESTING.md](./README_TESTING.md) | Testing guide |
| [GOOGLE_OAUTH_SETUP.md](./GOOGLE_OAUTH_SETUP.md) | OAuth setup |
| [STATUS.md](./STATUS.md) | Project status |
| [PHASE1_SUMMARY.md](./PHASE1_SUMMARY.md) | Phase 1 details |
| [PHASE2_SUMMARY.md](./PHASE2_SUMMARY.md) | Phase 2 details |
| [PHASE3_SUMMARY.md](./PHASE3_SUMMARY.md) | Phase 3 details |

## 🎯 Current Features

### ✅ Implemented
- Full-stack infrastructure (Next.js + FastAPI)
- Comprehensive testing (28 tests)
- Google OAuth authentication
- User management system
- Protected routes
- User preferences
- Watchlist functionality
- CI/CD pipelines

### 🔜 Coming Next (Phase 4)
- 30-day ML predictions API
- Outlier detection (scalp, swing, longterm)
- Market data pipeline
- News & sentiment analysis

## 🐛 Troubleshooting

### OAuth Issues
See [GOOGLE_OAUTH_SETUP.md](./GOOGLE_OAUTH_SETUP.md) troubleshooting section

### Backend Won't Start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Reinstall dependencies
pip install -r api/requirements.txt
```

### Frontend Won't Start
```bash
# Check if port 3000 is in use
netstat -ano | findstr :3000

# Reinstall dependencies
cd web
rm -rf node_modules .next
pnpm install
```

### Tests Failing
```bash
# Backend: Make sure venv is activated
venv\Scripts\activate
pytest

# Frontend: Make sure in web directory
cd web
pnpm vitest run
```

## 📊 Project Metrics

- **Total Tests**: 28 tests (19 backend, 9 frontend)
- **Backend Coverage**: 85%
- **API Endpoints**: 12 endpoints
- **Database Tables**: 5 tables
- **Frontend Pages**: 4 pages (home, login, dashboard, error)
- **Documentation**: 1,500+ lines

## 🎉 You're Ready!

Once setup is complete:
- Start developing new features
- Run tests frequently
- Follow the PLAN.md for next phases
- Keep documentation updated

## 🆘 Need Help?

1. Check relevant documentation file
2. Review GitHub Issues
3. Check error logs in terminal
4. Run tests to identify issues

---

**Last Updated**: 2025-10-10
**Status**: Phases 1-3 Complete, Ready for Phase 4
**Next**: ML Backend Migration 🤖

