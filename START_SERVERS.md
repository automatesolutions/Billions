# 🚀 Start BILLIONS Locally - Quick Guide

## ✅ Step 1: Start Backend (Terminal 1)

Open **Terminal 1** (PowerShell) and run:

```powershell
# Make sure you're in the root directory
cd C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Billions

# Activate virtual environment
.venv\Scripts\activate

# Start backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

✅ **Backend running at**: http://localhost:8000  
✅ **API Docs**: http://localhost:8000/docs

---

## ✅ Step 2: Start Frontend (Terminal 2)

Open **Terminal 2** (PowerShell) and run:

```powershell
# Navigate to web directory
cd C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Billions\web

# Start frontend
pnpm dev
```

**Expected Output:**
```
  ▲ Next.js 15.5.4
  - Local:        http://localhost:3000
  - Network:      http://192.168.x.x:3000

 ✓ Starting...
 ✓ Ready in 2.5s
```

✅ **Frontend running at**: http://localhost:3000

---

## ✅ Step 3: Test the Application

### Test Backend (API)

Open browser and visit:
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Root**: http://localhost:8000/

**Expected Response** (health):
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Test Frontend (Web App)

1. **Open**: http://localhost:3000
2. **You should see**: Login page with Google OAuth button
3. **Without logging in**, you can check:
   - Homepage loads ✅
   - No console errors ✅

### Test Integration

In browser console (F12), test an API call:

```javascript
fetch('http://localhost:8000/health')
  .then(r => r.json())
  .then(console.log)
```

**Expected**: Should return health status

---

## 🧪 Quick API Tests (Optional)

Open a **Terminal 3** and test endpoints:

```powershell
# Test health
curl http://localhost:8000/health

# Test root
curl http://localhost:8000/

# Test outliers
curl http://localhost:8000/api/outliers/strategies

# Test predictions (replace TSLA with any ticker)
curl http://localhost:8000/api/predictions/info/TSLA
```

---

## ✅ Success Checklist

- [ ] Backend starts without errors
- [ ] Frontend starts without errors
- [ ] Can access http://localhost:8000/docs
- [ ] Can access http://localhost:3000
- [ ] No console errors in browser
- [ ] API health check returns JSON

---

## 🛑 Stop Servers

When done testing:

**Terminal 1 (Backend)**: Press `Ctrl+C`  
**Terminal 2 (Frontend)**: Press `Ctrl+C`

---

## ❌ Common Issues & Fixes

### Issue 1: "No module named uvicorn"
**Fix**: Install dependencies
```powershell
pip install -r api/requirements.txt
```

### Issue 2: "No package.json found" (pnpm)
**Fix**: Make sure you're in web/ directory
```powershell
cd web
pnpm dev
```

### Issue 3: "Port 8000 already in use"
**Fix**: Kill the process or use different port
```powershell
# Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Issue 4: Frontend can't connect to backend
**Fix**: Check CORS settings in `api/main.py` and ensure backend is running

---

## 📊 What You're Testing

### Backend (21 APIs working)
- ✅ Health check
- ✅ Predictions API
- ✅ Outliers API
- ✅ News API
- ✅ Users API
- ✅ Market data API

### Frontend (5 pages working)
- ✅ Login page
- ✅ Dashboard (requires auth)
- ✅ Analyze page (requires auth)
- ✅ Outliers page (requires auth)
- ✅ Portfolio page (requires auth)

---

**Ready to test!** 🚀

Open **2 terminals** and follow Steps 1 & 2 above.

