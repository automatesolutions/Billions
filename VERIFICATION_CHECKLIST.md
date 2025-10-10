# Phase 1 Verification Checklist

## ✅ Installation Complete

### Backend Dependencies
- ✅ Python virtual environment created (`venv/`)
- ✅ All Python packages installed from `api/requirements.txt`
- ✅ FastAPI, uvicorn, pydantic, SQLAlchemy, and all ML dependencies installed

### Frontend Dependencies  
- ✅ Next.js 15.5.4 installed
- ✅ TypeScript configured
- ✅ Tailwind CSS v4 configured
- ✅ shadcn/ui initialized with base components
- ✅ pnpm as package manager

## 🧪 Manual Testing Required

Since you're ready to test, here's what to verify:

### Step 1: Test Backend

**Open Terminal 1 and run:**
```bash
# Make sure virtual environment is activated
venv\Scripts\activate

# Start backend
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Will watch for changes in these directories: ['C:\\...\\Billions']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [####] using WatchFiles
🚀 BILLIONS API starting up...
✅ Database initialized
INFO:     Started server process [####]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Then test in browser:**
- http://localhost:8000 → Should show welcome JSON
- http://localhost:8000/docs → Should show Swagger UI
- http://localhost:8000/health → Should return `{"status": "healthy", "service": "BILLIONS API", "version": "1.0.0"}`

### Step 2: Test Frontend

**Open Terminal 2 and run:**
```bash
cd web
pnpm dev
```

**Expected Output:**
```
  ▲ Next.js 15.5.4
  - Local:        http://localhost:3000
  - Network:      http://[IP]:3000

 ✓ Starting...
 ✓ Ready in [time]
```

**Then test in browser:**
- http://localhost:3000 → Should show BILLIONS homepage
- **API Status badge should show "Connected ✓" in green**
- Should see system status card with backend info

## ✅ Success Criteria Verification

| Criteria | How to Verify | Expected Result |
|----------|---------------|-----------------|
| Backend starts on port 8000 | Visit http://localhost:8000 | JSON response with "Welcome to BILLIONS API" |
| Frontend starts on port 3000 | Visit http://localhost:3000 | BILLIONS homepage loads |
| OpenAPI docs accessible | Visit http://localhost:8000/docs | Interactive Swagger UI |
| API connection works | Check homepage status badge | Green "Connected ✓" badge |
| Hot reload works (backend) | Edit api/main.py, save | Server reloads automatically |
| Hot reload works (frontend) | Edit web/app/page.tsx, save | Page updates without refresh |
| ESLint configured | Run `cd web && pnpm lint` | No errors (or only warnings) |

## 🐛 If Backend Doesn't Start

**Check these:**
1. Is virtual environment activated? (You should see `(venv)` in prompt)
2. Is port 8000 available? 
   ```bash
   netstat -ano | findstr :8000
   ```
3. Are dependencies installed?
   ```bash
   pip list | findstr fastapi
   # Should show fastapi with version
   ```

## 🐛 If Frontend Doesn't Start

**Check these:**
1. Are you in the `web/` directory?
2. Are dependencies installed?
   ```bash
   cd web
   pnpm install
   ```
3. Is port 3000 available?
   ```bash
   netstat -ano | findstr :3000
   ```

## 📝 After Successful Verification

Once both are running and the API Status shows "Connected ✓":

### Test the API Endpoints

**In a new terminal or browser:**

1. **Health Check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Get Outliers (if data exists):**
   ```bash
   curl http://localhost:8000/api/v1/market/outliers/swing
   ```

3. **Get Performance Metrics:**
   ```bash
   curl http://localhost:8000/api/v1/market/performance/scalp
   ```

## ✅ Phase 1 Complete Checklist

- [ ] Backend starts without errors
- [ ] Frontend starts without errors  
- [ ] http://localhost:8000/docs shows Swagger UI
- [ ] http://localhost:3000 shows homepage
- [ ] API Status badge shows "Connected ✓"
- [ ] Can navigate to API docs from homepage
- [ ] Hot reload works on both sides
- [ ] No ESLint errors in frontend

---

**Once all checkboxes are complete, Phase 1 is fully verified and ready for Phase 2!**

## 📂 Quick Reference

### Start Backend
```bash
venv\Scripts\activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Or use the helper script:
```bash
start-backend.bat
```

### Start Frontend
```bash
cd web
pnpm dev
```

Or use the helper script:
```bash
start-frontend.bat
```

### View Logs
- Backend: Watch Terminal 1
- Frontend: Watch Terminal 2
- Browser Console: F12 → Console tab

---

**Ready to test? Open two terminals and let's verify everything works!** 🚀

