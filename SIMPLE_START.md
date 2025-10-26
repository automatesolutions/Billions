# ⚡ Simple Start Guide for BILLIONS

## ✅ The venv is already active!

Your terminal shows `(venv)` which means the virtual environment is working.

---

## 🚀 Start Backend (Terminal 1)

In your PowerShell terminal, run these commands:

```powershell
# Make sure you're in the root directory
cd C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Billions

# Start backend (venv is already active)
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**You should see:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

✅ **Backend is running!**

Leave this terminal open. Backend is now at: http://localhost:8000

---

## 🌐 Start Frontend (Terminal 2)

Open a **NEW PowerShell terminal** and run:

```powershell
# Go to web directory
cd C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Billions\web

# Start frontend
pnpm dev
```

**You should see:**
```
▲ Next.js 15.5.4
- Local:   http://localhost:3000

✓ Ready in 2.5s
```

✅ **Frontend is running!**

---

## 🎉 Test It!

Open your browser:

1. **Backend API**: http://localhost:8000/docs
2. **Frontend App**: http://localhost:3000

---

## 🛑 Stop Servers

Press `Ctrl+C` in each terminal to stop.

---

That's it! 🚀

