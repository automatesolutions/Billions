@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ==============================
echo Creating backend .env file...
echo ==============================

REM Use existing environment variables if present; otherwise fall back to placeholders/defaults
set "POLY=%POLYGON_API_KEY%"
if "!POLY!"=="" set "POLY=N2_qdZVLhl1Cb7Xw5s0aNkcZj18MUp36"

set "ALP_KEY=%ALPACA_API_KEY%"
if "!ALP_KEY!"=="" set "ALP_KEY=PKNI1HFGNF44K7JCQSVR"

set "ALP_SECRET=%ALPACA_SECRET_KEY%"
if "!ALP_SECRET!"=="" set "ALP_SECRET=jverSR18LpEpQp43m3jBqBrZ6dhJVrEeVBagT0AT"

set "ALP_BASE=%ALPACA_BASE_URL%"
if "!ALP_BASE!"=="" set "ALP_BASE=https://paper-api.alpaca.markets"

(
echo # BILLIONS Backend Environment
echo POLYGON_API_KEY=!POLY!
echo ALPACA_API_KEY=!ALP_KEY!
echo ALPACA_SECRET_KEY=!ALP_SECRET!
echo ALPACA_BASE_URL=!ALP_BASE!
echo HFT_EDGE_THRESHOLD=0.001
echo HFT_MAX_POSITION_SIZE=1000
echo HFT_MAX_DAILY_LOSS=5000.0
echo HFT_MAX_LEVERAGE=2.0
echo DEBUG=true
) > .env

if exist .env (
  echo ✅ Created .env at project root
) else (
  echo ❌ Failed to create .env (check permissions)
)

echo.
echo ==============================
echo Creating frontend web\.env.local...
echo ==============================

cd web
(
echo NEXTAUTH_URL=http://localhost:3000
echo NEXTAUTH_SECRET=billions-dev-secret-12345
echo NEXT_PUBLIC_API_URL=http://localhost:8000
) > .env.local

if exist .env.local (
  echo ✅ Created web\.env.local
) else (
  echo ❌ Failed to create web\.env.local (check permissions)
)

echo.
echo Done. Next steps:
echo  1^) Restart backend:  python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
echo  2^) Restart frontend: cd web ^&^& pnpm dev
echo.
pause
endlocal

