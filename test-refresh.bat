@echo off
echo Testing BILLIONS Refresh API...
echo.

echo 1. Testing if backend is running...
curl -s http://localhost:8000/health
echo.
echo.

echo 2. Testing refresh status endpoint...
curl -s http://localhost:8000/api/v1/market/refresh/status
echo.
echo.

echo 3. Checking if we can reach the market API...
curl -s http://localhost:8000/api/v1/market/performance/swing
echo.
echo.

echo Test complete!
pause

