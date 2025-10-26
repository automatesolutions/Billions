@echo off
echo Starting BILLIONS Backend...
echo.

cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Virtual environment activated.
echo.

echo Starting FastAPI server on http://localhost:8000
echo API Docs will be at http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
