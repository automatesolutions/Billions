#!/bin/bash

echo "Starting BILLIONS Backend API..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Start FastAPI server
echo "Backend API starting on http://localhost:8000"
echo "API Docs available at http://localhost:8000/docs"
echo ""
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

