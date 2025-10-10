"""
BILLIONS FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.config import settings
from api.database import init_db
from api.routers import market, users, predictions, outliers

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    logger.info("🚀 BILLIONS API starting up...")
    # Initialize database
    init_db()
    logger.info("✅ Database initialized")
    yield
    logger.info("👋 BILLIONS API shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Machine Learning API for Stock Market Forecasting and Outlier Detection",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(market.router, prefix=settings.API_V1_PREFIX)
app.include_router(users.router, prefix=settings.API_V1_PREFIX)
app.include_router(predictions.router, prefix=settings.API_V1_PREFIX)
app.include_router(outliers.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to BILLIONS API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "BILLIONS API",
        "version": "1.0.0"
    }


@app.get("/api/v1/ping")
async def ping():
    """Simple ping endpoint for connectivity testing"""
    return {"message": "pong"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

