"""
BILLIONS FastAPI Backend
Main application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.config import settings
from api.database import init_db
from api.routers import market, users, predictions, outliers, news, historical, valuation, portfolio, trading, capitulation, hft, nasdaq_news, behavioral

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
app.include_router(news.router, prefix=settings.API_V1_PREFIX)
app.include_router(historical.router, prefix=settings.API_V1_PREFIX)
app.include_router(valuation.router, prefix=settings.API_V1_PREFIX)
app.include_router(portfolio.router, prefix=settings.API_V1_PREFIX)
app.include_router(trading.router, prefix=settings.API_V1_PREFIX)
app.include_router(capitulation.router, prefix=settings.API_V1_PREFIX)
app.include_router(hft.router, prefix=settings.API_V1_PREFIX)
app.include_router(nasdaq_news.router, prefix=settings.API_V1_PREFIX)
app.include_router(behavioral.router, prefix=settings.API_V1_PREFIX)


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

@app.get("/api/v1/test-hype")
async def test_hype():
    """Test HYPE detection with sample data"""
    try:
        from api.routers.news import detect_hype_indicators, detect_caveat_emptor
        
        # Test with hype-filled news
        hype_news = "TSLA TO THE MOON! DIAMOND HANDS! This stock will SKYROCKET and make you RICH! GUARANTEED PROFITS! Don't miss out!"
        risk_news = "TSLA faces bankruptcy risk, SEC investigation ongoing, highly volatile penny stock, buyer beware!"
        
        hype_analysis = detect_hype_indicators(hype_news)
        caveat_analysis = detect_caveat_emptor(risk_news)
        
        return {
            "hype_news": {
                "text": hype_news,
                "analysis": hype_analysis
            },
            "risk_news": {
                "text": risk_news,
                "analysis": caveat_analysis
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/valuation/{ticker}/fair-value")
async def get_fair_value(ticker: str, days_back: int = 252):
    """Get Black-Scholes-Merton fair value analysis"""
    try:
        from api.services.black_scholes import bsm_analyzer
        ticker = ticker.upper()
        result = bsm_analyzer.analyze_stock_valuation(ticker, days_back)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Return simplified version
        return {
            "ticker": result["ticker"],
            "current_price": result["current_price"],
            "fair_value": result["fair_value"],
            "valuation_status": result["valuation_status"],
            "valuation_color": result["valuation_color"],
            "valuation_ratio": result["valuation_ratio"],
            "volatility": result["volatility"],
            "risk_free_rate": result["risk_free_rate"],
            "analysis_date": result["analysis_date"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

