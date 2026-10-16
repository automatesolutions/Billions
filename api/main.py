"""
BILLIONS API: read-only market intelligence. This service never places or routes trades.
The Quant page records the positions its model would take, only to score them.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.config import settings
from api.database import init_db
from api.limits import limiter
from api.routers import analysis, outliers, quant, quotes
from api.services import outliers as outlier_service
from api.services import quant as quant_service

logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.OUTLIER_SCHEDULER:
        outlier_service.start_scheduler()
    if settings.QUANT_SCHEDULER:
        quant_service.start_scheduler()
    yield
    outlier_service.stop_scheduler()
    quant_service.stop_scheduler()


app = FastAPI(
    title=settings.APP_NAME,
    description="Read-only outlier detection and per-stock quant analysis. Information only, not financial advice.",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limited(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Wait a minute, then try again."},
        headers={"Retry-After": "60"},
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on our side. Try again later."})


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(outliers.router, prefix=settings.API_V1_PREFIX)
app.include_router(analysis.router, prefix=settings.API_V1_PREFIX)
app.include_router(quotes.router, prefix=settings.API_V1_PREFIX)
app.include_router(quant.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
@limiter.exempt
def health_check():
    return {"status": "healthy", "service": settings.APP_NAME, "version": settings.VERSION}
