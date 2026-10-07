"""
BILLIONS API: read-only market intelligence. This service never places or simulates trades.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.database import init_db
from api.routers import outliers
from api.services import outliers as outlier_service

logging.basicConfig(level=logging.DEBUG if settings.DEBUG else logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if settings.OUTLIER_SCHEDULER:
        outlier_service.start_scheduler()
    yield
    outlier_service.stop_scheduler()


app = FastAPI(
    title=settings.APP_NAME,
    description="Read-only outlier detection and per-stock quant analysis. Information only, not financial advice.",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(outliers.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": settings.APP_NAME, "version": settings.VERSION}
