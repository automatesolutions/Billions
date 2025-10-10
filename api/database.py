"""
Database configuration and session management
Reuses existing SQLAlchemy models from db/
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import sys
from pathlib import Path

# Add parent directory to Python path to import db module
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from db.core import Base, engine as existing_engine, SessionLocal as ExistingSession
from db.models import PerfMetric
from db.models_auth import User, UserPreference, Watchlist, Alert
from api.config import settings

# Use the existing engine and session from db/core.py
engine = existing_engine
SessionLocal = ExistingSession


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI endpoints to get database session
    
    Usage:
        @app.get("/items")
        async def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

