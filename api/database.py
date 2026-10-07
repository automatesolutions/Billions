"""
Database engine and session handling.

SQLite by default (a cache of computed outlier metrics). Set DATABASE_URL to
use PostgreSQL later; nothing here is SQLite-specific apart from connect_args.
"""

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from api.config import settings

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

if _is_sqlite:
    db_file = settings.DATABASE_URL.replace("sqlite:///", "", 1)
    if db_file and db_file != ":memory:":
        Path(db_file).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from api import models  # noqa: F401  (registers tables)

    Base.metadata.create_all(bind=engine)
