"""
Pytest configuration and fixtures for BILLIONS API tests
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from api.database import get_db
from db.core import Base


# Create in-memory test database
@pytest.fixture
def test_db():
    """Create a test database that is destroyed after each test"""
    # Use in-memory SQLite database for tests
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    # Override the dependency
    app.dependency_overrides[get_db] = override_get_db
    
    yield TestingSessionLocal
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client(test_db):
    """Create a test client"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session(test_db):
    """Create a database session for tests"""
    session = test_db()
    yield session
    session.close()

