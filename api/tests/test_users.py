"""
Tests for user management endpoints
"""

import pytest
from db.models_auth import User, UserPreference, Watchlist


def test_create_user(client, db_session):
    """Test creating a new user"""
    user_data = {
        "id": "google_12345",
        "email": "test@example.com",
        "name": "Test User",
        "image": "https://example.com/avatar.jpg"
    }
    
    response = client.post("/api/v1/users/", json=user_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == "google_12345"
    assert data["email"] == "test@example.com"
    assert data["name"] == "Test User"
    assert data["role"] == "free"
    assert data["is_active"] is True


def test_create_user_creates_preferences(client, db_session):
    """Test that creating user also creates default preferences"""
    user_data = {
        "id": "google_67890",
        "email": "user@example.com",
        "name": "Another User"
    }
    
    response = client.post("/api/v1/users/", json=user_data)
    assert response.status_code == 200
    
    # Check that preferences were created
    prefs = db_session.query(UserPreference).filter(
        UserPreference.user_id == "google_67890"
    ).first()
    
    assert prefs is not None
    assert prefs.theme == "dark"
    assert prefs.default_strategy == "swing"


def test_get_user(client, db_session):
    """Test getting user by ID"""
    # Create a user first
    user = User(
        id="google_test",
        email="get@example.com",
        name="Get User"
    )
    db_session.add(user)
    db_session.commit()
    
    response = client.get("/api/v1/users/google_test")
    assert response.status_code == 200
    
    data = response.json()
    assert data["id"] == "google_test"
    assert data["email"] == "get@example.com"


def test_get_user_not_found(client):
    """Test getting non-existent user"""
    response = client.get("/api/v1/users/nonexistent")
    assert response.status_code == 404


def test_get_user_preferences(client, db_session):
    """Test getting user preferences"""
    # Create user and preferences
    user = User(id="google_pref", email="pref@example.com")
    db_session.add(user)
    db_session.flush()
    
    prefs = UserPreference(
        user_id="google_pref",
        theme="light",
        default_strategy="scalp"
    )
    db_session.add(prefs)
    db_session.commit()
    
    response = client.get("/api/v1/users/google_pref/preferences")
    assert response.status_code == 200
    
    data = response.json()
    assert data["theme"] == "light"
    assert data["default_strategy"] == "scalp"


def test_update_user_preferences(client, db_session):
    """Test updating user preferences"""
    # Create user and preferences
    user = User(id="google_update", email="update@example.com")
    db_session.add(user)
    db_session.flush()
    
    prefs = UserPreference(user_id="google_update")
    db_session.add(prefs)
    db_session.commit()
    
    # Update preferences
    updates = {
        "theme": "light",
        "email_notifications": False,
        "risk_tolerance": "high"
    }
    
    response = client.put("/api/v1/users/google_update/preferences", json=updates)
    assert response.status_code == 200
    
    # Verify updates
    db_session.refresh(prefs)
    assert prefs.theme == "light"
    assert prefs.email_notifications is False
    assert prefs.risk_tolerance == "high"


def test_get_watchlist_empty(client, db_session):
    """Test getting empty watchlist"""
    user = User(id="google_watch", email="watch@example.com")
    db_session.add(user)
    db_session.commit()
    
    response = client.get("/api/v1/users/google_watch/watchlist")
    assert response.status_code == 200
    assert response.json() == []


def test_add_to_watchlist(client, db_session):
    """Test adding symbol to watchlist"""
    user = User(id="google_add", email="add@example.com")
    db_session.add(user)
    db_session.commit()
    
    response = client.post(
        "/api/v1/users/google_add/watchlist",
        params={
            "symbol": "TSLA",
            "name": "Tesla Inc",
            "notes": "Electric vehicles"
        }
    )
    assert response.status_code == 200
    assert "id" in response.json()
    
    # Verify it was added
    watchlist = db_session.query(Watchlist).filter(
        Watchlist.user_id == "google_add"
    ).all()
    assert len(watchlist) == 1
    assert watchlist[0].symbol == "TSLA"


def test_add_duplicate_to_watchlist(client, db_session):
    """Test adding duplicate symbol to watchlist"""
    user = User(id="google_dup", email="dup@example.com")
    db_session.add(user)
    db_session.flush()
    
    item = Watchlist(user_id="google_dup", symbol="AAPL")
    db_session.add(item)
    db_session.commit()
    
    # Try to add again
    response = client.post(
        "/api/v1/users/google_dup/watchlist",
        params={"symbol": "AAPL"}
    )
    assert response.status_code == 400
    assert "already in watchlist" in response.json()["detail"]


def test_remove_from_watchlist(client, db_session):
    """Test removing symbol from watchlist"""
    user = User(id="google_remove", email="remove@example.com")
    db_session.add(user)
    db_session.flush()
    
    item = Watchlist(user_id="google_remove", symbol="MSFT")
    db_session.add(item)
    db_session.commit()
    
    item_id = item.id
    
    response = client.delete(f"/api/v1/users/google_remove/watchlist/{item_id}")
    assert response.status_code == 200
    
    # Verify it was removed
    removed = db_session.query(Watchlist).filter(Watchlist.id == item_id).first()
    assert removed is None

