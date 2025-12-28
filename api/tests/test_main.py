"""
Tests for main API endpoints
"""

import pytest
from fastapi.testclient import TestClient


def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Welcome to BILLIONS API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "operational"


def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "BILLIONS API"
    assert data["version"] == "1.0.0"


def test_ping_endpoint(client):
    """Test the ping endpoint"""
    response = client.get("/api/v1/ping")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "pong"


def test_404_endpoint(client):
    """Test that invalid endpoints return 404"""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404

