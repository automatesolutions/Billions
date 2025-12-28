"""
Tests for outlier detection endpoints
"""

import pytest


def test_get_strategies(client):
    """Test getting all strategies"""
    response = client.get("/api/v1/outliers/strategies")
    assert response.status_code == 200
    data = response.json()
    assert "strategies" in data
    assert isinstance(data["strategies"], list)
    assert len(data["strategies"]) == 3  # scalp, swing, longterm


def test_get_strategy_info_valid(client):
    """Test getting info for valid strategy"""
    for strategy in ["scalp", "swing", "longterm"]:
        response = client.get(f"/api/v1/outliers/{strategy}/info")
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == strategy
        assert "x_period" in data
        assert "y_period" in data
        assert "lookback_x_days" in data
        assert "min_market_cap" in data


def test_get_strategy_info_invalid(client):
    """Test getting info for invalid strategy"""
    response = client.get("/api/v1/outliers/invalid/info")
    assert response.status_code == 404


def test_refresh_outliers_invalid_strategy(client):
    """Test refreshing outliers with invalid strategy"""
    response = client.post("/api/v1/outliers/invalid/refresh")
    assert response.status_code == 400


def test_refresh_outliers_valid_strategy(client):
    """Test refreshing outliers with valid strategy"""
    # This test just checks the endpoint accepts the request
    # Actual refresh happens in background
    response = client.post("/api/v1/outliers/scalp/refresh")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "processing"
    assert "scalp" in data["message"]

