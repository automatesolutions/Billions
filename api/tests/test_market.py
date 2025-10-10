"""
Tests for market data endpoints
"""

import pytest
from db.models import PerfMetric


def test_get_outliers_invalid_strategy(client):
    """Test outliers endpoint with invalid strategy"""
    response = client.get("/api/v1/market/outliers/invalid")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid strategy" in data["detail"]


def test_get_outliers_valid_strategy_empty(client):
    """Test outliers endpoint with valid strategy but no data"""
    response = client.get("/api/v1/market/outliers/scalp")
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "scalp"
    assert data["count"] == 0
    assert data["outliers"] == []


def test_get_outliers_with_data(client, db_session):
    """Test outliers endpoint with sample data"""
    # Create test data
    metric = PerfMetric(
        strategy="swing",
        symbol="TEST",
        metric_x=10.5,
        metric_y=15.2,
        z_x=2.5,
        z_y=3.1,
        is_outlier=True
    )
    db_session.add(metric)
    db_session.commit()
    
    response = client.get("/api/v1/market/outliers/swing")
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "swing"
    assert data["count"] == 1
    assert len(data["outliers"]) == 1
    assert data["outliers"][0]["symbol"] == "TEST"
    assert data["outliers"][0]["is_outlier"] is True


def test_get_performance_metrics_invalid_strategy(client):
    """Test performance metrics with invalid strategy"""
    response = client.get("/api/v1/market/performance/invalid")
    assert response.status_code == 400


def test_get_performance_metrics_valid(client, db_session):
    """Test performance metrics endpoint"""
    # Create test data
    metric = PerfMetric(
        strategy="longterm",
        symbol="AAPL",
        metric_x=5.0,
        metric_y=7.0,
        z_x=1.0,
        z_y=1.5,
        is_outlier=False
    )
    db_session.add(metric)
    db_session.commit()
    
    response = client.get("/api/v1/market/performance/longterm")
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "longterm"
    assert data["count"] == 1
    assert len(data["metrics"]) == 1
    assert data["metrics"][0]["symbol"] == "AAPL"

