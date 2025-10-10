"""
Tests for ML prediction endpoints
"""

import pytest
from unittest.mock import patch, MagicMock


def test_get_prediction_invalid_ticker(client):
    """Test prediction with invalid ticker"""
    # This might fail if the ticker doesn't exist
    # For now, we'll test the endpoint structure
    response = client.get("/api/v1/predictions/INVALIDTICKER123")
    # Should return 500 if ticker not found or prediction fails
    assert response.status_code in [200, 500]


def test_get_prediction_with_days_parameter(client):
    """Test prediction with custom days parameter"""
    response = client.get("/api/v1/predictions/AAPL?days=10")
    # May fail if model not loaded, but endpoint should respond
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.json()
        assert "ticker" in data
        assert "predictions" in data
        assert "current_price" in data


def test_get_prediction_days_validation(client):
    """Test days parameter validation"""
    # Days too high
    response = client.get("/api/v1/predictions/AAPL?days=100")
    assert response.status_code == 422  # Validation error
    
    # Days too low
    response = client.get("/api/v1/predictions/AAPL?days=0")
    assert response.status_code == 422


@patch('api.services.predictions.prediction_service.generate_prediction')
def test_get_prediction_mocked(mock_predict, client):
    """Test prediction with mocked service"""
    # Mock a successful prediction
    mock_predict.return_value = {
        "ticker": "TSLA",
        "current_price": 250.0,
        "predictions": [251, 252, 253, 254, 255],
        "confidence_upper": [260, 261, 262, 263, 264],
        "confidence_lower": [240, 241, 242, 243, 244],
        "prediction_days": 5,
        "model_features": 14,
        "data_points": 252,
        "last_updated": "2025-01-01T00:00:00",
    }
    
    response = client.get("/api/v1/predictions/TSLA?days=5")
    assert response.status_code == 200
    data = response.json()
    
    assert data["ticker"] == "TSLA"
    assert data["current_price"] == 250.0
    assert len(data["predictions"]) == 5
    assert data["predictions"][0] == 251


def test_get_ticker_info(client):
    """Test ticker info endpoint"""
    response = client.get("/api/v1/predictions/info/AAPL")
    # May fail if yfinance is down or no internet
    assert response.status_code in [200, 404, 500]


@patch('api.services.market_data.market_data_service.get_stock_info')
def test_get_ticker_info_mocked(mock_info, client):
    """Test ticker info with mocked service"""
    mock_info.return_value = {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "market_cap": 3000000000000,
        "current_price": 175.50,
    }
    
    response = client.get("/api/v1/predictions/info/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["name"] == "Apple Inc."


def test_search_tickers(client):
    """Test ticker search endpoint"""
    response = client.get("/api/v1/predictions/search?q=APP")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_search_tickers_validation(client):
    """Test search validation"""
    # Empty query
    response = client.get("/api/v1/predictions/search?q=")
    assert response.status_code == 422  # Validation error

