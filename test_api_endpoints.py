"""
Quick script to test BILLIONS API endpoints
Run this with the backend server running: python test_api_endpoints.py
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, description, expected_status=200, json_data=None, params=None):
    """Test an API endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Method: {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(url, params=params)
        elif method == "POST":
            response = requests.post(url, json=json_data, params=params)
        elif method == "PUT":
            response = requests.put(url, json=json_data)
        elif method == "DELETE":
            response = requests.delete(url)
        
        print(f"Status: {response.status_code} {'✅' if response.status_code == expected_status else '❌'}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)[:500]}...")
        else:
            print(f"Error: {response.text[:200]}")
            
        return response.status_code == expected_status
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Test all major API endpoints"""
    print("="*60)
    print("BILLIONS API Testing Suite")
    print("="*60)
    print("\nMake sure the backend is running on http://localhost:8000")
    print("Start it with: start-backend.bat")
    print("="*60)
    
    results = []
    
    # Health & Status
    results.append(test_endpoint("GET", "/", "Root endpoint"))
    results.append(test_endpoint("GET", "/health", "Health check"))
    results.append(test_endpoint("GET", "/api/v1/ping", "Ping endpoint"))
    
    # Market Data
    results.append(test_endpoint("GET", "/api/v1/market/outliers/swing", "Get swing outliers"))
    results.append(test_endpoint("GET", "/api/v1/market/performance/scalp", "Get scalp performance"))
    
    # ML Predictions
    results.append(test_endpoint("GET", "/api/v1/predictions/TSLA?days=5", "Get TSLA 5-day prediction"))
    results.append(test_endpoint("GET", "/api/v1/predictions/info/AAPL", "Get AAPL stock info"))
    results.append(test_endpoint("GET", "/api/v1/predictions/search?q=apple&limit=5", "Search tickers"))
    
    # Outlier Detection
    results.append(test_endpoint("GET", "/api/v1/outliers/strategies", "List strategies"))
    results.append(test_endpoint("GET", "/api/v1/outliers/swing/info", "Get swing strategy info"))
    
    # User Management
    test_user_data = {
        "id": "test_user_123",
        "email": "test@example.com",
        "name": "Test User"
    }
    results.append(test_endpoint("POST", "/api/v1/users/", "Create test user", 
                                 json_data=test_user_data))
    results.append(test_endpoint("GET", "/api/v1/users/test_user_123", "Get user"))
    results.append(test_endpoint("GET", "/api/v1/users/test_user_123/watchlist", "Get watchlist"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("✅ All API endpoints working!")
    else:
        print(f"⚠️  {total - passed} endpoint(s) failed")
        print("Note: Prediction endpoints may fail if LSTM model not loaded")
    
    print("\nFor detailed API documentation, visit: http://localhost:8000/docs")
    print("="*60)


if __name__ == "__main__":
    main()

