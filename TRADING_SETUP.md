# Trading Integration Setup Guide

## Required API Keys

### 1. Polygon.io API Key
- **Purpose**: Real-time market data, quotes, and orderbook
- **Sign up**: https://polygon.io/
- **Pricing**: Free tier available with rate limits
- **Add to .env**: `POLYGON_API_KEY=your_key_here`

### 2. Alpaca API Keys
- **Purpose**: Paper trading and portfolio management
- **Sign up**: https://alpaca.markets/
- **Pricing**: Free paper trading account
- **Add to .env**:
  ```
  ALPACA_API_KEY=your_api_key_here
  ALPACA_SECRET_KEY=your_secret_key_here
  ```

## Environment Setup

Create a `.env` file in the root directory with:

```bash
# Trading API Keys
POLYGON_API_KEY=your_polygon_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
```

## Features Available

### Real-time Market Data (Polygon.io)
- Live stock quotes
- Orderbook data (bid/ask)
- Market status
- Volume data

### Paper Trading (Alpaca)
- Account information
- Position tracking
- Order execution
- Portfolio synchronization

### Trading Dashboard
- Real-time market data display
- Position management
- Order history
- Trade execution interface

## API Endpoints

### Trading Status
- `GET /api/v1/trading/status` - Check service status
- `GET /api/v1/trading/account` - Get account info
- `GET /api/v1/trading/portfolio` - Get portfolio data

### Market Data
- `POST /api/v1/trading/quote/{symbol}` - Get real-time quote
- `POST /api/v1/trading/orderbook/{symbol}` - Get orderbook
- `POST /api/v1/trading/market-data` - Bulk market data

### Trading Operations
- `POST /api/v1/trading/execute` - Execute trades
- `GET /api/v1/trading/positions` - Get positions
- `GET /api/v1/trading/orders` - Get order history

## Security Notes

- **Paper Trading Only**: Default configuration uses Alpaca's paper trading
- **API Key Security**: Never commit API keys to version control
- **Rate Limits**: Respect API rate limits to avoid service interruption
- **Error Handling**: All trading operations include error handling

## Testing

1. Start the backend: `python -m uvicorn api.main:app --reload`
2. Navigate to `/portfolio` in the frontend
3. Check the Trading Dashboard tab
4. Verify API connections in the Overview tab

## Next Steps

1. **Get API Keys**: Sign up for Polygon.io and Alpaca accounts
2. **Add Keys**: Add keys to your `.env` file
3. **Test Connection**: Verify services are connected
4. **Start Trading**: Use the Trading Dashboard for paper trading
