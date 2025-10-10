"""
ML Prediction Service
Handles LSTM-based stock price predictions
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from funda.enhanced_features import enhanced_feature_engineering

logger = logging.getLogger(__name__)


class StockLSTM(nn.Module):
    """LSTM model for stock price prediction"""
    
    def __init__(self, input_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.permute(1, 0, 2)
        attn_output, _ = self.attention(out, out, out)
        context = attn_output[-1]
        out = self.fc(context)
        return out


class PredictionService:
    """Service for generating stock predictions"""
    
    def __init__(self):
        self.model_dir = Path(__file__).parent.parent.parent / "funda" / "model"
        self.cache_dir = Path(__file__).parent.parent.parent / "funda" / "cache"
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Prediction service initialized. Device: {self.device}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the trained LSTM model"""
        try:
            if model_path is None:
                model_path = self.model_dir / "lstm_daily_model.pt"
            
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Initialize model with default parameters
            self.model = StockLSTM(input_size=14, output_size=30)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def fetch_stock_data(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch stock data from yfinance"""
        try:
            logger.info(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d")
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Get sector data
            info = stock.info
            sector = info.get('sector', 'Technology')
            sector_map = {
                'Technology': 'XLK',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Industrials': 'XLI',
                'Utilities': 'XLU',
                'Healthcare': 'XLV',
                'Communication Services': 'XLC',
                'Consumer Defensive': 'XLP',
                'Basic Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Energy': 'XLE'
            }
            sector_ticker = sector_map.get(sector, 'SPY')
            
            # Fetch sector data
            sector_df = yf.Ticker(sector_ticker).history(period=period, interval="1d")
            df['Sector_Close'] = sector_df['Close'].reindex(df.index, method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
    
    def prepare_prediction_data(
        self, 
        df: pd.DataFrame, 
        seq_length: int = 60
    ) -> Tuple[Optional[np.ndarray], Optional[MinMaxScaler], Optional[MinMaxScaler], List[str]]:
        """Prepare data for prediction"""
        try:
            # Use enhanced feature engineering
            df_enhanced, features = enhanced_feature_engineering(df)
            
            if len(df_enhanced) < seq_length:
                logger.warning(f"Insufficient data: {len(df_enhanced)} rows, need {seq_length}")
                return None, None, None, []
            
            # Scale features
            feature_scaler = MinMaxScaler()
            data_scaled = feature_scaler.fit_transform(df_enhanced[features].values)
            
            # Scale target (Close price)
            target_scaler = MinMaxScaler()
            target_scaler.fit(df_enhanced[['Close']].values)
            
            # Get the last sequence for prediction
            X = data_scaled[-seq_length:].reshape(1, seq_length, -1)
            
            return X, feature_scaler, target_scaler, features
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None, []
    
    def generate_prediction(
        self, 
        ticker: str, 
        days: int = 30
    ) -> Optional[Dict]:
        """Generate stock price prediction"""
        try:
            # Load model if not already loaded
            if self.model is None:
                if not self.load_model():
                    return None
            
            # Fetch stock data
            df = self.fetch_stock_data(ticker)
            if df is None:
                return None
            
            # Prepare data
            X, feature_scaler, target_scaler, features = self.prepare_prediction_data(df)
            if X is None:
                return None
            
            # Generate prediction
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                prediction_scaled = self.model(X_tensor).cpu().numpy()
            
            # Inverse transform to get actual prices
            predictions = target_scaler.inverse_transform(
                prediction_scaled.reshape(-1, 1)
            ).flatten()[:days]
            
            # Get current price
            current_price = df['Close'].iloc[-1]
            
            # Calculate confidence intervals (simplified)
            volatility = df['Close'].pct_change().std()
            confidence_upper = predictions * (1 + volatility * 2)
            confidence_lower = predictions * (1 - volatility * 2)
            
            # Create response
            result = {
                "ticker": ticker,
                "current_price": float(current_price),
                "predictions": predictions.tolist(),
                "confidence_upper": confidence_upper.tolist(),
                "confidence_lower": confidence_lower.tolist(),
                "prediction_days": days,
                "model_features": len(features),
                "data_points": len(df),
                "last_updated": df.index[-1].isoformat(),
            }
            
            logger.info(f"Generated {days}-day prediction for {ticker}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {ticker}: {e}")
            return None


# Global service instance
prediction_service = PredictionService()

