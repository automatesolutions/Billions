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
from api.services.markov_predictor import MarkovChainPredictor

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
        self.markov_predictor = MarkovChainPredictor(num_states=20)
        logger.info(f"Prediction service initialized. Device: {self.device}")
        
        # Try to load model immediately
        self.load_model()
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load the trained LSTM model"""
        try:
            if model_path is None:
                model_path = self.model_dir / "lstm_daily_model.pt"
            
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Initialize model with original features for compatibility
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
            df_enhanced, all_features = enhanced_feature_engineering(df)
            
            if len(df_enhanced) < seq_length:
                logger.warning(f"Insufficient data: {len(df_enhanced)} rows, need {seq_length}")
                return None, None, None, []
            
            # Use original 14 features that the model was trained on
            model_features = [
                'Close', 'Volume', 'Price_Change', 'Log_Returns', 
                'Momentum_10', 'Momentum_20', 
                'Volume_Ratio_20', 
                'Price_to_SMA20', 'SMA20_Slope',
                'RSI_14', 'MACD', 'MACD_Signal', 
                'BB_Position', 'Sector_Alpha'
            ]
            
            # Verify all required features exist
            missing_features = [f for f in model_features if f not in df_enhanced.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                return None, None, None, []
            
            # Scale features (only the 14 model expects)
            feature_scaler = MinMaxScaler()
            data_scaled = feature_scaler.fit_transform(df_enhanced[model_features].values)
            
            # Scale target (Close price)
            target_scaler = MinMaxScaler()
            target_scaler.fit(df_enhanced[['Close']].values)
            
            # Get the last sequence for prediction
            X = data_scaled[-seq_length:].reshape(1, seq_length, -1)
            
            logger.info(f"Prepared data shape: {X.shape} (expected: [1, 60, 14])")
            
            return X, feature_scaler, target_scaler, model_features
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None, []
    
    def generate_prediction(
        self, 
        ticker: str, 
        days: int = 30
    ) -> Optional[Dict]:
        """Generate stock price prediction using hybrid LSTM + Markov approach"""
        try:
            logger.info(f"Generating hybrid prediction for {ticker}, {days} days")
            
            # Use hybrid prediction method
            result = self.generate_hybrid_prediction(ticker, days)
            
            if result is None:
                logger.warning(f"Hybrid prediction failed for {ticker}, using fallback")
                return self._generate_fallback_prediction(ticker, days)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating prediction for {ticker}: {e}")
            return None
    
    def _generate_fallback_prediction(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """Generate fallback prediction when model fails"""
        try:
            logger.info(f"Generating fallback prediction for {ticker}")
            
            # Fetch basic stock data
            df = self.fetch_stock_data(ticker)
            if df is None or len(df) < 30:
                return None
            
            current_price = df['Close'].iloc[-1]
            
            # Simple trend-based prediction
            recent_trend = df['Close'].pct_change(20).iloc[-1]  # 20-day trend
            volatility = df['Close'].pct_change().std()
            
            # Generate predictions based on trend
            predictions = []
            for i in range(1, days + 1):
                # Simple linear trend with some randomness
                trend_factor = recent_trend * (i / 30)  # Scale trend over time
                random_factor = np.random.normal(0, volatility * 0.5)
                predicted_price = current_price * (1 + trend_factor + random_factor)
                predictions.append(max(predicted_price, current_price * 0.5))  # Floor at 50% of current
            
            predictions = np.array(predictions)
            
            # Calculate confidence intervals
            confidence_range = predictions * volatility * 2
            confidence_upper = predictions + confidence_range
            confidence_lower = predictions - confidence_range
            
            return {
                "ticker": ticker,
                "current_price": float(current_price),
                "predictions": predictions.tolist(),
                "confidence_upper": confidence_upper.tolist(),
                "confidence_lower": confidence_lower.tolist(),
                "prediction_days": days,
                "model_features": 0,  # Fallback
                "data_points": len(df),
                "last_updated": df.index[-1].isoformat(),
                "prediction_type": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Error in fallback prediction for {ticker}: {e}")
            return None
    
    def generate_hybrid_prediction(self, ticker: str, days: int = 30) -> Optional[Dict]:
        """
        Generate hybrid prediction using LSTM + Markov Chain
        
        Args:
            ticker: Stock symbol
            days: Number of days to predict
            
        Returns:
            Dictionary with hybrid predictions
        """
        try:
            logger.info(f"Generating hybrid prediction for {ticker}, {days} days")
            
            # Fetch stock data
            df = self.fetch_stock_data(ticker)
            if df is None or len(df) < 60:
                logger.warning("Insufficient data for hybrid prediction, using fallback")
                return self._generate_fallback_prediction(ticker, days)
            
            current_price = df['Close'].iloc[-1]
            
            # 1. Generate LSTM prediction
            lstm_result = None
            try:
                if self.model is not None:
                    X, feature_scaler, target_scaler, features = self.prepare_prediction_data(df)
                    if X is not None:
                        with torch.no_grad():
                            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                            prediction_scaled = self.model(X_tensor).cpu().numpy()
                        
                        lstm_predictions = target_scaler.inverse_transform(
                            prediction_scaled.reshape(-1, 1)
                        ).flatten()[:days]
                        lstm_result = lstm_predictions
                        logger.info("LSTM prediction generated successfully")
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")
            
            # 2. Generate Markov Chain prediction
            markov_predictions = None
            markov_upper = None
            markov_lower = None
            try:
                # Build Markov chain from historical data
                self.markov_predictor.build_transition_matrix(df['Close'].values)
                
                # Generate Markov predictions
                markov_predictions, markov_upper, markov_lower = self.markov_predictor.predict_with_uncertainty(
                    current_price, days
                )
                logger.info("Markov chain prediction generated successfully")
            except Exception as e:
                logger.warning(f"Markov prediction failed: {e}")
            
            # 3. Combine predictions
            if lstm_result is not None and markov_predictions is not None:
                # Hybrid approach: weighted average
                lstm_weight = 0.6  # LSTM gets more weight for trend
                markov_weight = 0.4  # Markov for pattern recognition
                
                hybrid_predictions = []
                for i in range(days):
                    hybrid_pred = (lstm_weight * lstm_result[i] + 
                                 markov_weight * markov_predictions[i])
                    hybrid_predictions.append(hybrid_pred)
                
                # Calculate much more visible confidence intervals
                prediction_std = np.std(hybrid_predictions) * 0.4  # Much wider: 40% of std
                # Ensure minimum confidence interval for visibility
                min_confidence = np.array(hybrid_predictions) * 0.05  # At least 5% of price
                prediction_std = np.maximum(prediction_std, min_confidence)
                
                confidence_upper = np.array(hybrid_predictions) + prediction_std
                confidence_lower = np.array(hybrid_predictions) - prediction_std
                
                method = "hybrid_lstm_markov"
                
            elif lstm_result is not None:
                # Use LSTM only with much more visible confidence
                hybrid_predictions = lstm_result.tolist()
                prediction_std = np.std(hybrid_predictions) * 0.5  # 50% of std for maximum visibility
                # Ensure minimum confidence interval for visibility
                min_confidence = np.array(hybrid_predictions) * 0.05  # At least 5% of price
                prediction_std = np.maximum(prediction_std, min_confidence)
                
                confidence_upper = np.array(hybrid_predictions) + prediction_std
                confidence_lower = np.array(hybrid_predictions) - prediction_std
                method = "lstm_only"
                
            elif markov_predictions is not None:
                # Use Markov only
                hybrid_predictions = markov_predictions
                confidence_upper = markov_upper
                confidence_lower = markov_lower
                method = "markov_only"
                
            else:
                # Fallback to simple trend-based prediction
                logger.warning("Both LSTM and Markov failed, using fallback")
                return self._generate_fallback_prediction(ticker, days)
            
            # Apply sentiment adjustment if available
            try:
                from api.routers.news import get_news_sentiment
                sentiment_score = get_news_sentiment(ticker)
                
                if abs(sentiment_score) > 0.1:  # Only apply if significant sentiment
                    sentiment_multiplier = 1 + (sentiment_score * 0.03)  # Reduced to 3% max
                    hybrid_predictions = [p * sentiment_multiplier for p in hybrid_predictions]
                    confidence_upper = [u * sentiment_multiplier for u in confidence_upper]
                    confidence_lower = [l * sentiment_multiplier for l in confidence_lower]
                    
                    logger.info(f"Applied sentiment adjustment: {sentiment_score:.3f}")
            except Exception as e:
                logger.warning(f"Could not apply sentiment adjustment: {e}")
            
            # Create response
            result = {
                "ticker": ticker,
                "current_price": float(current_price),
                "predictions": hybrid_predictions,
                "confidence_upper": confidence_upper.tolist() if isinstance(confidence_upper, np.ndarray) else confidence_upper,
                "confidence_lower": confidence_lower.tolist() if isinstance(confidence_lower, np.ndarray) else confidence_lower,
                "prediction_days": days,
                "model_features": len(features) if 'features' in locals() else 0,
                "data_points": len(df),
                "last_updated": df.index[-1].isoformat(),
                "prediction_method": method,
                "lstm_available": lstm_result is not None,
                "markov_available": markov_predictions is not None
            }
            
            logger.info(f"Generated hybrid {days}-day prediction for {ticker} using {method}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating hybrid prediction for {ticker}: {e}")
            return self._generate_fallback_prediction(ticker, days)


# Global service instance
prediction_service = PredictionService()

