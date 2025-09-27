"""
Enhanced Feature Engineering Module for 30-Day Predictions
Optimized feature set with improved signal-to-noise ratio and reduced redundancy
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_enhanced_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Enhanced RSI calculation with better handling of edge cases
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Smooth extreme values
    rsi = rsi.clip(0, 100)
    return rsi

def compute_realized_volatility_enhanced(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Enhanced realized volatility calculation with multiple timeframes
    """
    returns = df['Close'].pct_change().dropna()
    
    # Realized volatility with multiple components
    rv_squared = returns**2
    rv_sum = rv_squared.rolling(window=window).sum()
    
    # Add jump component (large price movements)
    jump_threshold = returns.rolling(window=window).std() * 3
    jumps = (np.abs(returns) > jump_threshold).astype(int)
    jump_component = (returns * jumps).rolling(window=window).sum()
    
    # Combine components
    realized_vol = np.sqrt(rv_sum + 0.1 * jump_component**2)
    
    return realized_vol

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multiple momentum indicators
    """
    df = df.copy()
    
    # Price momentum
    df['Momentum_5'] = df['Close'].pct_change(5)
    df['Momentum_10'] = df['Close'].pct_change(10)
    df['Momentum_20'] = df['Close'].pct_change(20)
    
    # Weighted momentum (recent changes have more weight)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    df['Weighted_Momentum'] = df['Close'].pct_change().rolling(4).apply(
        lambda x: np.sum(x * weights) if len(x) == 4 else np.nan
    )
    
    # Momentum acceleration
    df['Momentum_Accel'] = df['Momentum_10'].diff(5)
    
    return df

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced volume analysis features
    """
    df = df.copy()
    
    # Volume moving averages
    df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    
    # Volume ratios
    df['Volume_Ratio_10'] = df['Volume'] / df['Volume_SMA_10']
    df['Volume_Ratio_20'] = df['Volume'] / df['Volume_SMA_20']
    
    # Price-Volume relationship
    df['Price_Volume_Trend'] = df['Close'].pct_change() * df['Volume_Ratio_10']
    df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
    
    # Volume volatility
    df['Volume_Volatility'] = df['Volume'].rolling(10).std()
    df['Volume_Spike'] = (df['Volume'] > df['Volume'].rolling(20).quantile(0.9)).astype(int)
    
    return df

def compute_technical_indicators_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced technical indicators with better parameters
    """
    df = df.copy()
    
    # Moving averages with multiple timeframes
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    
    # Price position relative to moving averages
    df['Price_to_SMA10'] = df['Close'] / df['SMA_10']
    df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
    df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
    
    # Moving average slopes (trend strength)
    df['SMA20_Slope'] = df['SMA_20'].diff(5)
    df['SMA50_Slope'] = df['SMA_50'].diff(10)
    
    # RSI with multiple timeframes
    df['RSI_14'] = compute_enhanced_rsi(df['Close'], 14)
    df['RSI_7'] = compute_enhanced_rsi(df['Close'], 7)
    df['RSI_21'] = compute_enhanced_rsi(df['Close'], 21)
    
    # MACD with signal line and histogram
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands with position and squeeze
    bb_period = 20
    bb_std = df['Close'].rolling(bb_period).std()
    df['BB_Upper'] = df['SMA_20'] + (bb_std * 2)
    df['BB_Lower'] = df['SMA_20'] - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
    df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).mean()).astype(int)
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
    
    return df

def compute_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Market regime and volatility features
    """
    df = df.copy()
    
    # Volatility regime
    df['Realized_Vol_20'] = compute_realized_volatility_enhanced(df, 20)
    df['Vol_Regime'] = (df['Realized_Vol_20'] > df['Realized_Vol_20'].rolling(60).quantile(0.7)).astype(int)
    df['Vol_Spike'] = (df['Realized_Vol_20'] > df['Realized_Vol_20'].rolling(20).quantile(0.9)).astype(int)
    
    # Market trend classification
    df['Market_Trend'] = (df['SMA_20'] > df['SMA_50']).astype(int)
    df['Trend_Strength'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
    
    # Support and resistance levels
    df['Resistance'] = df['High'].rolling(20).max()
    df['Support'] = df['Low'].rolling(20).min()
    df['Price_Position'] = (df['Close'] - df['Support']) / (df['Resistance'] - df['Support'])
    
    # Volatility clustering
    df['Vol_Clustering'] = df['Realized_Vol_20'].rolling(5).mean()
    
    return df

def compute_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sector relative performance features
    """
    df = df.copy()
    
    # Create synthetic sector benchmark if not available
    if 'Sector_Close' not in df.columns:
        np.random.seed(42)  # For reproducibility
        df['Sector_Close'] = df['Close'] * (1 + np.random.normal(0, 0.01, len(df)).cumsum())
    
    # Sector relative strength
    df['Sector_Alpha'] = df['Close'].pct_change() - df['Sector_Close'].pct_change()
    df['Sector_Relative_Strength'] = df['Close'] / df['Sector_Close']
    df['Sector_RSI'] = compute_enhanced_rsi(df['Sector_Relative_Strength'], 14)
    
    # Sector momentum
    df['Sector_Momentum'] = df['Sector_Close'].pct_change(10)
    df['Sector_Outperformance'] = df['Momentum_10'] - df['Sector_Momentum']
    
    # Sector volatility
    df['Sector_Vol'] = df['Sector_Close'].rolling(20).std()
    df['Vol_Relative_to_Sector'] = df['Realized_Vol_20'] / df['Sector_Vol']
    
    return df

def enhanced_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Main enhanced feature engineering function
    """
    logger.info("Starting enhanced feature engineering...")
    
    # Create a copy to avoid modifying original
    df_enhanced = df.copy()
    
    # Basic price features
    df_enhanced['Price_Change'] = df_enhanced['Close'].pct_change()
    df_enhanced['Log_Returns'] = np.log(df_enhanced['Close'] / df_enhanced['Close'].shift(1))
    df_enhanced['High_Low_Ratio'] = df_enhanced['High'] / df_enhanced['Low']
    df_enhanced['Close_to_Open'] = df_enhanced['Close'] / df_enhanced['Open']
    
    # Apply all feature computation functions
    df_enhanced = compute_momentum_features(df_enhanced)
    df_enhanced = compute_volume_features(df_enhanced)
    df_enhanced = compute_technical_indicators_enhanced(df_enhanced)
    df_enhanced = compute_market_regime_features(df_enhanced)
    df_enhanced = compute_sector_relative_features(df_enhanced)
    
    # Define core features for model input
    core_features = [
        # Price features
        'Close', 'Volume', 'Price_Change', 'Log_Returns',
        'High_Low_Ratio', 'Close_to_Open',
        
        # Volatility features
        'Realized_Vol_20', 'Vol_Regime', 'Vol_Spike', 'Vol_Clustering',
        
        # Momentum features
        'Momentum_5', 'Momentum_10', 'Momentum_20', 'Weighted_Momentum', 'Momentum_Accel',
        
        # Volume features
        'Volume_Ratio_10', 'Volume_Ratio_20', 'Price_Volume_Trend', 'Volume_Spike',
        
        # Technical indicators
        'Price_to_SMA10', 'Price_to_SMA20', 'Price_to_SMA50',
        'SMA20_Slope', 'SMA50_Slope',
        'RSI_14', 'RSI_7', 'RSI_21',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'BB_Position', 'BB_Width', 'BB_Squeeze',
        'Stoch_K', 'Stoch_D',
        
        # Market regime
        'Market_Trend', 'Trend_Strength', 'Price_Position',
        
        # Sector relative
        'Sector_Alpha', 'Sector_Relative_Strength', 'Sector_Outperformance',
        'Vol_Relative_to_Sector'
    ]
    
    # Ensure original OHLC columns are preserved for model compatibility
    original_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    all_features = original_columns + core_features
    
    # Filter to available features (including original columns)
    available_features = [f for f in all_features if f in df_enhanced.columns]
    
    # Remove rows with NaN values
    df_processed = df_enhanced[available_features].dropna()
    
    logger.info(f"Enhanced feature engineering completed. Features: {len(available_features)}, Rows: {len(df_processed)}")
    logger.info(f"Features: {available_features}")
    
    return df_processed, available_features

def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'Close', 
                             top_k: int = 20) -> Dict[str, float]:
    """
    Analyze feature importance using multiple methods
    """
    logger.info("Analyzing feature importance...")
    
    try:
        # Prepare features and target
        feature_cols = [col for col in df.columns if col != target_col]
        
        if not feature_cols:
            logger.warning("No features available for analysis")
            return {}
        
        # Create target (future returns) - use simpler approach
        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found")
            return {}
        
        # Create target using simple forward returns
        target_returns = df[target_col].pct_change().shift(-1)  # Next day returns
        
        # Align features and target, remove NaN
        valid_idx = ~(df[feature_cols].isna().any(axis=1) | target_returns.isna())
        
        if valid_idx.sum() < 30:  # Need minimum samples
            logger.warning(f"Insufficient valid data: {valid_idx.sum()} samples")
            return {}
        
        X = df.loc[valid_idx, feature_cols]
        y = target_returns.loc[valid_idx]
        
        logger.info(f"Using {len(X)} samples and {len(feature_cols)} features for analysis")
        
        # Method 1: Simple correlation analysis (most reliable)
        corr_scores = {}
        for col in feature_cols:
            try:
                if X[col].var() > 0 and y.var() > 0:  # Both have variance
                    corr = abs(X[col].corr(y))
                    corr_scores[col] = corr if not np.isnan(corr) else 0
                else:
                    corr_scores[col] = 0
            except:
                corr_scores[col] = 0
        
        # Method 2: Variance-based importance
        var_scores = {}
        for col in feature_cols:
            try:
                var_score = X[col].var()
                var_scores[col] = var_score if not np.isnan(var_score) else 0
            except:
                var_scores[col] = 0
        
        # Method 3: Price proximity (for price-related features)
        price_scores = {}
        for col in feature_cols:
            try:
                if 'Price' in col or 'Close' in col or col == 'Volume':
                    # These are inherently important
                    price_scores[col] = 1.0
                elif 'Momentum' in col or 'RSI' in col or 'MACD' in col:
                    # Technical indicators
                    price_scores[col] = 0.8
                elif 'Vol' in col or 'Volatility' in col:
                    # Volatility features
                    price_scores[col] = 0.9
                else:
                    price_scores[col] = 0.5
            except:
                price_scores[col] = 0.5
        
        # Combine scores with weights
        combined_scores = {}
        for feature in feature_cols:
            corr_score = corr_scores.get(feature, 0)
            var_score = var_scores.get(feature, 0)
            price_score = price_scores.get(feature, 0)
            
            # Normalize variance score
            max_var = max(var_scores.values()) if var_scores.values() else 1
            normalized_var = var_score / max_var if max_var > 0 else 0
            
            # Weighted combination: 50% correlation, 30% variance, 20% domain knowledge
            combined_score = (0.5 * corr_score + 
                             0.3 * normalized_var + 
                             0.2 * price_score)
            
            combined_scores[feature] = combined_score
        
        # Sort by importance
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Top {min(top_k, len(sorted_features))} most important features:")
        for i, (feature, score) in enumerate(sorted_features[:top_k]):
            logger.info(f"{i+1:2d}. {feature:25s} - {score:.4f}")
        
        return dict(sorted_features)
        
    except Exception as e:
        logger.error(f"Error in feature importance analysis: {e}")
        return {}

def select_optimal_features(df: pd.DataFrame, target_col: str = 'Close', 
                          max_features: int = 25) -> List[str]:
    """
    Select optimal features based on importance analysis
    """
    try:
        importance_scores = analyze_feature_importance(df, target_col)
        
        if not importance_scores:
            logger.warning("No importance scores available, using core features")
            # Fallback to core features
            core_features = [
                'Close', 'Volume', 'Price_Change', 'Log_Returns',
                'Realized_Vol_20', 'Momentum_10', 'Volume_Ratio_10',
                'Price_to_SMA20', 'RSI_14', 'MACD', 'BB_Position',
                'Market_Trend', 'Sector_Alpha'
            ]
            available_core = [f for f in core_features if f in df.columns]
            return available_core[:max_features]
        
        # Select top features
        selected_features = list(importance_scores.keys())[:max_features]
        
        logger.info(f"Selected {len(selected_features)} optimal features")
        return selected_features
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        # Fallback to basic features
        basic_features = ['Close', 'Volume', 'Price_Change']
        available_basic = [f for f in basic_features if f in df.columns]
        return available_basic[:max_features]

if __name__ == "__main__":
    # Test the enhanced feature engineering
    import yfinance as yf
    
    # Download sample data
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="2y", interval="1d")
    
    # Add sector data
    sector_ticker = yf.Ticker("XLK")
    sector_df = sector_ticker.history(period="2y", interval="1d")
    df['Sector_Close'] = sector_df['Close'].reindex(df.index, method='ffill')
    
    # Apply enhanced feature engineering
    df_enhanced, features = enhanced_feature_engineering(df)
    
    print(f"Enhanced features shape: {df_enhanced.shape}")
    print(f"Features: {features}")
    
    # Analyze feature importance
    importance = analyze_feature_importance(df_enhanced)
    
    # Select optimal features
    optimal_features = select_optimal_features(df_enhanced, max_features=20)
    print(f"Optimal features: {optimal_features}")
