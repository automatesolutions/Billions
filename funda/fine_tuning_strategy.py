"""
Phase 3: Fine-tuning Strategy - Improve prediction accuracy for specific tickers
Based on model diagnostics, implement targeted improvements
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TickerSpecificFineTuner:
    """Fine-tune models for specific ticker characteristics"""
    
    def __init__(self):
        self.ticker_profiles = {
            'TSLA': {
                'category': 'high_volatility_growth',
                'characteristics': ['high_volatility', 'growth_stock', 'tech_leader'],
                'fine_tuning_params': {
                    'learning_rate': 0.001,
                    'sequence_length': 90,  # Longer sequence for volatile stocks
                    'regularization': 0.01,
                    'focus_features': ['Volume', 'Realized_Vol_20', 'RSI_14', 'MACD']
                }
            },
            'AAPL': {
                'category': 'stable_large_cap',
                'characteristics': ['stable', 'large_cap', 'dividend_stock'],
                'fine_tuning_params': {
                    'learning_rate': 0.0005,
                    'sequence_length': 60,
                    'regularization': 0.005,
                    'focus_features': ['Close', 'Price_to_SMA20', 'Volume', 'Price_Change']
                }
            },
            'MSFT': {
                'category': 'stable_large_cap',
                'characteristics': ['stable', 'large_cap', 'tech_leader'],
                'fine_tuning_params': {
                    'learning_rate': 0.0005,
                    'sequence_length': 60,
                    'regularization': 0.005,
                    'focus_features': ['Close', 'Price_to_SMA20', 'Volume', 'RSI_14']
                }
            },
            'NVDA': {
                'category': 'high_volatility_growth',
                'characteristics': ['high_volatility', 'growth_stock', 'tech_leader'],
                'fine_tuning_params': {
                    'learning_rate': 0.001,
                    'sequence_length': 75,
                    'regularization': 0.01,
                    'focus_features': ['Volume', 'Realized_Vol_20', 'RSI_14', 'Momentum_10']
                }
            },
            'GOOGL': {
                'category': 'stable_large_cap',
                'characteristics': ['stable', 'large_cap', 'tech_leader'],
                'fine_tuning_params': {
                    'learning_rate': 0.0005,
                    'sequence_length': 60,
                    'regularization': 0.005,
                    'focus_features': ['Close', 'Price_to_SMA20', 'Volume', 'Price_Change']
                }
            }
        }
    
    def analyze_ticker_characteristics(self, ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze ticker-specific characteristics for fine-tuning"""
        logger.info(f"Analyzing characteristics for {ticker}")
        
        # Calculate key metrics
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        avg_volume = df['Volume'].mean()
        price_range = (df['Close'].max() - df['Close'].min()) / df['Close'].mean()
        
        # Trend analysis
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        
        # Volatility clustering
        volatility_clustering = returns.rolling(20).std().std()
        
        characteristics = {
            'volatility': volatility,
            'avg_volume': avg_volume,
            'price_range': price_range,
            'trend': trend,
            'volatility_clustering': volatility_clustering,
            'data_points': len(df)
        }
        
        # Classify ticker type
        if volatility > 0.4:
            characteristics['type'] = 'high_volatility'
        elif volatility < 0.2:
            characteristics['type'] = 'low_volatility'
        else:
            characteristics['type'] = 'medium_volatility'
        
        if avg_volume > df['Volume'].quantile(0.8):
            characteristics['liquidity'] = 'high'
        elif avg_volume < df['Volume'].quantile(0.2):
            characteristics['liquidity'] = 'low'
        else:
            characteristics['liquidity'] = 'medium'
        
        return characteristics
    
    def recommend_fine_tuning_strategy(self, ticker: str, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend fine-tuning strategy based on ticker characteristics"""
        logger.info(f"Recommending fine-tuning strategy for {ticker}")
        
        strategy = {
            'learning_rate': 0.001,  # Default
            'sequence_length': 60,   # Default
            'regularization': 0.01,  # Default
            'focus_features': ['Close', 'Volume', 'RSI', 'MACD'],  # Default
            'training_epochs': 50,
            'batch_size': 32,
            'early_stopping_patience': 10
        }
        
        # Adjust based on volatility
        if characteristics['type'] == 'high_volatility':
            strategy['learning_rate'] = 0.001  # Higher LR for volatile stocks
            strategy['sequence_length'] = 90   # Longer sequence
            strategy['regularization'] = 0.015  # More regularization
            strategy['focus_features'] = ['Volume', 'Volatility', 'RSI', 'MACD', 'ATR']
            strategy['training_epochs'] = 75
        elif characteristics['type'] == 'low_volatility':
            strategy['learning_rate'] = 0.0005  # Lower LR for stable stocks
            strategy['sequence_length'] = 45    # Shorter sequence
            strategy['regularization'] = 0.005  # Less regularization
            strategy['focus_features'] = ['Close', 'SMA_20', 'Volume', 'Price_Change']
            strategy['training_epochs'] = 30
        
        # Adjust based on liquidity
        if characteristics['liquidity'] == 'high':
            strategy['batch_size'] = 64  # Larger batches for liquid stocks
        elif characteristics['liquidity'] == 'low':
            strategy['batch_size'] = 16  # Smaller batches for illiquid stocks
        
        # Adjust based on data availability
        if characteristics['data_points'] < 200:
            strategy['training_epochs'] = min(strategy['training_epochs'], 25)
            strategy['early_stopping_patience'] = 5
        
        return strategy
    
    def create_enhanced_features_for_ticker(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """Create ticker-specific enhanced features"""
        logger.info(f"Creating enhanced features for {ticker}")
        
        from enhanced_features import enhanced_feature_engineering
        
        # Apply standard enhanced features
        df_processed, enhanced_features = enhanced_feature_engineering(df)
        
        # Add ticker-specific features
        if ticker in self.ticker_profiles:
            profile = self.ticker_profiles[ticker]
            
            if profile['category'] == 'high_volatility_growth':
                # Add volatility-focused features
                df_processed[f'{ticker}_volatility_ratio'] = (
                    df_processed['Realized_Vol_20'] / df_processed['Realized_Vol_20'].rolling(20).mean()
                )
                df_processed[f'{ticker}_volume_spike'] = (
                    df_processed['Volume'] / df_processed['Volume'].rolling(10).mean()
                )
                df_processed[f'{ticker}_momentum_acceleration'] = (
                    df_processed['Momentum_10'].diff()
                )
                
            elif profile['category'] == 'stable_large_cap':
                # Add stability-focused features
                df_processed[f'{ticker}_stability_score'] = (
                    1 / (df_processed['Realized_Vol_20'] + 0.01)
                )
                df_processed[f'{ticker}_trend_consistency'] = (
                    df_processed['Close'].rolling(20).apply(
                        lambda x: 1 if (x.iloc[-1] - x.iloc[0]) > 0 else -1
                    )
                )
        
        return df_processed
    
    def adaptive_feature_selection(self, ticker: str, df_processed: pd.DataFrame, 
                                 target_col: str = 'Close') -> List[str]:
        """Select optimal features for specific ticker"""
        logger.info(f"Performing adaptive feature selection for {ticker}")
        
        # Get ticker-specific focus features
        if ticker in self.ticker_profiles:
            focus_features = self.ticker_profiles[ticker]['fine_tuning_params']['focus_features']
        else:
            focus_features = ['Close', 'Volume', 'RSI', 'MACD']
        
        # Calculate feature importance for this ticker
        future_returns = df_processed[target_col].pct_change(5).shift(-5)
        
        feature_importance = {}
        for col in df_processed.columns:
            if col in ['Date', target_col]:
                continue
            try:
                corr = abs(df_processed[col].corr(future_returns))
                if not pd.isna(corr):
                    feature_importance[col] = corr
            except:
                continue
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Combine focus features with top correlated features
        selected_features = []
        
        # Add focus features first (prioritized)
        for feature in focus_features:
            if feature in df_processed.columns and feature not in selected_features:
                selected_features.append(feature)
        
        # Add top correlated features
        for feature, importance in sorted_features:
            if feature not in selected_features and len(selected_features) < 20:
                selected_features.append(feature)
        
        logger.info(f"Selected {len(selected_features)} features for {ticker}")
        return selected_features
    
    def generate_fine_tuning_recommendations(self, ticker: str, 
                                           current_performance: Dict[str, float],
                                           characteristics: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improving ticker performance"""
        recommendations = []
        
        # Performance-based recommendations
        if current_performance.get('r2', 0) < 0.5:
            recommendations.append(f"{ticker}: Low R² ({current_performance.get('r2', 0):.3f}) - Consider longer training or more data")
        
        if current_performance.get('directional_accuracy', 0) < 60:
            recommendations.append(f"{ticker}: Poor directional accuracy - Focus on momentum and trend features")
        
        # Characteristic-based recommendations
        if characteristics['type'] == 'high_volatility':
            recommendations.append(f"{ticker}: High volatility stock - Use longer sequences (90+ days) and focus on volatility features")
        
        if characteristics['liquidity'] == 'low':
            recommendations.append(f"{ticker}: Low liquidity - Reduce batch size and increase regularization")
        
        if characteristics['data_points'] < 200:
            recommendations.append(f"{ticker}: Limited data - Use transfer learning from similar tickers")
        
        # Specific improvement strategies
        if ticker == 'TSLA':
            recommendations.extend([
                "TSLA: Implement volatility-adjusted predictions",
                "TSLA: Add sentiment analysis features",
                "TSLA: Use ensemble methods for better stability"
            ])
        
        return recommendations

def run_fine_tuning_analysis():
    """Run comprehensive fine-tuning analysis"""
    logger.info("Starting fine-tuning analysis...")
    
    fine_tuner = TickerSpecificFineTuner()
    
    # Sample tickers for analysis
    sample_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    results = {
        'ticker_analysis': {},
        'fine_tuning_strategies': {},
        'recommendations': {}
    }
    
    # Generate sample data for each ticker
    for ticker in sample_tickers:
        logger.info(f"Analyzing {ticker}...")
        
        # Create realistic sample data
        np.random.seed(42 + hash(ticker) % 1000)
        dates = pd.date_range('2023-01-01', periods=250, freq='D')
        
        # Generate ticker-specific data patterns
        if ticker == 'TSLA':
            # High volatility
            volatility = 0.05
            trend = 0.002
        elif ticker in ['AAPL', 'MSFT']:
            # Stable large cap
            volatility = 0.02
            trend = 0.001
        else:
            # Default
            volatility = 0.03
            trend = 0.001
        
        price = 100
        data = []
        for i in range(250):
            change = np.random.normal(trend, volatility)
            price *= (1 + change)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'Date': dates[i],
                'Open': price,
                'High': high,
                'Low': low,
                'Close': price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Analyze characteristics
        characteristics = fine_tuner.analyze_ticker_characteristics(ticker, df)
        
        # Create enhanced features
        df_processed = fine_tuner.create_enhanced_features_for_ticker(ticker, df)
        
        # Get fine-tuning strategy
        strategy = fine_tuner.recommend_fine_tuning_strategy(ticker, characteristics)
        
        # Adaptive feature selection
        selected_features = fine_tuner.adaptive_feature_selection(ticker, df_processed)
        
        # Mock current performance (based on diagnostics)
        current_performance = {
            'r2': 0.75 if ticker == 'AAPL' else (0.45 if ticker == 'TSLA' else 0.65),
            'directional_accuracy': 72 if ticker == 'AAPL' else (62 if ticker == 'TSLA' else 68)
        }
        
        # Generate recommendations
        recommendations = fine_tuner.generate_fine_tuning_recommendations(
            ticker, current_performance, characteristics
        )
        
        results['ticker_analysis'][ticker] = characteristics
        results['fine_tuning_strategies'][ticker] = strategy
        results['recommendations'][ticker] = recommendations
    
    # Generate summary report
    report = generate_fine_tuning_report(results)
    print(report)
    
    return results, report

def generate_fine_tuning_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive fine-tuning report"""
    report = []
    report.append("=" * 70)
    report.append("FINE-TUNING STRATEGY REPORT - 30-DAY PREDICTIONS")
    report.append("=" * 70)
    
    # Summary by ticker type
    report.append("\n1. TICKER CHARACTERISTICS ANALYSIS")
    report.append("-" * 50)
    
    for ticker, analysis in results['ticker_analysis'].items():
        report.append(f"\n{ticker}:")
        report.append(f"  Type: {analysis['type']}")
        report.append(f"  Liquidity: {analysis['liquidity']}")
        report.append(f"  Volatility: {analysis['volatility']:.3f}")
        report.append(f"  Data Points: {analysis['data_points']}")
    
    # Fine-tuning strategies
    report.append("\n2. RECOMMENDED FINE-TUNING STRATEGIES")
    report.append("-" * 50)
    
    for ticker, strategy in results['fine_tuning_strategies'].items():
        report.append(f"\n{ticker}:")
        report.append(f"  Learning Rate: {strategy['learning_rate']}")
        report.append(f"  Sequence Length: {strategy['sequence_length']}")
        report.append(f"  Regularization: {strategy['regularization']}")
        report.append(f"  Training Epochs: {strategy['training_epochs']}")
        report.append(f"  Batch Size: {strategy['batch_size']}")
        report.append(f"  Focus Features: {', '.join(strategy['focus_features'])}")
    
    # Recommendations
    report.append("\n3. IMPROVEMENT RECOMMENDATIONS")
    report.append("-" * 50)
    
    for ticker, recommendations in results['recommendations'].items():
        if recommendations:
            report.append(f"\n{ticker}:")
            for i, rec in enumerate(recommendations, 1):
                report.append(f"  {i}. {rec}")
    
    # Priority actions
    report.append("\n4. PRIORITY ACTIONS")
    report.append("-" * 50)
    report.append("1. Implement ticker-specific fine-tuning for TSLA (highest improvement potential)")
    report.append("2. Create volatility-adjusted prediction models for high-volatility stocks")
    report.append("3. Implement ensemble methods combining multiple models")
    report.append("4. Add sentiment analysis features for growth stocks")
    report.append("5. Use transfer learning for tickers with limited data")
    
    return "\n".join(report)

if __name__ == "__main__":
    results, report = run_fine_tuning_analysis()
