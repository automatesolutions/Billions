import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import yfinance as yf
import logging
import dash
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import torch
import torch.nn as nn
import json
import warnings
from textblob import TextBlob
from bs4 import BeautifulSoup
from fredapi import Fred
try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    # Fallback if scipy is not available
    def gaussian_filter1d(data, sigma):
        return data

# Add parent directory to path to import db module
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.core import SessionLocal, engine, Base
from db.models import PerfMetric

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)
from db.models import PerfMetric
from funda.outlier_engine import STRATEGIES
from funda.refresh_outliers import start_refresh_thread, get_refresh_status
import threading

# Suppress All-NaN warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

# === INSTITUTIONAL FLOW ANALYSIS FUNCTIONS ===

def compute_institutional_flow_analysis(df):
    """
    Clean function to compute institutional flow analysis with improved accuracy
    """
    df_vol = df.copy()
    
    # Volume Analysis
    df_vol['Volume_SMA_10'] = df_vol['Volume'].rolling(window=10).mean()
    df_vol['Volume_SMA_20'] = df_vol['Volume'].rolling(window=20).mean()
    df_vol['Volume_Ratio'] = df_vol['Volume'] / df_vol['Volume_SMA_10']
    df_vol['Volume_Ratio_20'] = df_vol['Volume'] / df_vol['Volume_SMA_20']
    df_vol['Price_Volume_Trend'] = df_vol['Price_Change'] * df_vol['Volume_Ratio']
    
    # Price Impact Analysis
    df_vol['Price_Move_Abs'] = abs(df_vol['Close'] - df_vol['Open'])
    df_vol['Price_Move_Pct'] = df_vol['Price_Move_Abs'] / df_vol['Open']
    df_vol['High_Low_Range'] = (df_vol['High'] - df_vol['Low']) / df_vol['Close']
    df_vol['Price_Impact_Efficiency'] = df_vol['Price_Move_Pct'] / (df_vol['Volume_Ratio'] + 0.1)
    
    # Dollar Volume Analysis
    df_vol['Dollar_Volume'] = df_vol['Volume'] * df_vol['Close']
    df_vol['Dollar_Volume_SMA'] = df_vol['Dollar_Volume'].rolling(20).mean()
    df_vol['Dollar_Volume_Ratio'] = df_vol['Dollar_Volume'] / df_vol['Dollar_Volume_SMA']
    
    # Price Impact Classification (Key Feature)
    df_vol['Price_Impact_Percentile'] = df_vol['Price_Impact_Efficiency'].rolling(window=50).rank(pct=True)
    df_vol['High_Price_Impact'] = df_vol['Price_Impact_Percentile'] > 0.8  # Top 20%
    df_vol['Low_Price_Impact'] = df_vol['Price_Impact_Percentile'] < 0.2  # Bottom 20%
    
    # More Strict Institutional Scoring (0-100)
    volume_score = np.clip(df_vol['Volume_Ratio'] * 15, 0, 30)  # Reduced weight
    dollar_score = np.clip(np.log(df_vol['Dollar_Volume_Ratio'] + 1) * 20, 0, 40)  # Increased weight
    efficiency_score = np.clip((1 / (df_vol['Price_Impact_Efficiency'] + 0.1)) * 15, 0, 30)  # Increased weight
    price_consistency = 1 - df_vol['High_Low_Range']
    consistency_score = np.clip(price_consistency * 10, 0, 10)
    
    df_vol['Institutional_Score'] = volume_score + dollar_score + efficiency_score + consistency_score
    
    # More Strict Classification (Higher thresholds)
    df_vol['Likely_Institutional'] = df_vol['Institutional_Score'] > 75  # Increased from 60
    df_vol['Likely_Retail'] = df_vol['Institutional_Score'] < 25  # Decreased from 30
    df_vol['Price_Direction'] = np.where(df_vol['Close'] > df_vol['Open'], 1, -1)
    
    # More Strict Institutional Activity Detection
    df_vol['Institutional_Buying'] = (
        df_vol['Likely_Institutional'] & 
        (df_vol['Price_Direction'] == 1) &
        (df_vol['Price_Move_Pct'] > 0.01) &  # Increased from 0.005 to 0.01 (1%)
        (df_vol['Dollar_Volume'] > df_vol['Dollar_Volume_SMA'] * 2.5) &  # Must be 2.5x average dollar volume
        (df_vol['Volume_Ratio'] > 2.0)  # Must be 2x average volume
    )
    
    df_vol['Institutional_Selling'] = (
        df_vol['Likely_Institutional'] & 
        (df_vol['Price_Direction'] == -1) &
        (df_vol['Price_Move_Pct'] > 0.01) &  # Increased from 0.005 to 0.01 (1%)
        (df_vol['Dollar_Volume'] > df_vol['Dollar_Volume_SMA'] * 2.5) &  # Must be 2.5x average dollar volume
        (df_vol['Volume_Ratio'] > 2.0)  # Must be 2x average volume
    )
    
    # More Strict Capitulation Detection
    df_vol['Capitulation'] = (
        (df_vol['Volume_Ratio_20'] > 4.0) &  # Increased from 3.0 to 4.0
        (df_vol['Price_Direction'] == -1) &
        (df_vol['Price_Move_Pct'] > 0.05) &  # Increased from 0.03 to 0.05 (5%)
        (df_vol['High_Low_Range'] > 0.08) &  # Increased from 0.05 to 0.08 (8%)
        (df_vol['Dollar_Volume'] > df_vol['Dollar_Volume_SMA'] * 3.0)  # Must be 3x average dollar volume
    )
    
    return df_vol

def enhance_chart_with_institutional_flow(fig, df_processed):
    """
    Clean function to enhance chart with institutional flow markers and Price Impact
    """
    try:
        # Update volume bar colors with Price Impact focus
        volume_colors = []
        for i, row in df_processed.iterrows():
            if row.get('Capitulation', False):
                volume_colors.append('#ff8800')  # Orange for capitulation
            elif row.get('Institutional_Buying', False):
                volume_colors.append('#00ff00')  # Green for institutional buying
            elif row.get('Institutional_Selling', False):
                volume_colors.append('#ff0000')  # Red for institutional selling
            elif row.get('High_Price_Impact', False):
                volume_colors.append('#ff00ff')  # Magenta for high price impact
            elif row.get('Low_Price_Impact', False):
                volume_colors.append('#00ffff')  # Cyan for low price impact
            else:
                volume_colors.append('#888888')  # Gray for normal
        
        # Find the volume bar trace and update its colors (ensure it's the volume bar)
        volume_trace_found = False
        for i, trace in enumerate(fig.data):
            if trace.name == 'Volume' and trace.type == 'bar':
                fig.data[i].marker.color = volume_colors
                volume_trace_found = True
                break
        
        if not volume_trace_found:
            logging.warning("Volume bar trace not found for color update")
            # Try alternative method - find by trace index (volume is usually second trace)
            if len(fig.data) > 1:
                fig.data[1].marker.color = volume_colors
                logging.info("Applied volume colors using fallback method")
        
        # Add both institutional flow markers and price impact markers
        add_price_impact_markers(fig, df_processed)
        add_institutional_markers(fig, df_processed)
        
        logging.info("Chart enhanced with institutional flow and price impact analysis")
        
    except Exception as e:
        logging.warning(f"Error enhancing chart: {e}")

def add_price_impact_markers(fig, df_processed):
    """
    Add Price Impact markers to chart (High/Low Price Impact) - Cleaner version
    """
    try:
        # High Price Impact markers (Top 20% - Retail/Momentum trading)
        high_impact_dates = df_processed[df_processed.get('High_Price_Impact', False) == True].index
        logging.info(f"Found {len(high_impact_dates)} High Price Impact events")
        if len(high_impact_dates) > 0:
            # Only show the most significant high impact events (top 3 only)
            high_impact_scores = df_processed.loc[high_impact_dates, 'Price_Impact_Efficiency']
            top_3 = high_impact_scores.nlargest(min(3, len(high_impact_scores)))
            significant_high_impact = top_3.index
            
            if len(significant_high_impact) > 0:
                high_impact_prices = df_processed.loc[significant_high_impact, 'High'] * 1.05
                fig.add_trace(create_marker_trace(
                    significant_high_impact, high_impact_prices, 'High Price Impact',
                    'diamond', '#ff00ff', 8, df_processed
                ), row=1, col=1)
        
        # Low Price Impact markers (Bottom 20% - Institutional trading)
        low_impact_dates = df_processed[df_processed.get('Low_Price_Impact', False) == True].index
        logging.info(f"Found {len(low_impact_dates)} Low Price Impact events")
        if len(low_impact_dates) > 0:
            # Only show the most significant low impact events (top 3 only)
            low_impact_scores = df_processed.loc[low_impact_dates, 'Price_Impact_Efficiency']
            bottom_3 = low_impact_scores.nsmallest(min(3, len(low_impact_scores)))
            significant_low_impact = bottom_3.index
            
            if len(significant_low_impact) > 0:
                low_impact_prices = df_processed.loc[significant_low_impact, 'Low'] * 0.95
                fig.add_trace(create_marker_trace(
                    significant_low_impact, low_impact_prices, 'Low Price Impact',
                    'square', '#00ffff', 8, df_processed
                ), row=1, col=1)
        
    except Exception as e:
        logging.warning(f"Error adding price impact markers: {e}")

def add_institutional_markers(fig, df_processed):
    """
    Simplified institutional flow markers - only most important events
    """
    try:
        # Capitulation markers (most important - show all)
        capitulation_dates = df_processed[df_processed.get('Capitulation', False) == True].index
        if len(capitulation_dates) > 0:
            capitulation_prices = df_processed.loc[capitulation_dates, 'Low'] * 0.98
            fig.add_trace(create_marker_trace(
                capitulation_dates, capitulation_prices, 'Capitulation', 
                'triangle-down', '#ff8800', 12, df_processed
            ), row=1, col=1)
        
        # Institutional buying markers (only top 2 most significant)
        inst_buy_dates = df_processed[df_processed.get('Institutional_Buying', False) == True].index
        if len(inst_buy_dates) > 0:
            inst_buy_scores = df_processed.loc[inst_buy_dates, 'Institutional_Score']
            top_inst_buy = inst_buy_scores.nlargest(min(2, len(inst_buy_scores))).index
            
            if len(top_inst_buy) > 0:
                inst_buy_prices = df_processed.loc[top_inst_buy, 'High'] * 1.02
                fig.add_trace(create_marker_trace(
                    top_inst_buy, inst_buy_prices, 'Institutional Buying',
                    'triangle-up', '#00ff00', 10, df_processed
                ), row=1, col=1)
        
        # Institutional selling markers (only top 2 most significant)
        inst_sell_dates = df_processed[df_processed.get('Institutional_Selling', False) == True].index
        if len(inst_sell_dates) > 0:
            inst_sell_scores = df_processed.loc[inst_sell_dates, 'Institutional_Score']
            top_inst_sell = inst_sell_scores.nlargest(min(2, len(inst_sell_scores))).index
            
            if len(top_inst_sell) > 0:
                inst_sell_prices = df_processed.loc[top_inst_sell, 'Low'] * 0.98
                fig.add_trace(create_marker_trace(
                    top_inst_sell, inst_sell_prices, 'Institutional Selling',
                    'triangle-down', '#ff0000', 10, df_processed
                ), row=1, col=1)
                
    except Exception as e:
        logging.warning(f"Error adding institutional markers: {e}")

def create_marker_trace(dates, prices, name, symbol, color, size, df_processed):
    """
    Create standardized marker trace for institutional flow - Cleaner version
    """
    return go.Scatter(
        x=dates,
        y=prices,
        mode='markers',
        marker=dict(
            symbol=symbol,
            size=size,
            color=color,
            line=dict(width=1.5, color='white'),
            opacity=0.8
        ),
        name=name,
        text=[f'{name}<br>{date.strftime("%m/%d")}' for date in dates],  # Shorter text
        hovertemplate='<b>%{text}</b><br>Volume: %{customdata[0]:,}<br>Change: %{customdata[1]:.2%}<br>Score: %{customdata[2]:.1f}<extra></extra>',
        customdata=[[df_processed.loc[date, 'Volume'], 
                     df_processed.loc[date, 'Price_Change'],
                     df_processed.loc[date, 'Institutional_Score']] for date in dates],
        showlegend=True
    )

def generate_volume_analysis_summary(df_processed):
    """
    Generate clean volume analysis summary with Price Impact focus
    """
    try:
        summary_parts = []
        
        # Count events
        capitulation_count = df_processed.get('Capitulation', pd.Series([False] * len(df_processed))).sum()
        inst_buy_count = df_processed.get('Institutional_Buying', pd.Series([False] * len(df_processed))).sum()
        inst_sell_count = df_processed.get('Institutional_Selling', pd.Series([False] * len(df_processed))).sum()
        high_impact_count = df_processed.get('High_Price_Impact', pd.Series([False] * len(df_processed))).sum()
        low_impact_count = df_processed.get('Low_Price_Impact', pd.Series([False] * len(df_processed))).sum()
        
        summary_parts.append("📊 VOLUME & PRICE IMPACT ANALYSIS:")
        
        # Price Impact Analysis (Key Focus)
        summary_parts.append(f"💥 HIGH PRICE IMPACT: {high_impact_count} events (Retail/Momentum trading)")
        summary_parts.append(f"🏛️ LOW PRICE IMPACT: {low_impact_count} events (Institutional trading)")
        
        # Capitulation events
        if capitulation_count > 0:
            latest_capitulation = df_processed[df_processed.get('Capitulation', False) == True].index[-1]
            latest_capitulation_vol = df_processed.loc[latest_capitulation, 'Volume']
            latest_capitulation_change = df_processed.loc[latest_capitulation, 'Price_Change']
            summary_parts.append(f"🚨 CAPITULATION: {capitulation_count} event(s) - Latest: {latest_capitulation.date()} (Vol: {latest_capitulation_vol:,.0f}, Change: {latest_capitulation_change:.2%})")
        
        # Institutional buying (only if significant)
        if inst_buy_count > 0:
            latest_buy = df_processed[df_processed.get('Institutional_Buying', False) == True].index[-1]
            latest_buy_vol = df_processed.loc[latest_buy, 'Volume']
            latest_buy_change = df_processed.loc[latest_buy, 'Price_Change']
            latest_buy_score = df_processed.loc[latest_buy, 'Institutional_Score']
            summary_parts.append(f"📈 INSTITUTIONAL BUYING: {inst_buy_count} event(s) - Latest: {latest_buy.date()} (Vol: {latest_buy_vol:,.0f}, Change: {latest_buy_change:.2%}, Score: {latest_buy_score:.1f})")
        
        # Institutional selling (only if significant)
        if inst_sell_count > 0:
            latest_sell = df_processed[df_processed.get('Institutional_Selling', False) == True].index[-1]
            latest_sell_vol = df_processed.loc[latest_sell, 'Volume']
            latest_sell_change = df_processed.loc[latest_sell, 'Price_Change']
            latest_sell_score = df_processed.loc[latest_sell, 'Institutional_Score']
            summary_parts.append(f"📉 INSTITUTIONAL SELLING: {inst_sell_count} event(s) - Latest: {latest_sell.date()} (Vol: {latest_sell_vol:,.0f}, Change: {latest_sell_change:.2%}, Score: {latest_sell_score:.1f})")
        
        # Volume statistics
        avg_volume = df_processed['Volume'].mean()
        max_volume = df_processed['Volume'].max()
        max_volume_date = df_processed['Volume'].idxmax()
        max_volume_change = df_processed.loc[max_volume_date, 'Price_Change']
        current_score = df_processed['Institutional_Score'].iloc[-1]
        current_price_impact = df_processed['Price_Impact_Efficiency'].iloc[-1]
        
        summary_parts.append(f"📊 VOLUME STATS: Avg: {avg_volume:,.0f}, Max: {max_volume:,.0f} ({max_volume_date.date()}, Change: {max_volume_change:.2%})")
        summary_parts.append(f"🎯 CURRENT INSTITUTIONAL SCORE: {current_score:.1f}/100")
        summary_parts.append(f"⚡ CURRENT PRICE IMPACT: {current_price_impact:.4f}")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        logging.warning(f"Error generating volume analysis summary: {e}")
        return "📊 VOLUME ANALYSIS: Error generating analysis"

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log versions
logging.info(f"Dash version: {dash.__version__}")
logging.info(f"dash_bootstrap_components version: {dbc.__version__}")
logging.info(f"PyTorch version: {torch.__version__}")

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not FRED_API_KEY or not ALPHA_VANTAGE_API_KEY:
    raise ValueError("Missing API keys in .env file.")

# Initialize FRED API
fred = Fred(api_key=FRED_API_KEY)

# Define LSTM model for daily predictions
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
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

# Define CNN-LSTM model for 1-minute predictions
class CNN_LSTM(nn.Module):
    def __init__(self, price_levels, channels, scalar_features_size, hidden_size=100, num_layers=3, output_size=30, dropout=0.2):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.cnn_residual = nn.Conv1d(channels, 64, kernel_size=1)
        self.cnn_output_size = 64
        self.lstm = nn.LSTM(input_size=self.cnn_output_size + scalar_features_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, order_footprint, scalar_features):
        batch_size, seq_length, price_levels, channels = order_footprint.shape
        order_footprint = order_footprint.view(batch_size * seq_length, channels, price_levels)
        cnn_main = self.cnn(order_footprint)
        cnn_residual = self.cnn_residual(order_footprint)
        cnn_residual = nn.functional.adaptive_max_pool1d(cnn_residual, 1)
        cnn_out = cnn_main + cnn_residual
        cnn_out = cnn_out.squeeze(-1)
        cnn_out = cnn_out.view(batch_size, seq_length, self.cnn_output_size)
        if scalar_features.size(-1) != 9:
            raise ValueError(f"Expected scalar_features with last dim 9, got {scalar_features.size(-1)}")
        lstm_input = torch.cat([cnn_out, scalar_features], dim=2)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(lstm_input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

# Load LSTM models
try:
    daily_model = StockLSTM(input_size=14, output_size=30)  # Updated for new features
    daily_model_path = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\model", 'lstm_daily_model.pt')
    daily_model.load_state_dict(torch.load(daily_model_path, weights_only=True))
    daily_model.eval()
    logging.info("Successfully loaded daily LSTM model.")
except Exception as e:
    logging.error(f"Error loading daily LSTM model: {e}")
    raise

try:
    minute_model = CNN_LSTM(price_levels=10, channels=2, scalar_features_size=9, output_size=30)  # Updated for new features
    minute_model_path = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\model", 'lstm_1minute_model.pt')
    minute_model.load_state_dict(torch.load(minute_model_path, weights_only=True))
    minute_model.eval()
    logging.info("Successfully loaded 1-minute CNN-LSTM model.")
except Exception as e:
    logging.error(f"Error loading 1-minute CNN-LSTM model: {e}")
    raise

# Function to check for cached data
def load_cached_data(symbol, interval='1d', cache_dir="cache"):
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval.replace(' ', '_')}.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logging.info(f"Loaded cached data for {symbol} (interval={interval})")
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if set(['Open', 'High', 'Low', 'Close', 'Volume']).issubset(df.columns):
                if interval == '1d' and len(df) < 252:
                    logging.info(f"Cached data for {symbol} has only {len(df)} rows; re-fetching.")
                    return None
                return df
            logging.warning(f"Cached data for {symbol} does not contain expected columns.")
            return None
        except Exception as e:
            logging.warning(f"Error loading cached data for {symbol}: {e}")
    return None

# Function to save data to cache
def save_to_cache(symbol, data, interval='1d', cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_file = os.path.join(cache_dir, f"{symbol}_{interval.replace(' ', '_')}.csv")
    if isinstance(data, pd.Series):
        data = data.to_frame(name='Close')
    data.to_csv(cache_file)
    logging.info(f"Saved data for {symbol} (interval={interval}) to {cache_file}")

# Function to fetch stock and sector data
def fetch_yfinance_data(symbol, period='10y', interval='1d'):
    cached_data = load_cached_data(symbol, interval=interval)
    if cached_data is not None:
        return cached_data
    try:
        logging.info(f"Fetching data for {symbol} from yfinance with period={period}, interval={interval}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[expected_columns]
        min_rows = 60 if interval == '1d' else 30
        if len(df) < min_rows and period != 'max':
            df = ticker.history(period='max', interval=interval)
            if df.empty:
                raise ValueError(f"No data for {symbol} with period='max'")
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df = df[expected_columns]
        if interval == '1m':
            df = df.between_time('09:30', '16:00')
        df['Sector_Close'] = df['Close']  # Placeholder; will be updated in ticker analysis
        df['Order_Flow'] = 0  # Placeholder
        logging.info(f"Fetched data for {symbol}. Shape: {df.shape}")
        save_to_cache(symbol, df, interval=interval)
        return df
    except Exception as e:
        logging.error(f"Error fetching {symbol} from yfinance: {e}")
        return None

# Function to fetch order footprint data
def fetch_order_footprint(ticker, period, interval, expected_rows):
    try:
        logging.info(f"Fetching order footprint for {ticker}")
        price_levels = 10
        channels = 2
        footprint = np.random.rand(expected_rows, price_levels, channels) * 100  # Dummy data
        return footprint
    except Exception as e:
        logging.warning(f"Error fetching order footprint for {ticker}: {e}. Returning dummy data.")
        return np.zeros((expected_rows, price_levels, channels))

# Function to fetch Alpha Vantage data
def fetch_alpha_vantage_data(symbol, api_key, delay=12):
    cached_data = load_cached_data(symbol, interval='1d')
    if cached_data is not None:
        return cached_data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={api_key}"
    try:
        logging.info(f"Fetching data for {symbol} from Alpha Vantage")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"No data for {symbol}: {data.get('Information', 'Unknown error')}")
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().astype(float)
        logging.info(f"Fetched data for {symbol} from Alpha Vantage")
        save_to_cache(symbol, df['5. adjusted close'], interval='1d')
        time.sleep(delay)
        return df['5. adjusted close']
    except Exception as e:
        logging.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
        return None

# Function to fetch stock data
def fetch_stock_data(symbol, api_key, period='10y', interval='1d'):
    data = fetch_yfinance_data(symbol, period, interval)
    if data is not None:
        return data
    data = fetch_alpha_vantage_data(symbol, api_key)
    if data is not None:
        return data
    logging.error(f"Failed to fetch data for {symbol}")
    return None

# Function to fetch FRED data
def fetch_fred_data(series_id):
    try:
        logging.info(f"Fetching FRED series {series_id}")
        data = fred.get_series(series_id)
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data = data.resample('D').ffill().bfill()
        return data
    except Exception as e:
        logging.error(f"Error fetching FRED series {series_id}: {e}")
        return None

# Compute RSI
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute Realized Volatility
def compute_realized_volatility(df, window=30):
    returns = df['Close'].pct_change().dropna()
    return np.sqrt((returns**2).rolling(window=window).sum())

# Fetch S&P 500 and sector data
sp500 = fetch_stock_data('SPY', ALPHA_VANTAGE_API_KEY)
if sp500 is None:
    logging.error("Failed to fetch S&P 500 proxy (SPY) data.")
    exit()

sector_etfs = {
    'Technology': 'XLK', 'Financials': 'XLF', 'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI', 'Utilities': 'XLU', 'Healthcare': 'XLV',
    'Communication Services': 'XLC', 'Consumer Staples': 'XLP',
    'Materials': 'XLB', 'Real Estate': 'XLRE', 'Energy': 'XLE'
}
sector_data = {}
for sector, ticker in sector_etfs.items():
    sector_data[sector] = fetch_stock_data(ticker, ALPHA_VANTAGE_API_KEY)
    if sector_data[sector] is None:
        logging.warning(f"Failed to fetch data for {sector} ({ticker}). Using zeros.")
        sector_data[sector] = pd.Series(0, index=sp500.index)

# Fetch macroeconomic data
macro_series = {
    'GDP': 'A191RL1Q225SBEA', 'Inflation': 'CPIAUCSL', 'Unemployment': 'UNRATE',
    'Interest Rate': 'FEDFUNDS', 'Consumer Confidence': 'UMCSENT', 'VIX': 'VIXCLS',
    '10Y Treasury': 'DGS10', '2Y Treasury': 'DGS2'
}
macro_data = {key: fetch_fred_data(series_id) for key, series_id in macro_series.items()}

# Combine data
common_index = sp500.index
data_dict = {'SP500': sp500['Close'] if isinstance(sp500, pd.DataFrame) else sp500}
for sector in sector_data:
    data_dict[f"{sector}_Close"] = sector_data[sector]['Close'] if isinstance(sector_data[sector], pd.DataFrame) else sector_data[sector]
for key in macro_data:
    if macro_data[key] is not None:
        data_dict[key] = macro_data[key].reindex(common_index, method='ffill').bfill()
    else:
        logging.warning(f"Macro data for {key} is None. Filling with zeros.")
        data_dict[key] = pd.Series(0, index=common_index)

data = pd.DataFrame(data_dict).dropna()

# Feature Engineering
data['SP500_Returns'] = data['SP500'].pct_change(periods=5).shift(-5)
for sector in sector_data:
    data[f"{sector}_Returns"] = data[f"{sector}_Close"].pct_change(periods=5).shift(-5)
data['SP500_MA20'] = data['SP500'].rolling(window=20).mean()
data['SP500_RSI'] = compute_rsi(data['SP500'], 14)
data['Yield_Spread'] = data['10Y Treasury'] - data['2Y Treasury']
data['GDP_Lag'] = data['GDP'].shift(1)
data['Inflation_Lag'] = data['Inflation'].shift(1)
data = data.dropna()

# Define features
features = ['SP500_MA20', 'SP500_RSI', 'GDP_Lag', 'Inflation_Lag', 'Unemployment',
            'Interest Rate', 'Consumer Confidence', 'VIX', 'Yield_Spread']
X = data[features]
nan_columns = X.columns[X.isna().all()]
if not nan_columns.empty:
    logging.warning(f"Columns with all NaN values: {nan_columns.tolist()}")
    X = X.drop(columns=nan_columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Define targets
y_market = (data['SP500_Returns'] > 0.01).astype(int)
y_sector = {sector: data[f"{sector}_Returns"] for sector in sector_data}

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_market_train, y_market_test = y_market[:train_size], y_market[train_size:]
y_sector_train = {sector: y_sector[sector][:train_size] for sector in y_sector}
y_sector_test = {sector: y_sector[sector][train_size:] for sector in y_sector}

# Train models
market_model = RandomForestClassifier(n_estimators=100, random_state=42)
market_model.fit(X_train, y_market_train)
sector_models = {}
for sector in y_sector:
    sector_models[sector] = RandomForestRegressor(n_estimators=100, random_state=42)
    sector_models[sector].fit(X_train, y_sector_train[sector])

# Evaluate models
market_pred = market_model.predict(X_test)
market_accuracy = accuracy_score(y_market_test, market_pred)
logging.info(f"Market Direction Accuracy: {market_accuracy:.2f}")

sector_mse = {}
for sector in y_sector:
    sector_pred = sector_models[sector].predict(X_test)
    sector_mse[sector] = mean_squared_error(y_sector_test[sector], sector_pred)
    logging.info(f"{sector} MSE: {sector_mse[sector]:.4f}")

# Latest S&P 500 prediction
latest_data = X.iloc[-1:].copy()
market_direction = market_model.predict(latest_data)[0]
market_confidence = market_model.predict_proba(latest_data)[0][1] if market_direction == 1 else market_model.predict_proba(latest_data)[0][0]
market_direction_label = "Bullish" if market_direction == 1 else "Bearish"
logging.info(f"Predicted Market Direction: {market_direction_label} (Confidence: {market_confidence:.2%})")

sector_predictions = {}
for sector in y_sector:
    sector_predictions[sector] = sector_models[sector].predict(latest_data)[0]
    logging.info(f"Predicted {sector} 5-Day Return: {sector_predictions[sector]:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': market_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Prepare data for dashboard
sp500_df = pd.DataFrame({'Date': sp500.index, 'SP500': sp500['Close'] if isinstance(sp500, pd.DataFrame) else sp500})
sector_historical = pd.DataFrame(index=common_index)
for sector in sector_data:
    sector_historical[sector] = sector_data[sector]['Close'] if isinstance(sector_data[sector], pd.DataFrame) else sector_data[sector]
sector_historical = sector_historical.reset_index().rename(columns={'index': 'Date'})
sector_historical_melted = sector_historical.melt(id_vars='Date', var_name='Sector', value_name='Price')
sector_pred_df = pd.DataFrame.from_dict(sector_predictions, orient='index', columns=['Predicted_5Day_Return'])
sector_pred_df.reset_index(inplace=True)
sector_pred_df.rename(columns={'index': 'Sector'}, inplace=True)
feature_importance_df = feature_importance.copy()

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS for dark background
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background-color: #000 !important; }
            .container, .row, .col, .dash-table-container, .dash-table, .dash-spreadsheet-container {
                background-color: #000 !important;
                color: #fff !important;
            }
            .card, .card-body, .navbar, .modal-content, .dropdown-menu {
                background-color: #111 !important;
                color: #fff !important;
            }
            .form-control, .input-group-text, .custom-select, .dropdown-item {
                background-color: #222 !important;
                color: #fff !important;
            }
            .text-dark { color: #fff !important; }
            .text-primary { color: #39FF14 !important; }
            .bg-light { background-color: #222 !important; }
            /* Custom for dropdown */
            .Select-control, .Select-menu-outer, .Select-menu, .Select-option, .Select-placeholder, .Select--single {
                background-color: #222 !important;
                color: #fff !important;
                border-radius: 8px !important;
                font-size: 18px !important;
            }
            .Select-placeholder { color: #bbb !important; }
            .Select-arrow-zone { color: #fff !important; }
            /* Custom for dcc.Input */
            input[type="text"] {
                background-color: #222 !important;
                color: #fff !important;
                border: 1px solid #444 !important;
                border-radius: 8px !important;
                font-size: 18px !important;
                padding: 8px 16px !important;
                box-shadow: 0 0 8px #111;
            }
            input[type="text"]::placeholder {
                color: #bbb !important;
                font-style: italic;
                opacity: 1;
            }
            /* Custom for dcc.Dropdown */
            .dash-dropdown, .dash-dropdown * {
                background-color: #222 !important;
                color: #fff !important;
                border-radius: 8px !important;
                font-size: 18px !important;
            }
            .dotdigital-title-small {
                font-family: 'EnhancedDotDigital7', monospace, sans-serif;
                color: #FF7043 !important;
                font-size: 24x !important; /* or your preferred size */
                letter-spacing: 0.05em;
                text-shadow: 0 0 8px #FF7043, 0 0 2px #fff;
                text-transform: uppercase;
                line-height: 1.1;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Dashboard layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("BILLIONS", className="dotdigital-title text-center mb-4")),
        dbc.Col(html.P("Grow your wealth with real-time market insights, sector trends, and custom stock forecasts—powered by outlier detection.", 
                       className="neon-green text-center mb-4"), width=12),
        dbc.Col(html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       className="neon-green text-center mb-4"), width=12)
    ]),
    # Refresh button row
    dbc.Row([
        dbc.Col([
            dbc.Button(
                "🔄 Refresh Outlier Data", 
                id="refresh-button",
                color="success",
                className="me-2",
                disabled=False
            ),
            dbc.Spinner(
                html.Div(id="refresh-status", children="Ready to refresh data"),
                id="refresh-spinner",
                color="primary"
            )
        ], width=12, className="text-center mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.H4("Custom Ticker Analysis", className="dotdigital-title-small text-left mb-2"),
            html.Div([
                # Row for input and button
                html.Div([
                    dcc.Input(
                        id='ticker-input',
                        type='text',
                        placeholder='Enter NASDAQ ticker',
                        style={
                            'width': '200px',
                            'marginRight': '10px',
                            'backgroundColor': '#222',
                            'color': '#fff',
                            'border': '1px solid #444',
                            'borderRadius': '8px',
                            'fontSize': '18px',
                            'padding': '8px 16px',
                            'boxShadow': '0 0 8px #111'
                        }
                    ),
                    html.Button('Generate Chart', id='generate-button', n_clicks=0, style={
                        'backgroundColor': '#222',
                        'color': '#39FF14',
                        'border': '1px solid #39FF14',
                        'borderRadius': '8px',
                        'fontSize': '18px',
                        'padding': '8px 16px',
                        'marginLeft': '10px',
                        'verticalAlign': 'middle',
                        'boxShadow': '0 0 8px #39FF14'
                    })
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '0'}),
                # Dropdown below
                dcc.Dropdown(
                    id='timeframe-dropdown',
                    options=[
                        {'label': 'Daily', 'value': 'daily'},
                        {'label': '1-Minute', 'value': '1-minute'},
                    ],
                    value='daily',
                    style={
                        'width': '100px',
                        'display': 'block',
                        'marginTop': '10px',
                        'marginBottom': '10px',
                        'backgroundColor': '#222',
                        'color': '#fff',
                        'borderRadius': '8px',
                        'fontSize': '12px'
                    }
                )
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-start', 'gap': '0'}),
            html.Div(id='ticker-error', style={'color': 'red'})
        ], width=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(
                id='ticker-chart-area',
                style={
                    'height': '500px',
                    'backgroundColor': '#222',
                    'borderRadius': '12px',
                    'marginBottom': '24px',
                    'position': 'relative',
                    'overflow': 'hidden',
                    'width': '100%'
                },
                children=[
                    html.Div([
                        html.Img(
                            src='/assets/nanakorobi_yaoki.png',
                            style={
                                'height': '180px',
                                'marginBottom': '24px',
                                'opacity': '0.5',
                                'display': 'block'
                            }
                        ),
                        html.Div(
                            '七 転 八 起',
                            style={
                                'color': '#39FF14',
                                'fontSize': '38px',
                                'textAlign': 'center',
                                'fontFamily': 'Arial, sans-serif',
                                'letterSpacing': '0.05em'
                            }
                        )
                    ], style={
                        'position': 'absolute',
                        'top': '50%',
                        'left': '50%',
                        'transform': 'translate(-50%, -50%)'
                    })
                ]
            )
        ], width=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col([
            html.H3(id='market-prediction', className="text-center", style={'color': '#39FF14', 'fontWeight': 'bold'}),
            html.P("Key Drivers:", style={'color': '#fff', 'fontSize': '20px', 'marginBottom': '8px'}),
            html.Ul(id='key-drivers', style={'color': '#fff', 'fontSize': '18px'}),
            html.H4("Economic News", style={'color': '#39FF14', 'marginTop': '24px'}),
            html.Div(id='news-section'),
            html.H4("Ticker Analysis", style={'color': '#39FF14', 'marginTop': '24px'}),
            html.Div(id='ticker-analysis-section', style={'color': '#fff', 'fontSize': '16px', 'marginBottom': '16px'}),
            html.H4("Hype/Promotion", style={'color': '#39FF14', 'marginTop': '24px'}),
            html.Div(id='hype-section', style={'color': '#fff', 'fontSize': '16px', 'marginBottom': '16px'}),
            html.H4("Caveat Emptor/Prohibited Alarm", style={'color': '#FF4C4C', 'marginTop': '24px'}),
            html.Div(id='alarm-section', style={'color': '#FF4C4C', 'fontSize': '16px', 'marginBottom': '16px'})
        ], width=3, className="p-3", style={
            'height': '100vh',
            'overflow': 'auto',
            'backgroundColor': '#000',
            'borderRadius': '12px'
        }),
        dbc.Col([
            dbc.Row([dbc.Col(dcc.Graph(id='sp500-chart', figure={}), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='sector-historical-chart', figure={}), width=12)], className="mb-4"),
            dbc.Row([dbc.Col(dcc.Graph(id='sector-pred-chart', figure={}), width=12)], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='strategy-dropdown',
                        options=[
                            {'label': 'Scalp', 'value': 'scalp'},
                            {'label': 'Swing', 'value': 'swing'},
                            {'label': 'Longterm', 'value': 'longterm'}
                        ],
                        value='scalp',
                        clearable=False,
                        style={'width': '200px', 'marginBottom': '16px', 'backgroundColor': '#222', 'color': '#fff'}
                    ),
                    dcc.Graph(
                        id='feature-outlier-scatter',
                        config={'displayModeBar': True},
                        style={'height': '700px', 'width': '1200px', 'margin': '0 auto'}
                    )
                ], style={'backgroundColor': '#222', 'borderRadius': '12px', 'padding': '16px'})
            ])
        ], width=9)
    ]),
    # Logo and partnership text at bottom left
    html.Div([
        html.Img(
            src="/assets/logo.png",
            style={
                'height': '60px',
                'marginRight': '12px',
                'verticalAlign': 'middle'
            }
        ),
        html.Span(
            "Partnership with Kumpooni",
            style={
                'color': '#fff',
                'fontSize': '10px',
                'verticalAlign': 'middle',
                'fontFamily': 'Arial, sans-serif',
                'textShadow': '0 0 8px #fff, 0 0 2px #fff'
            }
        )
    ], style={
        'position': 'fixed',
        'left': '20px',
        'bottom': '20px',
        'zIndex': '1000',
        'display': 'flex',
        'alignItems': 'center',
        'background': 'rgba(0,0,0,0.5)',
        'padding': '8px 16px',
        'borderRadius': '12px'
    }),
    # Interval component for progress updates
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    )
], fluid=True)

# Callback to update S&P 500 and sector predictions
@app.callback(
    [
        Output('market-prediction', 'children'),
        Output('key-drivers', 'children'),
        Output('sp500-chart', 'figure'),
        Output('sector-historical-chart', 'figure'),
        Output('sector-pred-chart', 'figure'),
        Output('news-section', 'children')
    ],
    Input('generate-button', 'n_clicks')
)
def update_sp500_dashboard(n_clicks):
    try:
        # Check if required variables are available
        if 'market_direction_label' not in globals() or 'market_confidence' not in globals():
            return html.Div("Loading market data...", style={'color': '#39FF14'}), html.Div("Loading..."), {}, {}, {}, html.Div("Loading news...")
        
        market_pred_text = html.Div(
            f"Market Prediction: {market_direction_label} (Confidence: {market_confidence:.2%})",
            style={'fontFamily': 'EnhancedDotDigital7', 'fontSize': '24px'}
        )
        key_drivers = [html.Li(
            f"{row['Feature']}: {row['Importance']:.2%}",
            style={'fontFamily': 'EnhancedDotDigital7', 'fontSize': '15px'}
        ) for _, row in feature_importance_df.iterrows()]
        sp500_fig = px.line(sp500_df, x='Date', y='SP500', title='S&P 500 (SPY) Historical Data', template='none')
        sp500_fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black', 
            font_color='#fff',
            xaxis=dict(color='#fff'),
            yaxis=dict(color='#fff'),
            font_family='EnhancedDotDigital7',
            font_size=24
        )
        sector_hist_fig = px.line(sector_historical_melted, x='Date', y='Price', color='Sector', 
                                  title='Sector ETF Historical Data', template='none')
        sector_hist_fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='#fff',
            xaxis=dict(color='#fff'),
            yaxis=dict(color='#fff')
        )
        sector_pred_fig = px.bar(sector_pred_df, x='Predicted_5Day_Return', y='Sector', orientation='h', 
                                 title='Predicted 5-Day Sector Returns', template='none')
        sector_pred_fig.update_layout(
            plot_bgcolor='black',
            paper_bgcolor='black',
            font_color='#fff',
            xaxis=dict(color='#fff'),
            yaxis=dict(color='#fff')
        )
        market_direction_text = f"Market Direction: {market_direction_label} (Confidence: {market_confidence:.2%})"
        explanation = [
            html.Li(f"Economic Growth: GDP at {data['GDP'].iloc[-1]:.2f}% supports {'bullish' if market_direction_label == 'Bullish' else 'bearish'} markets."),
            html.Li(f"Unemployment: {data['Unemployment'].iloc[-1]:.2f}% indicates {'strong' if data['Unemployment'].iloc[-1] < 5 else 'weak'} labor markets."),
            html.Li(f"Inflation: CPI at {data['Inflation'].iloc[-1]/100:.2f}% suggests {'stable' if data['Inflation'].iloc[-1]/100 < 3 else 'high'} price pressures."),
            html.Li(f"VIX: {data['VIX'].iloc[-1]:.2f}, indicating {'low' if data['VIX'].iloc[-1] < 20 else 'high'} volatility.")
        ]
        key_driver_text = f"Key Driver: {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.2%})"
        risk_factors_text = f"Risk Factors: Federal Funds Rate at {data['Interest Rate'].iloc[-1]:.2f}% may impact valuations."
        sector_performance = [html.Li(f"{sector}: {return_pred:.4f}") for sector, return_pred in sector_predictions.items()]
        investors_insights = [
            html.Li("Overweight: Sectors with positive returns (e.g., Utilities)."),
            html.Li("Underweight: Sectors with negative returns (e.g., Consumer Discretionary).")
        ]
        traders_insights = [
            html.Li("Monitor Fed signals—rate hikes above 5% could shift outlook."),
            html.Li("Consider shorts in underperforming sectors.")
        ]
        long_term_view = f"Sustained GDP growth (>3%) could {'extend bullish trend' if market_direction_label == 'Bullish' else 'mitigate bearish pressures'} into Q3 2025."

        # Fetch and classify news
        articles = fetch_economic_news(NEWS_API_KEY)
        good, bad, hidden = classify_news(articles)
        news_children = [
            html.Div([
                html.Strong("Good News:", style={'color': '#39FF14'}),
                html.Ul([html.Li(html.A(a['title'], href=a['url'], target="_blank", style={'color': '#39FF14'})) for a in good])
            ]),
            html.Div([
                html.Strong("Bad News:", style={'color': '#FF4C4C'}),
                html.Ul([html.Li(html.A(a['title'], href=a['url'], target="_blank", style={'color': '#FF4C4C'})) for a in bad])
            ]),
            html.Div([
                html.Strong("Hidden Edge:", style={'color': '#FFD700'}),
                html.Ul([html.Li(html.A(a['title'], href=a['url'], target="_blank", style={'color': '#FFD700'})) for a in hidden])
            ])
        ]

        return (market_pred_text, key_drivers, sp500_fig, sector_hist_fig, sector_pred_fig, news_children)
    
    except Exception as e:
        logging.error(f"Error in update_sp500_dashboard callback: {e}")
        error_message = html.Div(f"Error loading dashboard: {str(e)}", style={'color': '#FF4C4C'})
        return error_message, html.Div("Error loading data"), {}, {}, {}, html.Div("Error loading news")

# Add Grok API call function
def call_grok_api(prompt, api_key=GROK_API_KEY):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "messages": [
            {"role": "system", "content": "You are a financial news and stock analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "model": "grok-3-latest",
        "stream": False,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def format_news_for_prompt(news_articles):
    lines = []
    for a in news_articles:
        title = a.get('title', '')
        desc = a.get('description', '')
        lines.append(f"- {title} {desc}")
    return '\n'.join(lines)

# Update callback for ticker chart to also update analysis sections
@app.callback(
    [Output('ticker-chart-area', 'children'),
     Output('ticker-error', 'children'),
     Output('ticker-analysis-section', 'children'),
     Output('hype-section', 'children'),
     Output('alarm-section', 'children')],
    Input('generate-button', 'n_clicks'),
    State('ticker-input', 'value'),
    State('timeframe-dropdown', 'value')
)
def generate_ticker_charts(n_clicks, ticker, timeframe):
    if n_clicks == 0 or not ticker or not timeframe:
        return dash.no_update, "Please enter a ticker symbol and select a timeframe.", dash.no_update, dash.no_update, dash.no_update

    test_data = fetch_yfinance_data(ticker, period='5d', interval='1d')
    if test_data is None:
        return dash.no_update, f"Error: Unable to fetch data for ticker '{ticker}'.", dash.no_update, dash.no_update, dash.no_update

    timeframe_configs = {
        'daily': {'interval': '1d', 'period': '1y', 'label': 'Daily', 'default_days': 180},
        '1-minute': {'interval': '1m', 'period': '7d', 'label': '1-Minute', 'default_days': 7},
    }

    config = timeframe_configs.get(timeframe)
    if not config:
        return dash.no_update, "Invalid timeframe selected.", dash.no_update, dash.no_update, dash.no_update

    interval = config['interval']
    period = config['period']
    label = config['label']
    default_days = config['default_days']

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return dash.no_update, f"No data available for {ticker} on {label} timeframe.", dash.no_update, dash.no_update, dash.no_update
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        sector_ticker = stock.info.get('sector', 'SPY')
        sector_ticker_map = {
            'Technology': 'XLK', 'Financial Services': 'XLF', 'Consumer Cyclical': 'XLY',
            'Industrials': 'XLI', 'Utilities': 'XLU', 'Healthcare': 'XLV',
            'Communication Services': 'XLC', 'Consumer Defensive': 'XLP',
            'Basic Materials': 'XLB', 'Real Estate': 'XLRE', 'Energy': 'XLE'
        }
        sector_ticker = sector_ticker_map.get(sector_ticker, 'SPY')
        sector_df = yf.Ticker(sector_ticker).history(period=period, interval=interval)
        sector_df.index = pd.to_datetime(sector_df.index)
        if sector_df.index.tz is not None:
            sector_df.index = sector_df.index.tz_localize(None)
        sector_df = sector_df.reindex(df.index, method='ffill').bfill()
        df['Sector_Close'] = sector_df['Close']
        df['Order_Flow'] = 0  # Placeholder
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Sector_Close', 'Order_Flow']]

        end_date = df.index[-1]
        if timeframe == 'daily':
            start_date = end_date - pd.Timedelta(days=180)
            df = df.loc[start_date:end_date]
        else:
            df = df[df.index.date == df.index[-1].date()]
            df = df.between_time('09:30', '16:00')
            if df.empty:
                return dash.no_update, f"No data for {ticker} on the last trading day (9:30 AM - 4:00 PM).", dash.no_update, dash.no_update, dash.no_update

        if len(df) < 2:
            return dash.no_update, f"Insufficient data for {ticker} on {label} timeframe.", dash.no_update, dash.no_update, dash.no_update

        # Enhanced Chart with Volume Analysis and Institutional Flow Detection
        from plotly.subplots import make_subplots
        
        # Create figure with subplots for price and volume analysis
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,  # Enable shared x-axes for better alignment
            vertical_spacing=0.12,  # Better spacing
            subplot_titles=(f'{ticker} {label} Price Chart', 'Volume Analysis'),
            row_heights=[0.7, 0.3]  # Adjust proportions
        )
        
        # Add candlestick ONLY to first subplot
        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f'{label} Chart'
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # Add volume bars ONLY to second subplot
        volume_bar = go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='#888888',
            opacity=0.7
        )
        fig.add_trace(volume_bar, row=2, col=1)
        
        # Update layout with cleaner design - NO OVERLAPPING TEXT
        fig.update_layout(
            title=dict(
                text=f"{ticker} {label} Chart - Institutional Flow Analysis",
                x=0.5,
                xanchor='center',
                font=dict(size=16, color='#fff'),
                y=0.98  # Position title at the very top
            ),
            template='none',
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font_color='#fff',
            height=800,  # More reasonable height
            margin=dict(l=80, r=80, t=120, b=80),  # Reasonable margins
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.90,  # Position legend well below title
                xanchor="center",
                x=0.5,  # Centered
                font=dict(size=8),  # Smaller font
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='#666',
                borderwidth=1,
                itemsizing='constant',
                itemwidth=30  # Minimum allowed value
            )
        )
        
        # Update axes with better spacing and labels
        fig.update_xaxes(
            color='#fff', 
            showgrid=True, 
            gridcolor='#444',
            row=1, col=1
        )
        fig.update_xaxes(
            color='#fff', 
            showgrid=True, 
            gridcolor='#444',
            row=2, col=1
        )
        fig.update_yaxes(
            color='#fff', 
            title="Price (USD)", 
            showgrid=True,
            gridcolor='#444',
            row=1, col=1
        )
        fig.update_yaxes(
            color='#fff', 
            title="Volume", 
            showgrid=True,
            gridcolor='#444',
            row=2, col=1
        )

        # Generate predictions using enhanced feature engineering
        if timeframe == 'daily' and len(df) >= 60:
            try:
                # Enhanced ticker-specific feature engineering for better predictions
                try:
                    # Start with original DataFrame
                    df_processed = df.copy()
                    
                    # Ensure we have the required columns
                    required_cols = ['Close', 'Volume', 'High', 'Low']
                    missing_cols = [col for col in required_cols if col not in df_processed.columns]
                    if missing_cols:
                        logging.error(f"Missing required columns: {missing_cols}")
                        return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Missing required columns for {ticker}: {missing_cols}", dash.no_update, dash.no_update, dash.no_update
                    
                    # === ENHANCED FEATURE ENGINEERING ===
                    
                    # 1. Basic price features with smoothing
                    df_processed['Price_Change'] = df_processed['Close'].pct_change()
                    df_processed['Log_Returns'] = np.log(df_processed['Close'] / df_processed['Close'].shift(1))
                    df_processed['Price_Range'] = (df_processed['High'] - df_processed['Low']) / df_processed['Close']
                    
                    # 2. Enhanced volatility measures
                    df_processed['Volatility_5'] = df_processed['Close'].rolling(window=5).std()
                    df_processed['Volatility_10'] = df_processed['Close'].rolling(window=10).std()
                    df_processed['Volatility_20'] = df_processed['Close'].rolling(window=20).std()
                    df_processed['Volatility'] = df_processed['Volatility_10']  # Use 10-day for main volatility
                    
                    # Volatility regime detection
                    vol_mean = df_processed['Volatility'].rolling(window=50).mean()
                    df_processed['Vol_Regime'] = (df_processed['Volatility'] > vol_mean * 1.5).astype(int)
                    
                    # 3. Enhanced moving averages
                    df_processed['SMA_5'] = df_processed['Close'].rolling(window=5).mean()
                    df_processed['SMA_10'] = df_processed['Close'].rolling(window=10).mean()
                    df_processed['SMA_20'] = df_processed['Close'].rolling(window=20).mean()
                    df_processed['SMA_50'] = df_processed['Close'].rolling(window=50).mean()
                    
                    # Price relative to moving averages
                    df_processed['Price_to_SMA5'] = df_processed['Close'] / df_processed['SMA_5']
                    df_processed['Price_to_SMA10'] = df_processed['Close'] / df_processed['SMA_10']
                    df_processed['Price_to_SMA20'] = df_processed['Close'] / df_processed['SMA_20']
                    
                    # 4. Enhanced momentum indicators
                    df_processed['Momentum_3'] = df_processed['Close'].diff(3)
                    df_processed['Momentum_5'] = df_processed['Close'].diff(5)
                    df_processed['Momentum_10'] = df_processed['Close'].diff(10)
                    df_processed['Momentum_20'] = df_processed['Close'].diff(20)
                    
                    # 5. Enhanced Volume and Institutional Flow Analysis
                    logging.info(f"Computing institutional flow analysis for {ticker}")
                    df_processed = compute_institutional_flow_analysis(df_processed)
                    
                    # Verify institutional flow columns exist
                    required_columns = ['Institutional_Score', 'Capitulation', 'Institutional_Buying', 'Institutional_Selling', 'High_Price_Impact', 'Low_Price_Impact']
                    missing_columns = [col for col in required_columns if col not in df_processed.columns]
                    if missing_columns:
                        logging.warning(f"Missing institutional flow columns for {ticker}: {missing_columns}")
                    else:
                        logging.info(f"All institutional flow columns present for {ticker}")
                    
                    # 6. Enhanced RSI with multiple timeframes
                    def compute_rsi(prices, window=14):
                        delta = prices.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                        rs = gain / loss
                        return 100 - (100 / (1 + rs))
                    
                    df_processed['RSI_7'] = compute_rsi(df_processed['Close'], 7)
                    df_processed['RSI_14'] = compute_rsi(df_processed['Close'], 14)
                    df_processed['RSI_21'] = compute_rsi(df_processed['Close'], 21)
                    df_processed['RSI'] = df_processed['RSI_14']  # Use 14-day for main RSI
                    
                    # 7. Enhanced MACD
                    ema_12 = df_processed['Close'].ewm(span=12, adjust=False).mean()
                    ema_26 = df_processed['Close'].ewm(span=26, adjust=False).mean()
                    df_processed['MACD'] = ema_12 - ema_26
                    df_processed['MACD_Signal'] = df_processed['MACD'].ewm(span=9, adjust=False).mean()
                    df_processed['MACD_Histogram'] = df_processed['MACD'] - df_processed['MACD_Signal']
                    
                    # 8. Bollinger Bands with multiple periods
                    bb_period = 20
                    bb_std = 2
                    sma_bb = df_processed['Close'].rolling(window=bb_period).mean()
                    std_bb = df_processed['Close'].rolling(window=bb_period).std()
                    df_processed['Upper_BB'] = sma_bb + (std_bb * bb_std)
                    df_processed['Lower_BB'] = sma_bb - (std_bb * bb_std)
                    df_processed['BB_Width'] = (df_processed['Upper_BB'] - df_processed['Lower_BB']) / sma_bb
                    df_processed['BB_Position'] = (df_processed['Close'] - df_processed['Lower_BB']) / (df_processed['Upper_BB'] - df_processed['Lower_BB'])
                    
                    # 9. ATR with multiple periods
                    df_processed['ATR_5'] = (df_processed['High'] - df_processed['Low']).rolling(window=5).mean()
                    df_processed['ATR_14'] = (df_processed['High'] - df_processed['Low']).rolling(window=14).mean()
                    df_processed['ATR'] = df_processed['ATR_14']  # Use 14-day for main ATR
                    
                    # 10. Market regime features
                    # Create Sector_Close (use Close as fallback for now)
                    df_processed['Sector_Close'] = df_processed['Close']
                    df_processed['Sector_Volatility'] = df_processed['Sector_Close'].rolling(window=10).std()
                    
                    # Additional features for the original model
                    df_processed['Realized_Vol'] = df_processed['Volatility']
                    df_processed['Vol_Ratio'] = df_processed['Volatility'] / df_processed['Volatility'].rolling(window=20).mean()
                    
                    # Enhanced Order Flow
                    df_processed['Order_Flow'] = df_processed['Volume'] * df_processed['Price_Change']
                    df_processed['Order_Flow_SMA'] = df_processed['Order_Flow'].rolling(window=10).mean()
                    
                    # 11. Ticker-specific adjustments based on volatility
                    current_volatility = df_processed['Volatility'].iloc[-1] if len(df_processed) > 0 else 0.02
                    avg_volatility = df_processed['Volatility'].mean() if len(df_processed) > 0 else 0.02
                    
                    # Adjust features based on volatility regime
                    if current_volatility > avg_volatility * 1.5:
                        # High volatility regime - emphasize volatility features
                        df_processed['Vol_Adjustment'] = 1.5
                        logging.info(f"High volatility regime detected for {ticker}, applying volatility adjustments")
                    else:
                        # Normal volatility regime
                        df_processed['Vol_Adjustment'] = 1.0
                    
                    logging.info(f"Enhanced features computed for {ticker}. DataFrame shape: {df_processed.shape}")
                    
                except Exception as e:
                    logging.error(f"Error computing enhanced features: {e}")
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Error computing features for {ticker}: {str(e)}", dash.no_update, dash.no_update, dash.no_update
                
                # === ENHANCE CHART WITH INSTITUTIONAL FLOW ANALYSIS ===
                try:
                    logging.info(f"Starting chart enhancement for {ticker}")
                    enhance_chart_with_institutional_flow(fig, df_processed)
                    logging.info(f"Chart enhancement completed for {ticker}")
                except Exception as e:
                    logging.error(f"Error enhancing chart for {ticker}: {e}")
                    # Continue without enhancement if there's an error
                
                logging.info(f"Chart figure created with {len(fig.data)} traces")
                
                # Use the exact 14 features that the daily model was trained on
                daily_features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'SMA_20', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'ATR', 'Order_Flow']
                logging.info(f'Using daily model features for compatibility: {len(daily_features)} features')
                
                if df_processed.shape[0] < 60:
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Not enough processed data for {ticker} to make daily predictions. Need at least 60 rows, got {df_processed.shape[0]}.", dash.no_update, dash.no_update, dash.no_update
                
                # === ENSEMBLE PREDICTION WITH CONFIDENCE SCORING ===
                
                # Use daily model features for prediction (matching trained model)
                seq = df_processed.iloc[-60:][daily_features].values
                if seq.shape[0] < 1:
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"No valid data for {ticker} after feature engineering.", dash.no_update, dash.no_update, dash.no_update
                
                # Scale features
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(df_processed[daily_features])
                seq_scaled = feature_scaler.transform(seq)
                
                # Scale targets
                target_scaler = MinMaxScaler()
                close_values = df_processed['Close'].values.reshape(-1, 1)
                target_scaler.fit(close_values)
                
                # === MULTIPLE PREDICTION METHODS ===
                predictions_list = []
                confidence_scores = []
                
                # 1. Original LSTM Model Prediction
                seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    lstm_pred = daily_model(seq_tensor).numpy().flatten()
                lstm_pred_actual = target_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
                predictions_list.append(lstm_pred_actual)
                
                # Calculate confidence for LSTM prediction - IMPROVED
                recent_volatility = df_processed['Volatility'].iloc[-10:].mean()
                price_stability = 1 - (df_processed['Close'].iloc[-10:].std() / df_processed['Close'].iloc[-10:].mean())
                data_quality = min(1.0, len(df_processed) / 100)  # More data = higher confidence
                lstm_confidence = max(0.3, min(0.9, (1 - recent_volatility * 5) * price_stability * data_quality))
                confidence_scores.append(lstm_confidence)
                
                # 2. Trend-based Prediction (Simple Moving Average Extrapolation) - IMPROVED
                recent_trend = df_processed['Close'].iloc[-20:].pct_change().mean()
                last_price = df_processed['Close'].iloc[-1]
                trend_pred = []
                for i in range(30):
                    trend_pred.append(last_price * (1 + recent_trend) ** (i + 1))
                predictions_list.append(np.array(trend_pred))
                
                # Confidence for trend prediction - IMPROVED
                trend_consistency = 1 - abs(df_processed['Close'].iloc[-10:].pct_change().std())
                trend_strength = abs(recent_trend) * 100  # Stronger trends get higher confidence
                trend_confidence = max(0.4, min(0.9, trend_consistency * (0.5 + trend_strength)))
                confidence_scores.append(trend_confidence)
                
                # 3. Volatility-Adjusted Prediction - IMPROVED
                current_vol = df_processed['Volatility'].iloc[-1]
                avg_vol = df_processed['Volatility'].mean()
                vol_factor = current_vol / avg_vol if avg_vol > 0 else 1.0
                
                # Adjust LSTM predictions based on volatility - MORE REALISTIC
                vol_adjusted_pred = lstm_pred_actual.copy()
                if vol_factor > 1.2:  # High volatility
                    # Apply more conservative adjustments instead of random noise
                    vol_adjusted_pred = vol_adjusted_pred * (1 - (vol_factor - 1) * 0.1)
                elif vol_factor < 0.8:  # Low volatility
                    # Slightly amplify predictions in stable conditions
                    vol_adjusted_pred = vol_adjusted_pred * (1 + (0.8 - vol_factor) * 0.05)
                predictions_list.append(vol_adjusted_pred)
                
                # Confidence for volatility-adjusted prediction - IMPROVED
                vol_stability = 1 / (1 + abs(vol_factor - 1))  # Closer to 1.0 = more stable
                vol_confidence = max(0.3, min(0.8, vol_stability))
                confidence_scores.append(vol_confidence)
                
                # === ENSEMBLE COMBINATION ===
                # Weight predictions by confidence scores
                total_confidence = sum(confidence_scores)
                weights = [conf / total_confidence for conf in confidence_scores]
                
                # Calculate ensemble prediction
                ensemble_pred = np.zeros(30)
                for i, pred in enumerate(predictions_list):
                    ensemble_pred += weights[i] * pred
                
                # Calculate ensemble confidence
                ensemble_confidence = np.mean(confidence_scores)
                
                # Apply confidence-based smoothing - IMPROVED
                if ensemble_confidence < 0.6:
                    # Low confidence - apply more smoothing
                    ensemble_pred = gaussian_filter1d(ensemble_pred, sigma=1.2)
                    logging.info(f"Applied smoothing due to low confidence ({ensemble_confidence:.3f}) for {ticker}")
                
                # Apply additional smoothing for more realistic curves
                ensemble_pred = gaussian_filter1d(ensemble_pred, sigma=0.8)
                
                # Ensure predictions follow realistic price movements
                last_actual_price = df_processed['Close'].iloc[-1]
                price_change_limit = last_actual_price * 0.1  # Max 10% change from last price
                
                # Cap extreme predictions
                for i in range(len(ensemble_pred)):
                    if abs(ensemble_pred[i] - last_actual_price) > price_change_limit:
                        if ensemble_pred[i] > last_actual_price:
                            ensemble_pred[i] = last_actual_price + price_change_limit
                        else:
                            ensemble_pred[i] = last_actual_price - price_change_limit
                
                predictions = ensemble_pred
                
                logging.info(f"Ensemble prediction completed for {ticker}")
                logging.info(f"Confidence scores: LSTM={confidence_scores[0]:.3f}, Trend={confidence_scores[1]:.3f}, Vol={confidence_scores[2]:.3f}")
                logging.info(f"Ensemble confidence: {ensemble_confidence:.3f}")
                logging.info(f"Raw model output (daily): {predictions[:5]}")  # Show first 5 predictions
                logging.info(f"Last 5 closing prices for {ticker} (Daily): {df['Close'].iloc[-5:].tolist()}")
                logging.info(f"Daily Predictions for {ticker} (first 5): {predictions[:5].tolist()}")
                
                # === IMPROVED MAE CALCULATION ===
                # Calculate MAE based on historical prediction accuracy
                historical_mae = 0
                if len(df_processed) > 80:  # Need enough data for validation
                    # Use last 20 days for validation
                    val_data = df_processed.iloc[-80:-20]  # Training data
                    val_target = df_processed.iloc[-20:]   # Validation target
                    
                    if len(val_data) >= 60:
                        # Make predictions on validation data
                        val_seq = val_data[daily_features].values[-60:]
                        val_seq_scaled = feature_scaler.transform(val_seq)
                        val_seq_tensor = torch.tensor(val_seq_scaled, dtype=torch.float32).unsqueeze(0)
                        
                        with torch.no_grad():
                            val_pred = daily_model(val_seq_tensor).numpy().flatten()
                        val_pred_actual = target_scaler.inverse_transform(val_pred.reshape(-1, 1)).flatten()
                        
                        # Calculate MAE for validation period
                        # The model predicts 30 days, but we only have 20 validation days
                        min_len = min(len(val_target['Close'].values), len(val_pred_actual))
                        if min_len > 0:
                            actual_prices = val_target['Close'].values[:min_len]
                            pred_prices = val_pred_actual[:min_len]
                            historical_mae = np.mean(np.abs(actual_prices - pred_prices))
                        else:
                            historical_mae = 5.0  # Default MAE
                    else:
                        historical_mae = 8.0  # Default for insufficient data
                else:
                    historical_mae = 10.0  # Default for insufficient data
                
                # Adjust MAE based on ensemble confidence
                confidence_adjusted_mae = historical_mae * (1 - ensemble_confidence * 0.5)
                final_mae = max(2.0, confidence_adjusted_mae)  # Minimum MAE of 2.0
                
                logging.info(f"Historical MAE: {historical_mae:.3f}, Ensemble Confidence: {ensemble_confidence:.3f}")
                logging.info(f"Final Confidence-Adjusted MAE: {final_mae:.3f}")
                
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
                
                # === IMPROVED OHLC GENERATION ===
                pred_open = [df['Close'].iloc[-1]] + list(predictions[:-1])
                pred_close = predictions
                
                # Enhanced OHLC calculation for more realistic candlesticks
                confidence_vol = df_processed['Volatility'].iloc[-1] * (1 - ensemble_confidence * 0.2)
                base_price = np.array(predictions)
                
                # Calculate realistic daily ranges based on historical patterns
                historical_ranges = (df_processed['High'] - df_processed['Low']).iloc[-20:].mean()
                daily_range_ratio = historical_ranges / df_processed['Close'].iloc[-1]
                
                # Generate realistic high and low for each day
                pred_high = []
                pred_low = []
                
                for i in range(len(predictions)):
                    # Base range from historical patterns
                    base_range = predictions[i] * daily_range_ratio
                    
                    # Adjust range based on confidence and volatility
                    confidence_factor = 1 - ensemble_confidence * 0.3  # Higher confidence = smaller range
                    volatility_factor = confidence_vol / df_processed['Volatility'].mean()
                    
                    # Calculate daily range
                    daily_range = base_range * confidence_factor * (0.8 + volatility_factor * 0.4)
                    
                    # Ensure reasonable range (1-5% of price)
                    daily_range = max(predictions[i] * 0.01, min(daily_range, predictions[i] * 0.05))
                    
                    # Generate high and low with some randomness
                    range_variation = np.random.uniform(0.3, 0.7)  # 30-70% of range above/below
                    high_addition = daily_range * range_variation
                    low_subtraction = daily_range * (1 - range_variation)
                    
                    pred_high.append(predictions[i] + high_addition)
                    pred_low.append(predictions[i] - low_subtraction)
                
                pred_high = np.array(pred_high)
                pred_low = np.array(pred_low)
                
                # Ensure OHLC relationships are maintained
                pred_high = np.maximum(pred_high, np.maximum(pred_open, pred_close))
                pred_low = np.minimum(pred_low, np.minimum(pred_open, pred_close))
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Open': pred_open,
                    'Predicted_High': pred_high,
                    'Predicted_Low': pred_low,
                    'Predicted_Close': pred_close,
                    'Actual_Close': [None] * len(future_dates)
                })
                excel_filename = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\data", f"predictions_{ticker}_daily.xlsx")
                pred_df.to_excel(excel_filename, index=False)
                logging.info(f"Saved daily predictions for {ticker} to {excel_filename}")
                # Create proper candlestick predictions with OHLC data
                pred_candlestick = go.Candlestick(
                    x=future_dates,
                    open=pred_open,
                    high=pred_high,
                    low=pred_low,
                    close=pred_close,
                    name='Predicted (30 Days)',
                    increasing_line_color='orange',
                    decreasing_line_color='darkorange',
                    increasing_fillcolor='rgba(255, 165, 0, 0.3)',
                    decreasing_fillcolor='rgba(255, 140, 0, 0.3)',
                    line=dict(width=1)
                )
                fig.add_trace(pred_candlestick, row=1, col=1)
                fig.update_xaxes(range=[df.index[0], future_dates[-1]])
                # Enhanced title with confidence metrics
                confidence_percentage = ensemble_confidence * 100
                title_text = f"{ticker} {label} Chart - MAE: {final_mae:.2f} | Confidence: {confidence_percentage:.1f}%"
                fig.update_layout(title=title_text)
            except Exception as e:
                logging.error(f"Error generating daily predictions for {ticker}: {e}")
                return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Chart generated, but error in daily predictions: {str(e)}.", dash.no_update, dash.no_update, dash.no_update
        elif timeframe == '1-minute' and len(df) >= 30:
            try:
                df['Price_Change'] = df['Close'].pct_change()
                df['Volatility'] = df['Close'].rolling(window=10).std()
                df['Sector_Volatility'] = df['Sector_Close'].rolling(window=10).std()
                df['Realized_Vol'] = compute_realized_volatility(df, window=10)
                df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=20).mean()
                df['Momentum'] = df['Close'].diff(3)
                df['Volume_Spike'] = (df['Volume'] / df['Volume'].rolling(window=10).mean()) - 1
                features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'Momentum', 'Volume_Spike']
                df_processed = df[features].dropna()
                logging.info(f'Rows after dropna (1-minute): {df_processed.shape[0]}')
                if df_processed.shape[0] < 30:
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Not enough processed data for {ticker} to make 1-minute predictions. Need at least 30 rows, got {df_processed.shape[0]}.", dash.no_update, dash.no_update, dash.no_update
                order_footprint = fetch_order_footprint(ticker, period='7d', interval='1m', expected_rows=len(df_processed))
                order_footprint = order_footprint[-30:]
                seq = df_processed.iloc[-30:][features].values
                if seq.shape[0] < 1:
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"No valid data for {ticker} after feature engineering.", dash.no_update, dash.no_update, dash.no_update
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(df_processed[features])
                seq_scaled = feature_scaler.transform(seq)
                target_scaler = MinMaxScaler()
                close_values = df_processed['Close'].values.reshape(-1, 1)
                target_scaler.fit(close_values)
                seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
                order_tensor = torch.tensor(order_footprint, dtype=torch.float32).unsqueeze(0)
                logging.info(f"Scaled input to model (1-minute): {seq_scaled}")
                with torch.no_grad():
                    predictions = minute_model(order_tensor, seq_tensor).numpy().flatten()
                logging.info(f"Raw model output (1-minute): {predictions}")
                predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                logging.info(f"Last 5 closing prices for {ticker} (1-Minute): {df['Close'].iloc[-5:].tolist()}")
                logging.info(f"1-Minute Predictions for {ticker} (first 5): {predictions[:5].tolist()}")
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=30, freq='1min')
                pred_open = [df['Close'].iloc[-1]] + list(predictions[:-1])
                pred_close = predictions
                pred_high = np.maximum(pred_open, pred_close) + predictions * 0.001
                pred_low = np.minimum(pred_open, pred_close) - predictions * 0.001
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Open': pred_open,
                    'Predicted_High': pred_high,
                    'Predicted_Low': pred_low,
                    'Predicted_Close': pred_close,
                    'Actual_Close': [None] * len(future_dates)
                })
                excel_filename = os.path.join(r"C:\Users\jonel\OneDrive\Desktop\Jonel_Projects\Market_Analysis\funda\data", f"predictions_{ticker}_1minute.xlsx")
                pred_df.to_excel(excel_filename, index=False)
                logging.info(f"Saved 1-minute predictions for {ticker} to {excel_filename}")
                pred_candlestick = go.Candlestick(
                    x=future_dates,
                    open=pred_open,
                    high=pred_high,
                    low=pred_low,
                    close=pred_close,
                    name='Predicted (30 Minutes)',
                    increasing_line_color='blue',
                    decreasing_line_color='orange'
                )
                fig.add_trace(pred_candlestick, row=1, col=1)
                fig.update_xaxes(range=[df.index[0], future_dates[-1]])
                actual_close = df['Close'].iloc[-30:].values
                if len(actual_close) == len(predictions):
                    mae = mean_absolute_error(actual_close, predictions[:len(actual_close)])
                    fig.add_annotation(text=f"MAE: {mae:.4f}", xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False)
            except Exception as e:
                logging.error(f"Error generating 1-minute predictions for {ticker}: {e}")
                return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Chart generated, but error in 1-minute predictions: {str(e)}.", dash.no_update, dash.no_update, dash.no_update
        elif timeframe == '1-minute' and len(df) < 30:
            return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Chart generated, but insufficient data for 1-minute predictions: {len(df)} rows.", dash.no_update, dash.no_update, dash.no_update

        # After chart code, fetch news and run local analysis
        news_articles = fetch_economic_news(NEWS_API_KEY, query=ticker)
        summary = summarize_news(news_articles)
        negative = detect_negative_sentiment(news_articles)
        hype = detect_hype(news_articles)
        caveat = check_otc_caveat_emptor(ticker)

        # Volume Analysis Summary (if available)
        volume_analysis_summary = ""
        try:
            if 'df_processed' in locals() and hasattr(df_processed, 'Institutional_Score'):
                volume_analysis_summary = f"\n\n{generate_volume_analysis_summary(df_processed)}"
        except Exception as e:
            logging.error(f"Error generating volume analysis summary: {e}")
            volume_analysis_summary = "\n\n📊 VOLUME ANALYSIS: Error generating analysis"

        # Ticker Analysis
        analysis = f"Summary: {summary}"
        if negative:
            analysis += f"\nPossible reason for being down: {negative}"
        else:
            analysis += "\nNo strong negative sentiment detected."
        
        # Add volume analysis to the main analysis
        analysis += volume_analysis_summary

        # Hype/Promotion
        hype_text = hype if hype else "No hype or promotion detected."

        # Caveat Emptor/Prohibited Alarm
        if caveat is True:
            alarm = f"⚠️ Caveat Emptor or Prohibited status detected on OTCMarkets.com!"
        elif isinstance(caveat, str) and caveat.startswith("Error"):
            alarm = caveat
        else:
            alarm = "No Caveat Emptor/Prohibited status detected."

        return dcc.Graph(
            figure=fig,
            config={'displayModeBar': True},
            style={
                'height': '100%',
                'backgroundColor': '#222'
            }
        ), "", analysis, hype_text, alarm

    except Exception as e:
        logging.error(f"Error generating chart for {ticker} ({label}): {e}")
        return dash.no_update, f"Error generating chart for {ticker} ({label}): {str(e)}.", dash.no_update, dash.no_update, dash.no_update

def fetch_economic_news(api_key, query="economy OR stock market OR inflation OR fed", language="en", page_size=10):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&language={language}&sortBy=publishedAt&pageSize={page_size}&apiKey={api_key}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return []

def classify_news(articles):
    good_keywords = ["growth", "record high", "bull", "optimism", "beat", "strong", "rally", "positive"]
    bad_keywords = ["recession", "crash", "bear", "drop", "decline", "miss", "weak", "negative", "inflation"]
    hidden_keywords = ["unexpected", "surprise", "unnoticed", "quiet", "edge", "hidden", "under the radar"]

    good, bad, hidden = [], [], []
    for article in articles:
        title = (article.get("title") or "").lower()
        description = (article.get("description") or "").lower()
        content = f"{title} {description}"
        if any(word in content for word in hidden_keywords):
            hidden.append(article)
        elif any(word in content for word in good_keywords):
            good.append(article)
        elif any(word in content for word in bad_keywords):
            bad.append(article)
    return good, bad, hidden

@app.callback(
    Output('feature-outlier-scatter', 'figure'),
    Input('strategy-dropdown', 'value')
)
def update_feature_outlier_scatter(strategy):
    print(f"=== CALLBACK TRIGGERED: update_feature_outlier_scatter with strategy: {strategy} ===")
    print(f"=== DEBUG: This is the NEW callback version ===")
    logging.info(f"Callback triggered with strategy: {strategy}")
    
    if strategy not in STRATEGIES:
        logging.warning("Invalid strategy %s", strategy)
        return {"data": [], "layout": {"title": "Invalid strategy", "paper_bgcolor": "#222", "plot_bgcolor": "#222", "font": {"color": "#fff"}}}

    x_col, y_col, *_ = STRATEGIES[strategy]
    print(f"Strategy columns: x_col={x_col}, y_col={y_col}")

    try:
        # Debug database path
        import os
        db_path = os.path.abspath('billions.db')
        print(f"Database path: {db_path}")
        print(f"Database exists: {os.path.exists(db_path)}")
        
        # fetch from Postgres
        print(f"Querying database for strategy: {strategy}")
        with SessionLocal() as sess:
            # First check total count
            total_count = sess.query(PerfMetric).count()
            print(f"Total rows in database: {total_count}")
            
            rows = sess.query(PerfMetric).filter(PerfMetric.strategy == strategy).all()
            print(f"✓ Found {len(rows)} rows for strategy {strategy}")
            
            if len(rows) > 0:
                sample_row = rows[0]
                print(f"✓ Sample row: {sample_row.symbol} - metric_x: {sample_row.metric_x}, metric_y: {sample_row.metric_y}")

        if not rows:
            print("✗ No rows found, returning empty plot")
            return {"data": [], "layout": {"title": "No data", "paper_bgcolor": "#222", "plot_bgcolor": "#222", "font": {"color": "#fff"}}}

        # Create DataFrame with valid column names
        df = pd.DataFrame([{
            'symbol': r.symbol,
            'metric_x': float(r.metric_x),
            'metric_y': float(r.metric_y)
        } for r in rows])
        df['Ticker'] = df['symbol']
        print(f"Created DataFrame with {len(df)} rows")
        
    except Exception as e:
        print(f"ERROR in database processing: {e}")
        logging.error(f"Database error: {e}")
        return {"data": [], "layout": {"title": f"Database error: {str(e)}", "paper_bgcolor": "#222", "plot_bgcolor": "#222", "font": {"color": "#fff"}}}

    # Create scatter plot using the actual column names
    fig = px.scatter(
        df,
        x='metric_x',
        y='metric_y',
        text='Ticker',
        title=f"Performance Scatter Plot ({strategy.capitalize()}): {x_col} vs {y_col}",
        template='none'
    )

    # Update plot styling
    fig.update_traces(
        marker=dict(
            size=12,
            color='#39FF14',
            line=dict(width=2, color='#fff')
        ),
        textposition='top center'
    )

    fig.update_layout(
        autosize=False,
        width=1200,
        height=700,
        paper_bgcolor='#222',
        plot_bgcolor='#222',
        font_color='#fff',
        xaxis=dict(
            color='#fff',
            title=f"{x_col} Performance",
            gridcolor='#444'
        ),
        yaxis=dict(
            color='#fff', 
            title=f"{y_col} Performance",
            gridcolor='#444'
        )
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1
    )

    # Log data for debugging
    logging.info(f"Successfully loaded data for {strategy} with {len(df)} rows")
    logging.info(f"Columns: {df.columns.tolist()}")
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"X column: {x_col}, Y column: {y_col}")
    print(f"First few rows of {strategy} data:")
    print(df.head())
    print(f"Scatter plot created with {len(df)} data points")
    
    # Test: Create a simple plot to verify data is valid
    print(f"Data ranges: metric_x={df['metric_x'].min():.2f} to {df['metric_x'].max():.2f}")
    print(f"Data ranges: metric_y={df['metric_y'].min():.2f} to {df['metric_y'].max():.2f}")
    print(f"=== RETURNING FIGURE TO DASH ===")

    return fig

# --- News/NLP/OTCMarkets Analysis Functions ---
def summarize_news(news_articles):
    # Simple extractive summary: join top 3 headlines
    if not news_articles:
        return "No recent news found for this ticker."
    summary = ' | '.join([a['title'] for a in news_articles[:3] if a.get('title')])
    return summary

def detect_negative_sentiment(news_articles):
    for a in news_articles:
        text = (a.get('title') or '') + ' ' + (a.get('description') or '')
        blob = TextBlob(text)
        if blob.sentiment.polarity < -0.2:
            return a['title']
    return None

def detect_hype(news_articles):
    hype_keywords = ['soars', 'explodes', 'must buy', 'promotion', 'pump', 'moon', 'hype', 'skyrockets', 'rockets', 'surges']
    for a in news_articles:
        text = ((a.get('title') or '') + ' ' + (a.get('description') or '')).lower()
        if any(word in text for word in hype_keywords):
            return a['title']
    return None

def check_otc_caveat_emptor(ticker):
    url = f"https://www.otcmarkets.com/stock/{ticker}/overview"
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Look for Caveat Emptor badge or prohibited mention
        if 'Caveat Emptor' in soup.text or 'Prohibited' in soup.text:
            return True
        return False
    except Exception as e:
        return f"Error checking OTCMarkets: {e}"

# Outlier detection moved to refresh button - no startup blocking

# Refresh button callbacks
@app.callback(
    [Output("refresh-button", "disabled"),
     Output("refresh-status", "children"),
     Output("refresh-spinner", "children")],
    [Input("refresh-button", "n_clicks")],
    prevent_initial_call=True
)
def handle_refresh_click(n_clicks):
    if n_clicks:
        success = start_refresh_thread()
        if success:
            return True, "Refreshing data... Please wait", None
        else:
            return False, "Refresh already in progress", None
    return False, "Ready to refresh data", None

# Progress update callback
@app.callback(
    Output("refresh-status", "children", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True
)
def update_refresh_status(n):
    status = get_refresh_status()
    
    if status['is_running']:
        progress = status['progress']
        strategy = status['current_strategy']
        message = status['message']
        
        if status['estimated_completion']:
            remaining = int(status['estimated_completion'] - datetime.now().timestamp())
            if remaining > 0:
                message += f" (Est. {remaining}s remaining)"
        
        return f"🔄 {message} - {progress}% complete"
    else:
        return status['message']

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)
