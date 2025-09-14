import pandas as pd
import numpy as np
import requests
from fredapi import Fred
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
import dash_bootstrap_components as dbc
import torch
import torch.nn as nn
import json
import warnings
from textblob import TextBlob
from bs4 import BeautifulSoup

# Suppress All-NaN warnings
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

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
    })
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

        candlestick = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f'{label} Chart'
        )
        fig = go.Figure(data=[candlestick])
        fig.update_layout(
            title=f"{ticker} {label} Chart",
            template='none',
            xaxis_rangeslider_visible=False,
            xaxis=dict(type='date', tickformat='%Y-%m-%d %H:%M', showticklabels=True),
            yaxis=dict(autorange=True, title="Price (USD)")
        )
        fig.update_layout(
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font_color='#fff',
            xaxis=dict(color='#fff'),
            yaxis=dict(color='#fff')
        )

        # Generate predictions
        if timeframe == 'daily' and len(df) >= 60:
            try:
                df['Price_Change'] = df['Close'].pct_change()
                logging.info(f'Rows after Price_Change: {df.shape[0]}')
                df['Volatility'] = df['Close'].rolling(window=5).std()
                logging.info(f'Rows after Volatility: {df.shape[0]}')
                df['Sector_Volatility'] = df['Sector_Close'].rolling(window=5).std()
                logging.info(f'Rows after Sector_Volatility: {df.shape[0]}')
                df['Realized_Vol'] = compute_realized_volatility(df, window=5)
                logging.info(f'Rows after Realized_Vol: {df.shape[0]}')
                df['Vol_Ratio'] = df['Volatility'] / df['Volatility'].rolling(window=10).mean()
                logging.info(f'Rows after Vol_Ratio: {df.shape[0]}')
                df['SMA_20'] = df['Close'].rolling(window=5).mean()
                logging.info(f'Rows after SMA_20: {df.shape[0]}')
                df['RSI'] = compute_rsi(df['Close'], 5)
                logging.info(f'Rows after RSI: {df.shape[0]}')
                df['MACD'] = df['Close'].ewm(span=6, adjust=False).mean() - df['Close'].ewm(span=13, adjust=False).mean()
                logging.info(f'Rows after MACD: {df.shape[0]}')
                df['Upper_BB'] = df['SMA_20'] + 2 * df['Close'].rolling(window=5).std()
                df['Lower_BB'] = df['SMA_20'] - 2 * df['Close'].rolling(window=5).std()
                logging.info(f'Rows after Bollinger Bands: {df.shape[0]}')
                df['ATR'] = (df['High'].rolling(window=5).max() - df['Low'].rolling(window=5).min())
                logging.info(f'Rows after ATR: {df.shape[0]}')
                df['Imbalance'] = df['High'] - df['Low']
                features = ['Close', 'Volume', 'Price_Change', 'Volatility', 'Sector_Volatility', 'Realized_Vol', 'Vol_Ratio', 'SMA_20', 'RSI', 'MACD', 'Upper_BB', 'Lower_BB', 'ATR', 'Order_Flow']
                df_processed = df[features].dropna()
                logging.info(f'Rows after dropna: {df_processed.shape[0]}')
                if df_processed.shape[0] < 60:
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"Not enough processed data for {ticker} to make daily predictions. Need at least 60 rows, got {df_processed.shape[0]}.", dash.no_update, dash.no_update, dash.no_update
                seq = df_processed.iloc[-60:][features].values
                if seq.shape[0] < 1:
                    return dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '100%', 'backgroundColor': '#222'}), f"No valid data for {ticker} after feature engineering.", dash.no_update, dash.no_update, dash.no_update
                feature_scaler = MinMaxScaler()
                feature_scaler.fit(df_processed[features])
                seq_scaled = feature_scaler.transform(seq)
                target_scaler = MinMaxScaler()
                close_values = df_processed['Close'].values.reshape(-1, 1)
                target_scaler.fit(close_values)
                seq_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)
                logging.info(f"Scaled input to model (daily): {seq_scaled}")
                with torch.no_grad():
                    predictions = daily_model(seq_tensor).numpy().flatten()
                logging.info(f"Raw model output (daily): {predictions}")
                predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
                logging.info(f"Last 5 closing prices for {ticker} (Daily): {df['Close'].iloc[-5:].tolist()}")
                logging.info(f"Daily Predictions for {ticker} (first 5): {predictions[:5].tolist()}")
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')
                pred_open = [df['Close'].iloc[-1]] + list(predictions[:-1])
                pred_close = predictions
                pred_high = np.maximum(pred_open, pred_close) + predictions * 0.005
                pred_low = np.minimum(pred_open, pred_close) - predictions * 0.005
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
                pred_candlestick = go.Candlestick(
                    x=future_dates,
                    open=pred_open,
                    high=pred_high,
                    low=pred_low,
                    close=pred_close,
                    name='Predicted (30 Days)',
                    increasing_line_color='blue',
                    decreasing_line_color='orange'
                )
                fig.add_trace(pred_candlestick)
                fig.update_xaxes(range=[df.index[0], future_dates[-1]])
                # Add MAE if applicable
                actual_close = df['Close'].iloc[-30:].values
                if len(actual_close) == len(predictions):
                    mae = mean_absolute_error(actual_close, predictions[:len(actual_close)])
                    fig.add_annotation(text=f"MAE: {mae:.4f}", xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False)
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
                fig.add_trace(pred_candlestick)
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

        # Ticker Analysis
        analysis = f"Summary: {summary}"
        if negative:
            analysis += f"\nPossible reason for being down: {negative}"
        else:
            analysis += "\nNo strong negative sentiment detected."

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
                'position': 'absolute',
                'top': 0,
                'left': 0,
                'width': '100%',
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
    # Map strategy to file and column names
    file_map = {
        'scalp': 'nasdaq_scalp_performance_metrics.csv',
        'swing': 'nasdaq_swing_performance_metrics.csv', 
        'longterm': 'nasdaq_longterm_performance_metrics.csv'
    }
    axis_map = {
        'scalp': ('1m', '1w'),
        'swing': ('3m', '1m'),
        'longterm': ('1y', '6m')
    }

    # Get file path and columns based on strategy
    if strategy not in file_map:
        logging.warning(f"Invalid strategy '{strategy}' provided, defaulting to 'scalp'")
        strategy = 'scalp'
        
    filename = file_map[strategy]  # Get exact file match for strategy
    filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
    x_col, y_col = axis_map[strategy]  # Get exact columns for strategy

    try:
        # Read CSV file
        df = pd.read_csv(filepath, index_col=0)
        
        # Add ticker column if not present
        if 'Ticker' in df.columns:
            df['Ticker'] = df['Ticker']
        else:
            df['Ticker'] = df.index

        # Validate columns exist
        if x_col not in df.columns or y_col not in df.columns:
            return {
                "data": [],
                "layout": {
                    "title": f"Missing columns ({x_col}, {y_col}) in {filename}",
                    "paper_bgcolor": "#222",
                    "plot_bgcolor": "#222", 
                    "font": {"color": "#fff"}
                }
            }

        # Create scatter plot
        fig = px.scatter(
            df, 
            x=x_col,
            y=y_col,
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
        logging.info(f"Successfully loaded {filename} with columns: {df.columns.tolist()}")
        print(f"First few rows of {filename}:")
        print(df.head())

        return fig

    except Exception as e:
        logging.error(f"Error loading {filename}: {str(e)}")
        return {
            "data": [],
            "layout": {
                "title": f"Error loading {filename}: {str(e)}",
                "paper_bgcolor": "#222",
                "plot_bgcolor": "#222",
                "font": {"color": "#fff"}
            }
        }

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

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)