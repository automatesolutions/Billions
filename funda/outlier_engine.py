import io
import requests
import yfinance as yf
import pandas as pd
from scipy.stats import zscore
from sqlalchemy import delete
from db.core import SessionLocal
from db.models import PerfMetric
import logging
import time
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), "..", "outlier", ".env")
load_dotenv(env_path)
api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

STRATEGIES = {
    "scalp":   ("1m", "1w", 21, 5, 1e9),      # 1B market cap
    "swing":   ("3m", "1m", 63, 21, 2e9),     # 2B market cap  
    "longterm":("1y", "6m", 252, 126, 10e9),  # 10B market cap
}

def _fetch_nasdaq_tickers():
    """Fetch NASDAQ tickers from Alpha Vantage API"""
    if not api_key:
        logging.warning("[Outlier] No Alpha Vantage API key found, using fallback tickers")
        return ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AMD', 'NFLX', 'INTC', 'CSCO', 'ADBE', 'PEP', 'AVGO', 'COST']
    
    url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}"
    logging.info("[Outlier] Fetching NASDAQ tickers from Alpha Vantage...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            nasdaq_data = pd.read_csv(io.StringIO(response.text))
            
            # Filter for NASDAQ and active tickers
            if 'exchange' in nasdaq_data.columns and 'status' in nasdaq_data.columns:
                nasdaq_tickers = nasdaq_data[
                    (nasdaq_data['exchange'] == 'NASDAQ') & 
                    (nasdaq_data['status'] == 'Active')
                ]['symbol'].tolist()
            else:
                nasdaq_tickers = nasdaq_data['symbol'].tolist()
            
            logging.info("[Outlier] Found %d active NASDAQ tickers", len(nasdaq_tickers))
            return nasdaq_tickers
        else:
            logging.error("[Outlier] Failed to fetch data, status code: %d", response.status_code)
            return []
    except Exception as e:
        logging.error("[Outlier] Error fetching NASDAQ tickers: %s", e)
        return []

def _filter_valid_tickers(tickers):
    """Filter out tickers with non-alphabetic characters or unusual lengths."""
    valid = []
    for t in tickers:
        if pd.isna(t):
            continue
        t = str(t)
        if t.isalpha() and 1 < len(t) <= 5:
            valid.append(t)
    return valid

def _filter_high_volume_tickers(tickers, min_volume=1000000, min_market_cap=1e9, batch_size=50):
    """Filter tickers based on volume and market cap."""
    filtered_tickers = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logging.info("[Outlier] Screening batch %d/%d for volume and market cap...", 
                    i//batch_size + 1, (len(tickers)//batch_size) + 1)
        
        for ticker in batch:
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info
                avg_volume = info.get('averageDailyVolume10Day', 0)
                market_cap = info.get('marketCap', 0)
                
                if avg_volume >= min_volume and market_cap >= min_market_cap:
                    filtered_tickers.append(ticker)
                    logging.debug("[Outlier] Added %s (Volume: %d, Market Cap: %.2fB)", 
                                ticker, avg_volume, market_cap/1e9)
                else:
                    logging.debug("[Outlier] Skipped %s (Volume: %d, Market Cap: %.2fB)", 
                                ticker, avg_volume, market_cap/1e9)
            except Exception as e:
                logging.debug("[Outlier] Error screening %s: %s", ticker, e)
        time.sleep(1)  # Rate limiting
    
    return filtered_tickers

def _fetch_batch(tickers: list[str], lookback_days: int):
    """Download Close price data from Yahoo Finance"""
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logging.info("[Outlier] Fetching batch %d/%d...", i//batch_size + 1, (len(tickers)//batch_size) + 1)
        
        try:
            batch_data = yf.download(batch, period=f"{lookback_days+5}d", interval="1d", group_by="ticker")
            for ticker in batch:
                if ticker in batch_data and "Close" in batch_data[ticker]:
                    all_data[ticker] = batch_data[ticker]["Close"].dropna()
        except Exception as e:
            logging.error("[Outlier] Error fetching batch %d: %s", i//batch_size + 1, e)
        time.sleep(1)  # Rate limiting
    
    return all_data

def _calc_pct(ser: pd.Series, lookback: int):
    return (ser.iloc[-1] - ser.iloc[-lookback]) / ser.iloc[-lookback] * 100

def run_outlier_detection(strategy: str, tickers: list[str] = None):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy}")
    
    x_lab, y_lab, back_x, back_y, min_market_cap = STRATEGIES[strategy]
    
    # If no tickers provided, fetch NASDAQ tickers
    if tickers is None:
        logging.info("[Outlier] No tickers provided, fetching NASDAQ tickers for %s", strategy)
        nasdaq_tickers = _fetch_nasdaq_tickers()
        if not nasdaq_tickers:
            logging.error("[Outlier] No NASDAQ tickers found for %s", strategy)
            return
        
        # Filter valid tickers
        valid_tickers = _filter_valid_tickers(nasdaq_tickers)
        logging.info("[Outlier] %d valid tickers after filtering", len(valid_tickers))
        
        # Filter by volume and market cap
        tickers = _filter_high_volume_tickers(valid_tickers, min_market_cap=min_market_cap)
        logging.info("[Outlier] %d tickers after volume/market cap filtering for %s", len(tickers), strategy)
    
    if not tickers:
        logging.warning("[Outlier] No tickers available for %s", strategy)
        return
    
    logging.info("[Outlier] Downloading prices for %s (%d tickers)", strategy, len(tickers))
    prices = _fetch_batch(tickers, back_x)

    rows = []
    for t, ser in prices.items():
        if len(ser) < back_x + 1:
            logging.debug("[Outlier] Skipping %s (insufficient data: %d days)", t, len(ser))
            continue
        m_x = _calc_pct(ser, back_x)
        m_y = _calc_pct(ser, back_y)
        rows.append({"symbol": t, "metric_x": m_x, "metric_y": m_y})

    if not rows:
        logging.warning("[Outlier] No rows for strategy %s", strategy)
        return

    df = pd.DataFrame(rows).set_index("symbol")
    z = df.apply(zscore)
    df["z_x"] = z["metric_x"]
    df["z_y"] = z["metric_y"]
    df["is_outlier"] = (df["z_x"].abs() > 2) | (df["z_y"].abs() > 2)

    with SessionLocal() as s:
        s.execute(delete(PerfMetric).where(PerfMetric.strategy == strategy))
        s.bulk_insert_mappings(
            PerfMetric,
            [
                dict(
                    strategy=strategy,
                    symbol=idx,
                    metric_x=row.metric_x,
                    metric_y=row.metric_y,
                    z_x=row.z_x,
                    z_y=row.z_y,
                    is_outlier=row.is_outlier,
                )
                for idx, row in df.iterrows()
            ],
        )
        s.commit()
    logging.info("[Outlier] Stored %d rows for %s", len(df), strategy)
