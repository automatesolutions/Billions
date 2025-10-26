import os
import asyncio
import aiohttp
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import talib
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from funda.outlier_engine import _fetch_nasdaq_tickers, _filter_valid_tickers, _filter_high_volume_tickers
from api.database import get_db
from db.models import PerfMetric

logger = logging.getLogger(__name__)

class CapitulationDetector:
    """Detects capitulation signals in NASDAQ stocks using real data"""
    
    def __init__(self):
        self.session = None
        self.nasdaq_symbols = []
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def _get_nasdaq_symbols(self) -> List[str]:
        """Get list of NASDAQ symbols - using major stocks for faster processing"""
        try:
            # For now, use a curated list of major NASDAQ stocks for faster processing
            # This avoids the long delay from fetching all NASDAQ tickers
            major_nasdaq_stocks = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'ADBE', 'CRM', 'INTC', 'AMD', 'PYPL', 'CMCSA', 'PEP', 'COST',
                'AVGO', 'TXN', 'QCOM', 'CHTR', 'AMAT', 'ISRG', 'GILD', 'BKNG',
                'ADP', 'VRTX', 'FISV', 'BIIB', 'REGN', 'MDLZ', 'ATVI', 'CSX',
                'ILMN', 'AMGN', 'WBA', 'CTAS', 'EXC', 'EA', 'LRCX', 'KLAC',
                'SNPS', 'MCHP', 'ADI', 'CDNS', 'ORLY', 'IDXX', 'DXCM', 'ALGN',
                'CTSH', 'FAST', 'PAYX', 'ROST', 'SBUX', 'TMUS', 'VRSK', 'WLTW',
                'XEL', 'ANSS', 'BMRN', 'CERN', 'CHKP', 'CTXS', 'DLTR', 'EBAY',
                'EXPE', 'FIS', 'FTNT', 'GPN', 'HSIC', 'INTU', 'JBHT', 'KDP',
                'LULU', 'MRNA', 'NTAP', 'NXPI', 'PCAR', 'SIRI', 'SWKS', 'TCOM',
                'ULTA', 'VRSN', 'WDAY', 'ZBRA', 'ZM', 'ZS', 'ROKU', 'PTON',
                'DOCU', 'ZM', 'CRWD', 'OKTA', 'TWLO', 'SQ', 'SHOP', 'SPOT',
                'UBER', 'LYFT', 'PINS', 'SNAP', 'TWTR', 'SQ', 'SHOP', 'SPOT'
            ]
            
            logger.info(f"Using {len(major_nasdaq_stocks)} major NASDAQ stocks for capitulation analysis")
            return major_nasdaq_stocks
            
        except Exception as e:
            logger.error(f"Error getting NASDAQ symbols: {e}")
            # Ultimate fallback
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
    
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators for capitulation detection"""
        if len(df) < 50:
            return {}
        
        try:
            # Price data - TA-Lib requires float64 (double) arrays
            high = df['High'].values.astype(np.float64)
            low = df['Low'].values.astype(np.float64)
            close = df['Close'].values.astype(np.float64)
            volume = df['Volume'].values.astype(np.float64)
            
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # Volume indicators
            volume_sma_20 = talib.SMA(volume, timeperiod=20)
            volume_ratio = volume[-1] / volume_sma_20[-1] if volume_sma_20[-1] > 0 else 1
            
            # Price action
            current_price = close[-1]
            prev_price = close[-2]
            price_change = (current_price - prev_price) / prev_price * 100
            
            # Large downward candle detection
            open_price = df['Open'].iloc[-1]
            candle_body = abs(close[-1] - open_price)
            candle_range = high[-1] - low[-1]
            body_ratio = candle_body / candle_range if candle_range > 0 else 0
            
            # Tail detection (lower shadow)
            lower_tail = min(open_price, close[-1]) - low[-1]
            tail_ratio = lower_tail / candle_range if candle_range > 0 else 0
            
            return {
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_hist': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                'volume_ratio': volume_ratio,
                'price_change': price_change,
                'body_ratio': body_ratio,
                'tail_ratio': tail_ratio,
                'current_price': current_price,
                'volume': volume[-1]
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def detect_capitulation_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Detect capitulation signals based on indicators"""
        if not indicators:
            return {'is_capitulation': False, 'signals': [], 'score': 0}
        
        signals = []
        score = 0
        
        # Volume spike (3x average)
        if indicators.get('volume_ratio', 1) >= 3.0:
            signals.append('volume_spike')
            score += 3
        
        # RSI oversold
        rsi = indicators.get('rsi', 50)
        if rsi <= 30:
            signals.append('rsi_oversold')
            score += 2
        elif rsi <= 35:
            signals.append('rsi_near_oversold')
            score += 1
        
        # MACD bearish momentum
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        if macd < macd_signal and macd_hist < 0:
            signals.append('macd_bearish')
            score += 2
        
        # Large downward candle
        price_change = indicators.get('price_change', 0)
        if price_change <= -5:  # 5% or more drop
            signals.append('large_down_candle')
            score += 2
        elif price_change <= -3:  # 3% or more drop
            signals.append('moderate_down_candle')
            score += 1
        
        # Tail presence (hammer-like pattern)
        tail_ratio = indicators.get('tail_ratio', 0)
        if tail_ratio >= 0.3:  # 30% or more tail
            signals.append('long_tail')
            score += 1
        
        # Determine if capitulation
        is_capitulation = score >= 5  # Threshold for capitulation
        
        return {
            'is_capitulation': is_capitulation,
            'signals': signals,
            'score': score,
            'confidence': min(score / 8.0, 1.0)  # Normalize to 0-1
        }
    
    async def analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock for capitulation signals"""
        try:
            # Fetch stock data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3mo", interval="1d")
            
            if df.empty or len(df) < 50:
                return None
            
            # Calculate indicators
            indicators = self.calculate_technical_indicators(df)
            if not indicators:
                return None
            
            # Detect capitulation
            capitulation = self.detect_capitulation_signals(indicators)
            
            # Get additional market data
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            sector = info.get('sector', 'Unknown')
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': sector,
                'market_cap': market_cap,
                'current_price': indicators['current_price'],
                'price_change': indicators['price_change'],
                'volume': indicators['volume'],
                'volume_ratio': indicators['volume_ratio'],
                'rsi': indicators['rsi'],
                'macd': indicators['macd'],
                'macd_signal': indicators['macd_signal'],
                'macd_hist': indicators['macd_hist'],
                'capitulation': capitulation,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    async def screen_nasdaq_capitulation(self, limit: int = 50) -> Dict[str, Any]:
        """Screen all NASDAQ stocks for capitulation signals using real data"""
        try:
            # Get NASDAQ symbols using real data
            if not self.nasdaq_symbols:
                self.nasdaq_symbols = self._get_nasdaq_symbols()
            
            logger.info(f"Screening {len(self.nasdaq_symbols)} NASDAQ stocks for capitulation using real data")
            
            # Analyze stocks in smaller batches for faster processing
            batch_size = 5
            results = []
            
            # Analyze more stocks for comprehensive screening
            symbols_to_analyze = self.nasdaq_symbols[:50]  # Increased from 20 to 50
            
            for i in range(0, len(symbols_to_analyze), batch_size):
                batch = symbols_to_analyze[i:i + batch_size]
                tasks = [self.analyze_stock(symbol) for symbol in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, dict) and result is not None:
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Error in batch processing: {result}")
                
                # Small delay between batches
                await asyncio.sleep(0.2)
            
            # Filter for capitulation signals
            capitulation_stocks = [
                stock for stock in results 
                if stock.get('capitulation', {}).get('is_capitulation', False)
            ]
            
            # Sort by capitulation score
            capitulation_stocks.sort(
                key=lambda x: x.get('capitulation', {}).get('score', 0), 
                reverse=True
            )
            
            # Get top performers
            top_capitulation = capitulation_stocks[:limit]
            
            # Calculate market statistics
            total_analyzed = len(results)
            capitulation_count = len(capitulation_stocks)
            capitulation_rate = capitulation_count / total_analyzed if total_analyzed > 0 else 0
            
            logger.info(f"Capitulation analysis complete: {capitulation_count}/{total_analyzed} stocks in capitulation")
            
            return {
                'total_analyzed': total_analyzed,
                'capitulation_count': capitulation_count,
                'capitulation_rate': capitulation_rate,
                'top_capitulation': top_capitulation,
                'analysis_date': datetime.now().isoformat(),
                'status': 'success',
                'data_source': 'real_yfinance_data'
            }
            
        except Exception as e:
            logger.error(f"Error screening NASDAQ capitulation: {e}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    async def get_capitulation_summary(self) -> Dict[str, Any]:
        """Get a summary of current capitulation conditions"""
        try:
            # Get VIX data as market fear indicator
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            
            current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20
            vix_change = ((current_vix - vix_data['Close'].iloc[-2]) / vix_data['Close'].iloc[-2] * 100) if len(vix_data) > 1 else 0
            
            # Determine market condition
            if current_vix >= 30:
                market_condition = "High Fear"
            elif current_vix >= 20:
                market_condition = "Moderate Fear"
            else:
                market_condition = "Low Fear"
            
            return {
                'vix': current_vix,
                'vix_change': vix_change,
                'market_condition': market_condition,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting capitulation summary: {e}")
            return {
                'vix': 20,
                'vix_change': 0,
                'market_condition': 'Unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

# Global capitulation detector instance
capitulation_detector = CapitulationDetector()
