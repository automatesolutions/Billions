"""
Enhanced Capitulation Detection System
More comprehensive and sensitive detection across all NASDAQ stocks
"""

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

logger = logging.getLogger(__name__)

class EnhancedCapitulationDetector:
    """Enhanced capitulation detection with more sensitive algorithms"""
    
    def __init__(self):
        self.session = None
        self.nasdaq_symbols = []
        
    async def get_session(self):
        """Get or create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    def _get_all_nasdaq_symbols(self) -> List[str]:
        """Get comprehensive list of NASDAQ symbols"""
        try:
            # Extended list of NASDAQ stocks for comprehensive screening
            nasdaq_stocks = [
                # Mega Cap Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'ADBE', 'CRM', 'INTC', 'AMD', 'PYPL', 'CMCSA', 'PEP', 'COST',
                'AVGO', 'TXN', 'QCOM', 'CHTR', 'AMAT', 'ISRG', 'GILD', 'BKNG',
                
                # Large Cap Growth
                'ADP', 'VRTX', 'FISV', 'BIIB', 'REGN', 'MDLZ', 'ATVI', 'CSX',
                'ILMN', 'AMGN', 'WBA', 'CTAS', 'EXC', 'EA', 'LRCX', 'KLAC',
                'SNPS', 'MCHP', 'ADI', 'CDNS', 'ORLY', 'IDXX', 'DXCM', 'ALGN',
                
                # Mid Cap Growth
                'CTSH', 'FAST', 'PAYX', 'ROST', 'SBUX', 'TMUS', 'VRSK', 'WLTW',
                'XEL', 'ANSS', 'BMRN', 'CERN', 'CHKP', 'CTXS', 'DLTR', 'EBAY',
                'EXPE', 'FIS', 'FTNT', 'GPN', 'HSIC', 'INTU', 'JBHT', 'KDP',
                'LULU', 'MRNA', 'NTAP', 'NXPI', 'PCAR', 'SIRI', 'SWKS', 'TCOM',
                
                # Small Cap & Emerging
                'ULTA', 'VRSN', 'WDAY', 'ZBRA', 'ZM', 'ZS', 'ROKU', 'PTON',
                'DOCU', 'CRWD', 'OKTA', 'TWLO', 'SQ', 'SHOP', 'SPOT', 'UBER',
                'LYFT', 'PINS', 'SNAP', 'TWTR', 'SNOW', 'PLTR', 'RBLX', 'COIN',
                
                # Biotech & Healthcare
                'GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'DXCM', 'ALGN',
                'IDXX', 'BMRN', 'MRNA', 'CERN', 'HSIC', 'INTU', 'JBHT', 'KDP',
                
                # Semiconductor & Hardware
                'NVDA', 'AMD', 'INTC', 'AVGO', 'TXN', 'QCOM', 'AMAT', 'LRCX',
                'KLAC', 'SNPS', 'MCHP', 'ADI', 'CDNS', 'NXPI', 'SWKS', 'TCOM',
                
                # Software & Cloud
                'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'ADBE', 'CRM', 'PYPL',
                'ORLY', 'CTSH', 'FAST', 'PAYX', 'VRSK', 'WLTW', 'ANSS', 'CHKP',
                'CTXS', 'FIS', 'FTNT', 'INTU', 'NTAP', 'VRSN', 'WDAY', 'ZM',
                'ZS', 'DOCU', 'CRWD', 'OKTA', 'TWLO', 'SQ', 'SHOP', 'SPOT',
                
                # Consumer & Retail
                'AMZN', 'COST', 'PEP', 'BKNG', 'ROST', 'SBUX', 'DLTR', 'LULU',
                'ULTA', 'ROKU', 'PTON', 'PINS', 'SNAP', 'TWTR', 'RBLX',
                
                # Financial & Payment
                'PYPL', 'SQ', 'COIN', 'FIS', 'FISV', 'GPN', 'ADP', 'PAYX',
                
                # Transportation & Logistics
                'TSLA', 'UBER', 'LYFT', 'CSX', 'JBHT', 'PCAR', 'EXPE', 'BKNG',
                
                # Energy & Utilities
                'EXC', 'XEL', 'PEP', 'KDP', 'COST', 'WBA', 'CTAS', 'FAST',
                
                # Additional Volatile Stocks (often show capitulation)
                'PLTR', 'SNOW', 'RBLX', 'COIN', 'ROKU', 'PTON', 'DOCU', 'ZM',
                'ZS', 'CRWD', 'OKTA', 'TWLO', 'SQ', 'SHOP', 'SPOT', 'UBER',
                'LYFT', 'PINS', 'SNAP', 'TWTR', 'SNOW', 'PLTR', 'RBLX', 'COIN',
                
                # Penny Stocks & High Volatility
                'NVAX', 'MRNA', 'BNTX', 'PFE', 'JNJ', 'ABBV', 'MRK', 'LLY',
                'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN',
                
                # Crypto & Blockchain
                'COIN', 'SQ', 'PYPL', 'NVDA', 'AMD', 'INTC', 'AVGO', 'TXN',
                
                # EV & Clean Energy
                'TSLA', 'NIO', 'XPEV', 'LI', 'LCID', 'RIVN', 'FSR', 'WKHS',
                'NKLA', 'HYLN', 'GOEV', 'RIDE', 'SOLO', 'AYRO', 'KNDI', 'WKHS',
                
                # Meme Stocks (high volatility)
                'GME', 'AMC', 'BB', 'NOK', 'EXPR', 'KOSS', 'NAKD', 'SNDL',
                'CLOV', 'WISH', 'SPCE', 'PLTR', 'RBLX', 'COIN', 'HOOD', 'SOFI',
                
                # SPACs & Recent IPOs
                'SPCE', 'PLTR', 'SNOW', 'RBLX', 'COIN', 'HOOD', 'SOFI', 'RIVN',
                'LCID', 'FSR', 'WKHS', 'NKLA', 'HYLN', 'GOEV', 'RIDE', 'SOLO',
                
                # Additional High Volume Stocks
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'AMD', 'INTC', 'AVGO', 'TXN', 'QCOM', 'AMAT', 'LRCX', 'KLAC',
                'SNPS', 'MCHP', 'ADI', 'CDNS', 'ORLY', 'IDXX', 'DXCM', 'ALGN',
                'CTSH', 'FAST', 'PAYX', 'ROST', 'SBUX', 'TMUS', 'VRSK', 'WLTW',
                'XEL', 'ANSS', 'BMRN', 'CERN', 'CHKP', 'CTXS', 'DLTR', 'EBAY',
                'EXPE', 'FIS', 'FTNT', 'GPN', 'HSIC', 'INTU', 'JBHT', 'KDP',
                'LULU', 'MRNA', 'NTAP', 'NXPI', 'PCAR', 'SIRI', 'SWKS', 'TCOM',
                'ULTA', 'VRSN', 'WDAY', 'ZBRA', 'ZM', 'ZS', 'ROKU', 'PTON',
                'DOCU', 'CRWD', 'OKTA', 'TWLO', 'SQ', 'SHOP', 'SPOT', 'UBER',
                'LYFT', 'PINS', 'SNAP', 'TWTR', 'SNOW', 'PLTR', 'RBLX', 'COIN'
            ]
            
            # Remove duplicates and sort
            unique_stocks = list(set(nasdaq_stocks))
            unique_stocks.sort()
            
            logger.info(f"Using {len(unique_stocks)} NASDAQ stocks for enhanced capitulation analysis")
            return unique_stocks
            
        except Exception as e:
            logger.error(f"Error getting NASDAQ symbols: {e}")
            # Fallback to major stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive technical indicators for capitulation detection"""
        if len(df) < 50:
            return {}
        
        try:
            # Price data
            high = df['High'].values.astype(np.float64)
            low = df['Low'].values.astype(np.float64)
            close = df['Close'].values.astype(np.float64)
            volume = df['Volume'].values.astype(np.float64)
            open_price = df['Open'].values.astype(np.float64)
            
            # Basic Technical Indicators
            rsi = talib.RSI(close, timeperiod=14)
            macd, macd_signal, macd_hist = talib.MACD(close)
            
            # Volume Analysis
            volume_sma_20 = talib.SMA(volume, timeperiod=20)
            volume_sma_50 = talib.SMA(volume, timeperiod=50)
            volume_ratio_20 = volume[-1] / volume_sma_20[-1] if volume_sma_20[-1] > 0 else 1
            volume_ratio_50 = volume[-1] / volume_sma_50[-1] if volume_sma_50[-1] > 0 else 1
            
            # Price Action Analysis
            current_price = close[-1]
            prev_price = close[-2]
            price_change = (current_price - prev_price) / prev_price * 100
            
            # Multi-day price changes
            price_change_3d = (current_price - close[-4]) / close[-4] * 100 if len(close) >= 4 else 0
            price_change_5d = (current_price - close[-6]) / close[-6] * 100 if len(close) >= 6 else 0
            price_change_10d = (current_price - close[-11]) / close[-11] * 100 if len(close) >= 11 else 0
            
            # Volatility Analysis
            atr = talib.ATR(high, low, close, timeperiod=14)
            volatility = atr[-1] / current_price * 100 if current_price > 0 else 0
            
            # Momentum Indicators
            stoch_k, stoch_d = talib.STOCH(high, low, close)
            williams_r = talib.WILLR(high, low, close, timeperiod=14)
            
            # Trend Analysis
            sma_20 = talib.SMA(close, timeperiod=20)
            sma_50 = talib.SMA(close, timeperiod=50)
            sma_200 = talib.SMA(close, timeperiod=200)
            
            # Support/Resistance Levels
            current_close = close[-1]
            sma_20_current = sma_20[-1]
            sma_50_current = sma_50[-1]
            sma_200_current = sma_200[-1]
            
            # Distance from moving averages
            distance_sma20 = (current_close - sma_20_current) / sma_20_current * 100
            distance_sma50 = (current_close - sma_50_current) / sma_50_current * 100
            distance_sma200 = (current_close - sma_200_current) / sma_200_current * 100
            
            # Candle Analysis
            open_price_current = open_price[-1]
            candle_body = abs(close[-1] - open_price_current)
            candle_range = high[-1] - low[-1]
            body_ratio = candle_body / candle_range if candle_range > 0 else 0
            
            # Tail Analysis
            upper_tail = high[-1] - max(open_price_current, close[-1])
            lower_tail = min(open_price_current, close[-1]) - low[-1]
            upper_tail_ratio = upper_tail / candle_range if candle_range > 0 else 0
            lower_tail_ratio = lower_tail / candle_range if candle_range > 0 else 0
            
            # Gap Analysis
            gap_up = open_price_current - close[-2] if open_price_current > close[-2] else 0
            gap_down = close[-2] - open_price_current if close[-2] > open_price_current else 0
            
            # Market Structure
            higher_highs = sum(1 for i in range(1, min(10, len(high))) if high[-i] > high[-i-1])
            lower_lows = sum(1 for i in range(1, min(10, len(low))) if low[-i] < low[-i-1])
            
            return {
                # Basic Indicators
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'macd': macd[-1] if not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0,
                'macd_hist': macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0,
                
                # Volume Analysis
                'volume_ratio_20': volume_ratio_20,
                'volume_ratio_50': volume_ratio_50,
                'volume': volume[-1],
                
                # Price Action
                'price_change': price_change,
                'price_change_3d': price_change_3d,
                'price_change_5d': price_change_5d,
                'price_change_10d': price_change_10d,
                'current_price': current_price,
                
                # Volatility
                'volatility': volatility,
                'atr': atr[-1] if not np.isnan(atr[-1]) else 0,
                
                # Momentum
                'stoch_k': stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50,
                'stoch_d': stoch_d[-1] if not np.isnan(stoch_d[-1]) else 50,
                'williams_r': williams_r[-1] if not np.isnan(williams_r[-1]) else -50,
                
                # Trend Analysis
                'sma_20': sma_20_current,
                'sma_50': sma_50_current,
                'sma_200': sma_200_current,
                'distance_sma20': distance_sma20,
                'distance_sma50': distance_sma50,
                'distance_sma200': distance_sma200,
                
                # Candle Analysis
                'body_ratio': body_ratio,
                'upper_tail_ratio': upper_tail_ratio,
                'lower_tail_ratio': lower_tail_ratio,
                'candle_range': candle_range,
                
                # Gap Analysis
                'gap_up': gap_up,
                'gap_down': gap_down,
                
                # Market Structure
                'higher_highs': higher_highs,
                'lower_lows': lower_lows
            }
            
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")
            return {}
    
    def detect_enhanced_capitulation_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced capitulation detection with more sensitive algorithms"""
        if not indicators:
            return {'is_capitulation': False, 'signals': [], 'score': 0, 'confidence': 0}
        
        signals = []
        score = 0
        confidence_factors = []
        
        # === VOLUME ANALYSIS (More Sensitive) ===
        volume_ratio_20 = indicators.get('volume_ratio_20', 1)
        volume_ratio_50 = indicators.get('volume_ratio_50', 1)
        
        # Volume spike detection (lowered thresholds)
        if volume_ratio_20 >= 2.5:  # Lowered from 3.0
            signals.append('volume_spike_20')
            score += 3
            confidence_factors.append(volume_ratio_20)
        elif volume_ratio_20 >= 1.8:  # New threshold
            signals.append('volume_elevated_20')
            score += 2
            confidence_factors.append(volume_ratio_20)
        
        if volume_ratio_50 >= 2.0:  # New threshold
            signals.append('volume_spike_50')
            score += 2
            confidence_factors.append(volume_ratio_50)
        
        # === RSI ANALYSIS (More Sensitive) ===
        rsi = indicators.get('rsi', 50)
        if rsi <= 25:  # More extreme oversold
            signals.append('rsi_extreme_oversold')
            score += 4
            confidence_factors.append(30 - rsi)
        elif rsi <= 30:
            signals.append('rsi_oversold')
            score += 3
            confidence_factors.append(30 - rsi)
        elif rsi <= 35:
            signals.append('rsi_near_oversold')
            score += 2
            confidence_factors.append(35 - rsi)
        elif rsi <= 40:  # New threshold
            signals.append('rsi_weak')
            score += 1
            confidence_factors.append(40 - rsi)
        
        # === MOMENTUM ANALYSIS ===
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        # MACD bearish momentum
        if macd < macd_signal and macd_hist < 0:
            signals.append('macd_bearish')
            score += 2
            confidence_factors.append(abs(macd_hist))
        
        # Stochastic oversold
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k <= 20 and stoch_d <= 20:
            signals.append('stoch_oversold')
            score += 2
            confidence_factors.append(20 - stoch_k)
        
        # Williams %R oversold
        williams_r = indicators.get('williams_r', -50)
        if williams_r <= -80:
            signals.append('williams_oversold')
            score += 2
            confidence_factors.append(abs(williams_r + 80))
        
        # === PRICE ACTION ANALYSIS (More Sensitive) ===
        price_change = indicators.get('price_change', 0)
        price_change_3d = indicators.get('price_change_3d', 0)
        price_change_5d = indicators.get('price_change_5d', 0)
        price_change_10d = indicators.get('price_change_10d', 0)
        
        # Single day drops
        if price_change <= -8:  # More extreme
            signals.append('extreme_down_day')
            score += 4
            confidence_factors.append(abs(price_change))
        elif price_change <= -5:
            signals.append('large_down_day')
            score += 3
            confidence_factors.append(abs(price_change))
        elif price_change <= -3:  # Lowered threshold
            signals.append('moderate_down_day')
            score += 2
            confidence_factors.append(abs(price_change))
        elif price_change <= -1.5:  # New threshold
            signals.append('small_down_day')
            score += 1
            confidence_factors.append(abs(price_change))
        
        # Multi-day declines
        if price_change_3d <= -15:
            signals.append('extreme_3d_decline')
            score += 4
            confidence_factors.append(abs(price_change_3d))
        elif price_change_3d <= -10:
            signals.append('large_3d_decline')
            score += 3
            confidence_factors.append(abs(price_change_3d))
        elif price_change_3d <= -5:
            signals.append('moderate_3d_decline')
            score += 2
            confidence_factors.append(abs(price_change_3d))
        
        if price_change_5d <= -20:
            signals.append('extreme_5d_decline')
            score += 4
            confidence_factors.append(abs(price_change_5d))
        elif price_change_5d <= -12:
            signals.append('large_5d_decline')
            score += 3
            confidence_factors.append(abs(price_change_5d))
        elif price_change_5d <= -7:
            signals.append('moderate_5d_decline')
            score += 2
            confidence_factors.append(abs(price_change_5d))
        
        # === TREND ANALYSIS ===
        distance_sma20 = indicators.get('distance_sma20', 0)
        distance_sma50 = indicators.get('distance_sma50', 0)
        distance_sma200 = indicators.get('distance_sma200', 0)
        
        # Below moving averages
        if distance_sma20 <= -10:
            signals.append('far_below_sma20')
            score += 3
            confidence_factors.append(abs(distance_sma20))
        elif distance_sma20 <= -5:
            signals.append('below_sma20')
            score += 2
            confidence_factors.append(abs(distance_sma20))
        elif distance_sma20 <= -2:
            signals.append('near_sma20')
            score += 1
            confidence_factors.append(abs(distance_sma20))
        
        if distance_sma50 <= -15:
            signals.append('far_below_sma50')
            score += 3
            confidence_factors.append(abs(distance_sma50))
        elif distance_sma50 <= -8:
            signals.append('below_sma50')
            score += 2
            confidence_factors.append(abs(distance_sma50))
        
        if distance_sma200 <= -20:
            signals.append('far_below_sma200')
            score += 4
            confidence_factors.append(abs(distance_sma200))
        elif distance_sma200 <= -10:
            signals.append('below_sma200')
            score += 3
            confidence_factors.append(abs(distance_sma200))
        
        # === VOLATILITY ANALYSIS ===
        volatility = indicators.get('volatility', 0)
        if volatility >= 8:  # High volatility
            signals.append('high_volatility')
            score += 2
            confidence_factors.append(volatility)
        elif volatility >= 5:
            signals.append('elevated_volatility')
            score += 1
            confidence_factors.append(volatility)
        
        # === CANDLE PATTERN ANALYSIS ===
        lower_tail_ratio = indicators.get('lower_tail_ratio', 0)
        upper_tail_ratio = indicators.get('upper_tail_ratio', 0)
        body_ratio = indicators.get('body_ratio', 0)
        
        # Hammer-like patterns
        if lower_tail_ratio >= 0.4 and body_ratio <= 0.3:
            signals.append('hammer_pattern')
            score += 2
            confidence_factors.append(lower_tail_ratio)
        elif lower_tail_ratio >= 0.3:
            signals.append('long_lower_tail')
            score += 1
            confidence_factors.append(lower_tail_ratio)
        
        # Doji patterns (indecision)
        if body_ratio <= 0.1 and (lower_tail_ratio >= 0.3 or upper_tail_ratio >= 0.3):
            signals.append('doji_pattern')
            score += 1
            confidence_factors.append(1 - body_ratio)
        
        # === GAP ANALYSIS ===
        gap_down = indicators.get('gap_down', 0)
        if gap_down >= 0.05:  # 5% gap down
            signals.append('gap_down')
            score += 2
            confidence_factors.append(gap_down * 100)
        
        # === MARKET STRUCTURE ===
        lower_lows = indicators.get('lower_lows', 0)
        if lower_lows >= 3:
            signals.append('lower_lows_pattern')
            score += 2
            confidence_factors.append(lower_lows)
        
        # === ENHANCED CAPITULATION DETECTION ===
        # Lowered threshold for more sensitive detection
        is_capitulation = score >= 3  # Lowered from 5
        
        # Calculate confidence based on multiple factors
        if confidence_factors:
            avg_confidence = sum(confidence_factors) / len(confidence_factors)
            normalized_confidence = min(avg_confidence / 10.0, 1.0)  # Normalize to 0-1
        else:
            normalized_confidence = 0
        
        # Additional confidence boost for multiple signal types
        signal_types = len(set(signal.split('_')[0] for signal in signals))
        if signal_types >= 3:
            normalized_confidence = min(normalized_confidence + 0.2, 1.0)
        
        return {
            'is_capitulation': is_capitulation,
            'signals': signals,
            'score': score,
            'confidence': normalized_confidence,
            'signal_count': len(signals),
            'signal_types': signal_types
        }
    
    async def analyze_stock_enhanced(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Enhanced analysis of a single stock for capitulation signals"""
        try:
            # Fetch stock data with multiple timeframes
            ticker = yf.Ticker(symbol)
            
            # Get daily data for main analysis
            df_daily = ticker.history(period="6mo", interval="1d")
            if df_daily.empty or len(df_daily) < 50:
                return None
            
            # Calculate enhanced indicators
            indicators = self.calculate_enhanced_indicators(df_daily)
            if not indicators:
                return None
            
            # Detect capitulation signals
            capitulation = self.detect_enhanced_capitulation_signals(indicators)
            
            # Get additional market data
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            avg_volume = info.get('averageVolume', 0)
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            
            # Calculate additional metrics
            current_price = indicators.get('current_price', 0)
            volume = indicators.get('volume', 0)
            
            # Risk assessment
            risk_level = 'Low'
            if capitulation['score'] >= 8:
                risk_level = 'Extreme'
            elif capitulation['score'] >= 6:
                risk_level = 'High'
            elif capitulation['score'] >= 4:
                risk_level = 'Moderate'
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'market_cap': market_cap,
                'avg_volume': avg_volume,
                'current_volume': volume,
                'sector': sector,
                'industry': industry,
                'is_capitulation': capitulation['is_capitulation'],
                'capitulation_score': capitulation['score'],
                'confidence': capitulation['confidence'],
                'signals': capitulation['signals'],
                'signal_count': capitulation['signal_count'],
                'signal_types': capitulation['signal_types'],
                'risk_level': risk_level,
                'indicators': {
                    'rsi': indicators.get('rsi', 50),
                    'volume_ratio_20': indicators.get('volume_ratio_20', 1),
                    'price_change': indicators.get('price_change', 0),
                    'price_change_3d': indicators.get('price_change_3d', 0),
                    'price_change_5d': indicators.get('price_change_5d', 0),
                    'distance_sma20': indicators.get('distance_sma20', 0),
                    'distance_sma50': indicators.get('distance_sma50', 0),
                    'volatility': indicators.get('volatility', 0)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return None
    
    async def screen_nasdaq_enhanced(self, limit: int = 100) -> Dict[str, Any]:
        """Enhanced screening of NASDAQ stocks for capitulation signals"""
        try:
            logger.info(f"Starting enhanced NASDAQ capitulation screening (limit: {limit})")
            
            # Get comprehensive list of NASDAQ symbols
            symbols = self._get_all_nasdaq_symbols()
            
            # Limit the number of stocks to analyze
            symbols_to_analyze = symbols[:limit] if limit else symbols
            
            logger.info(f"Analyzing {len(symbols_to_analyze)} stocks for capitulation signals")
            
            # Analyze stocks concurrently
            tasks = [self.analyze_stock_enhanced(symbol) for symbol in symbols_to_analyze]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            capitulation_stocks = []
            total_analyzed = 0
            errors = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {symbols_to_analyze[i]}: {result}")
                    errors += 1
                    continue
                
                if result is None:
                    continue
                
                total_analyzed += 1
                
                if result['is_capitulation']:
                    capitulation_stocks.append(result)
            
            # Sort by capitulation score (highest first)
            capitulation_stocks.sort(key=lambda x: x['capitulation_score'], reverse=True)
            
            # Calculate statistics
            total_capitulation = len(capitulation_stocks)
            capitulation_rate = (total_capitulation / total_analyzed * 100) if total_analyzed > 0 else 0
            
            # Get market summary
            market_summary = await self.get_market_summary_enhanced()
            
            logger.info(f"Enhanced screening completed: {total_capitulation}/{total_analyzed} stocks showing capitulation ({capitulation_rate:.1f}%)")
            
            return {
                'total_stocks_analyzed': total_analyzed,
                'capitulation_stocks': capitulation_stocks,
                'capitulation_count': total_capitulation,
                'capitulation_rate': round(capitulation_rate, 2),
                'errors': errors,
                'market_summary': market_summary,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'Enhanced Capitulation Detection'
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced NASDAQ screening: {e}")
            return {
                'total_stocks_analyzed': 0,
                'capitulation_stocks': [],
                'capitulation_count': 0,
                'capitulation_rate': 0,
                'errors': 1,
                'market_summary': {},
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    async def get_market_summary_enhanced(self) -> Dict[str, Any]:
        """Enhanced market summary with more indicators"""
        try:
            # Get VIX data
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="5d")
            
            current_vix = vix_data['Close'].iloc[-1] if not vix_data.empty else 20
            vix_change = ((current_vix - vix_data['Close'].iloc[-2]) / vix_data['Close'].iloc[-2] * 100) if len(vix_data) > 1 else 0
            
            # Get SPY data for market context
            spy_ticker = yf.Ticker("SPY")
            spy_data = spy_ticker.history(period="5d")
            
            spy_change = 0
            if not spy_data.empty and len(spy_data) > 1:
                spy_change = ((spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[-2]) / spy_data['Close'].iloc[-2] * 100)
            
            # Get QQQ data (NASDAQ proxy)
            qqq_ticker = yf.Ticker("QQQ")
            qqq_data = qqq_ticker.history(period="5d")
            
            qqq_change = 0
            if not qqq_data.empty and len(qqq_data) > 1:
                qqq_change = ((qqq_data['Close'].iloc[-1] - qqq_data['Close'].iloc[-2]) / qqq_data['Close'].iloc[-2] * 100)
            
            # Determine market condition
            if current_vix >= 35:
                market_condition = "Extreme Fear"
            elif current_vix >= 25:
                market_condition = "High Fear"
            elif current_vix >= 20:
                market_condition = "Moderate Fear"
            elif current_vix >= 15:
                market_condition = "Low Fear"
            else:
                market_condition = "Complacency"
            
            # Determine market trend
            if spy_change >= 2:
                market_trend = "Strong Bullish"
            elif spy_change >= 0.5:
                market_trend = "Bullish"
            elif spy_change >= -0.5:
                market_trend = "Neutral"
            elif spy_change >= -2:
                market_trend = "Bearish"
            else:
                market_trend = "Strong Bearish"
            
            return {
                'vix': round(current_vix, 2),
                'vix_change': round(vix_change, 2),
                'spy_change': round(spy_change, 2),
                'qqq_change': round(qqq_change, 2),
                'market_condition': market_condition,
                'market_trend': market_trend,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced market summary: {e}")
            return {
                'vix': 20,
                'vix_change': 0,
                'spy_change': 0,
                'qqq_change': 0,
                'market_condition': 'Unknown',
                'market_trend': 'Unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

# Global enhanced capitulation detector instance
enhanced_capitulation_detector = EnhancedCapitulationDetector()