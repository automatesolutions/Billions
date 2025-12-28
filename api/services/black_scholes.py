"""
Black-Scholes-Merton Option Pricing Model
Used to calculate fair value and determine if stock is undervalued/overvalued
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import yfinance as yf

logger = logging.getLogger(__name__)


class BlackScholesMerton:
    """Black-Scholes-Merton option pricing model for fair value calculation"""
    
    def __init__(self):
        self.risk_free_rate = 0.045  # 4.5% risk-free rate (10-year Treasury)
        
    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate from Treasury data"""
        try:
            # Try to get 10-year Treasury rate
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
                self.risk_free_rate = rate
                logger.info(f"Updated risk-free rate to {rate:.3f}")
        except Exception as e:
            logger.warning(f"Could not fetch risk-free rate: {e}, using default {self.risk_free_rate}")
        
        return self.risk_free_rate
    
    def calculate_historical_volatility(self, prices: pd.Series, days: int = 252) -> float:
        """Calculate historical volatility from price data"""
        try:
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            # Annualized volatility (252 trading days per year)
            volatility = returns.std() * np.sqrt(252)
            
            logger.info(f"Calculated historical volatility: {volatility:.3f}")
            return volatility
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.25  # Default 25% volatility
    
    def black_scholes_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes call option price"""
        try:
            if T <= 0:
                return max(S - K, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return call_price
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return S  # Return stock price as fallback
    
    def black_scholes_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Black-Scholes put option price"""
        try:
            if T <= 0:
                return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return put_price
        except Exception as e:
            logger.error(f"Error in Black-Scholes put calculation: {e}")
            return max(K - S, 0)
    
    def calculate_fair_value(self, current_price: float, strike_prices: list, 
                           time_to_expiry: float, volatility: float, 
                           risk_free_rate: float = None) -> Dict:
        """
        Calculate fair value using Black-Scholes-Merton model
        
        Args:
            current_price: Current stock price
            strike_prices: List of strike prices to evaluate
            time_to_expiry: Time to expiry in years
            volatility: Stock volatility
            risk_free_rate: Risk-free rate (optional)
            
        Returns:
            Dictionary with fair value analysis
        """
        try:
            if risk_free_rate is None:
                risk_free_rate = self.get_risk_free_rate()
            
            logger.info(f"Calculating fair value for price ${current_price:.2f}, volatility {volatility:.3f}")
            
            # Calculate call and put prices for different strikes
            call_prices = []
            put_prices = []
            
            for K in strike_prices:
                call_price = self.black_scholes_call(current_price, K, time_to_expiry, risk_free_rate, volatility)
                put_price = self.black_scholes_put(current_price, K, time_to_expiry, risk_free_rate, volatility)
                
                call_prices.append(call_price)
                put_prices.append(put_price)
            
            # Calculate implied volatility if possible (using current price as market price)
            implied_vol = self.calculate_implied_volatility(current_price, current_price, time_to_expiry, risk_free_rate, current_price)
            
            # Calculate fair value as weighted average of call and put at ATM
            atm_call = self.black_scholes_call(current_price, current_price, time_to_expiry, risk_free_rate, volatility)
            atm_put = self.black_scholes_put(current_price, current_price, time_to_expiry, risk_free_rate, volatility)
            
            # Fair value is the intrinsic value adjusted for time value
            fair_value = current_price * (1 + (risk_free_rate - volatility**2/2) * time_to_expiry)
            
            # Alternative fair value calculation using put-call parity
            put_call_fair_value = atm_call - atm_put + current_price * np.exp(-risk_free_rate * time_to_expiry)
            
            # Use the average of both methods
            final_fair_value = (fair_value + put_call_fair_value) / 2
            
            # Determine if undervalued or overvalued
            valuation_ratio = current_price / final_fair_value
            if valuation_ratio < 0.95:
                valuation_status = "Undervalued"
                valuation_color = "green"
            elif valuation_ratio > 1.05:
                valuation_status = "Overvalued"
                valuation_color = "red"
            else:
                valuation_status = "Fairly Valued"
                valuation_color = "yellow"
            
            result = {
                "current_price": current_price,
                "fair_value": final_fair_value,
                "valuation_ratio": valuation_ratio,
                "valuation_status": valuation_status,
                "valuation_color": valuation_color,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "time_to_expiry": time_to_expiry,
                "implied_volatility": implied_vol,
                "atm_call_price": atm_call,
                "atm_put_price": atm_put,
                "strike_prices": strike_prices,
                "call_prices": call_prices,
                "put_prices": put_prices
            }
            
            logger.info(f"Fair value calculated: ${final_fair_value:.2f}, Status: {valuation_status}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fair value: {e}")
            return {
                "current_price": current_price,
                "fair_value": current_price,
                "valuation_ratio": 1.0,
                "valuation_status": "Unable to Calculate",
                "valuation_color": "gray",
                "error": str(e)
            }
    
    def calculate_implied_volatility(self, S: float, K: float, T: float, r: float, 
                                   market_price: float) -> float:
        """Calculate implied volatility from market price"""
        try:
            def objective(sigma):
                theoretical_price = self.black_scholes_call(S, K, T, r, sigma)
                return (theoretical_price - market_price) ** 2
            
            result = minimize_scalar(objective, bounds=(0.01, 2.0), method='bounded')
            return result.x
        except Exception as e:
            logger.warning(f"Could not calculate implied volatility: {e}")
            return 0.25  # Default volatility
    
    def analyze_stock_valuation(self, ticker: str, days_back: int = 252) -> Dict:
        """
        Complete stock valuation analysis using Black-Scholes-Merton
        
        Args:
            ticker: Stock symbol
            days_back: Number of days to look back for volatility calculation
            
        Returns:
            Dictionary with complete valuation analysis
        """
        try:
            logger.info(f"Starting BSM valuation analysis for {ticker}")
            
            # Fetch stock data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days_back}d")
            
            if hist.empty:
                raise ValueError(f"No historical data available for {ticker}")
            
            current_price = hist['Close'].iloc[-1]
            current_date = hist.index[-1]
            
            # Calculate volatility
            volatility = self.calculate_historical_volatility(hist['Close'])
            
            # Set time to expiry (1 year)
            time_to_expiry = 1.0
            
            # Generate strike prices around current price
            strike_range = 0.2  # 20% range
            strikes = [
                current_price * (1 - strike_range),
                current_price * (1 - strike_range/2),
                current_price,
                current_price * (1 + strike_range/2),
                current_price * (1 + strike_range)
            ]
            
            # Calculate fair value
            valuation = self.calculate_fair_value(
                current_price=current_price,
                strike_prices=strikes,
                time_to_expiry=time_to_expiry,
                volatility=volatility
            )
            
            # Add additional analysis
            valuation.update({
                "ticker": ticker,
                "analysis_date": current_date.isoformat(),
                "days_analyzed": len(hist),
                "price_change_1d": hist['Close'].pct_change().iloc[-1] * 100,
                "price_change_30d": ((current_price / hist['Close'].iloc[-30]) - 1) * 100 if len(hist) >= 30 else 0,
                "price_change_1y": ((current_price / hist['Close'].iloc[0]) - 1) * 100,
                "volatility_30d": self.calculate_historical_volatility(hist['Close'].tail(30)) if len(hist) >= 30 else volatility,
                "beta": self.estimate_beta(hist),
                "sharpe_ratio": self.calculate_sharpe_ratio(hist, self.risk_free_rate)
            })
            
            return valuation
            
        except Exception as e:
            logger.error(f"Error in BSM analysis for {ticker}: {e}")
            return {
                "ticker": ticker,
                "current_price": 0,
                "fair_value": 0,
                "valuation_status": "Analysis Failed",
                "valuation_color": "gray",
                "error": str(e)
            }
    
    def estimate_beta(self, hist: pd.DataFrame) -> float:
        """Estimate beta using market correlation"""
        try:
            # Get S&P 500 data for beta calculation
            sp500 = yf.Ticker("^GSPC")
            sp500_hist = sp500.history(start=hist.index[0], end=hist.index[-1])
            
            if len(sp500_hist) < 30:
                return 1.0  # Default beta
            
            # Calculate returns
            stock_returns = hist['Close'].pct_change().dropna()
            market_returns = sp500_hist['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns[common_dates]
            market_returns = market_returns[common_dates]
            
            if len(stock_returns) < 10:
                return 1.0
            
            # Calculate beta
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            return beta
        except Exception as e:
            logger.warning(f"Could not calculate beta: {e}")
            return 1.0
    
    def calculate_sharpe_ratio(self, hist: pd.DataFrame, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        try:
            returns = hist['Close'].pct_change().dropna()
            excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            sharpe = excess_returns / volatility if volatility > 0 else 0
            return sharpe
        except Exception as e:
            logger.warning(f"Could not calculate Sharpe ratio: {e}")
            return 0.0


# Global instance
bsm_analyzer = BlackScholesMerton()


def test_black_scholes():
    """Test the Black-Scholes implementation"""
    analyzer = BlackScholesMerton()
    
    # Test with sample data
    result = analyzer.calculate_fair_value(
        current_price=100,
        strike_prices=[90, 95, 100, 105, 110],
        time_to_expiry=1.0,
        volatility=0.25
    )
    
    print("Black-Scholes Test Results:")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Fair Value: ${result['fair_value']:.2f}")
    print(f"Valuation Status: {result['valuation_status']}")
    print(f"Valuation Ratio: {result['valuation_ratio']:.3f}")


if __name__ == "__main__":
    test_black_scholes()
