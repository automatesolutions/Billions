from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

from api.database import get_db
from api.services.black_scholes import bsm_analyzer
from api.services.markov_predictor import MarkovChainPredictor
import yfinance as yf

logger = logging.getLogger(__name__)

router = APIRouter()

class PortfolioOptimizer:
    def __init__(self):
        self.markov_predictor = MarkovChainPredictor(num_states=20)
    
    def analyze_volatility(self, ticker: str, days: int = 252) -> dict:
        """Analyze historical volatility patterns"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days}d")
            
            if hist.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Calculate daily returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate volatility metrics
            volatility_20d = returns.tail(20).std() * np.sqrt(252)
            volatility_60d = returns.tail(60).std() * np.sqrt(252)
            volatility_252d = returns.std() * np.sqrt(252)
            
            # Identify volatility regimes
            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            current_vol = rolling_vol.iloc[-1]
            vol_percentile = stats.percentileofscore(rolling_vol.dropna(), current_vol)
            
            # Determine volatility regime
            if vol_percentile > 80:
                regime = "high"
            elif vol_percentile < 20:
                regime = "low"
            else:
                regime = "medium"
            
            return {
                "ticker": ticker,
                "current_volatility": float(current_vol),
                "volatility_20d": float(volatility_20d),
                "volatility_60d": float(volatility_60d),
                "volatility_252d": float(volatility_252d),
                "volatility_regime": regime,
                "volatility_percentile": float(vol_percentile),
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility for {ticker}: {e}")
            raise HTTPException(status_code=500, detail=f"Volatility analysis failed: {str(e)}")
    
    def calculate_optimal_allocation(self, tickers: List[str], capital: float, 
                                  risk_tolerance: str = 'medium') -> List[dict]:
        """Calculate optimal portfolio allocation using volatility analysis and BSM"""
        try:
            allocations = []
            
            # Analyze volatility for each ticker
            volatility_data = {}
            for ticker in tickers:
                volatility_data[ticker] = self.analyze_volatility(ticker)
            
            # Calculate base allocation (equal weight)
            base_allocation = 100 / len(tickers)
            
            # Adjust allocation based on volatility and risk tolerance
            risk_multiplier = {
                'low': 0.8,
                'medium': 1.0,
                'high': 1.2
            }.get(risk_tolerance, 1.0)
            
            for ticker in tickers:
                vol_data = volatility_data[ticker]
                
                # Adjust allocation based on volatility regime
                if vol_data['volatility_regime'] == 'high':
                    vol_adjustment = 0.7  # Reduce allocation for high volatility
                elif vol_data['volatility_regime'] == 'low':
                    vol_adjustment = 1.3  # Increase allocation for low volatility
                else:
                    vol_adjustment = 1.0
                
                # Calculate final allocation percentage
                adjusted_percentage = base_allocation * vol_adjustment * risk_multiplier
                
                # Calculate dollar allocation
                dollar_allocation = (capital * adjusted_percentage) / 100
                
                # Calculate stop loss based on volatility
                base_stop_loss = 0.05  # 5% base stop loss
                vol_stop_loss = min(vol_data['current_volatility'] * 0.5, 0.20)  # Max 20%
                stop_loss = max(base_stop_loss, vol_stop_loss)
                
                # Get current price for entry point
                stock = yf.Ticker(ticker)
                current_price = stock.history(period="1d")['Close'].iloc[-1]
                
                allocations.append({
                    "ticker": ticker,
                    "percentage": round(adjusted_percentage, 2),
                    "dollar_allocation": round(dollar_allocation, 2),
                    "current_price": round(float(current_price), 2),
                    "suggested_shares": int(dollar_allocation / current_price),
                    "stop_loss_percentage": round(stop_loss * 100, 1),
                    "stop_loss_price": round(current_price * (1 - stop_loss), 2),
                    "volatility_regime": vol_data['volatility_regime'],
                    "volatility_percentile": round(vol_data['volatility_percentile'], 1),
                    "entry_comment": f"Entry based on {vol_data['volatility_regime']} volatility regime analysis"
                })
            
            # Normalize allocations to 100%
            total_percentage = sum(alloc['percentage'] for alloc in allocations)
            for alloc in allocations:
                alloc['percentage'] = round((alloc['percentage'] / total_percentage) * 100, 2)
                alloc['dollar_allocation'] = round((capital * alloc['percentage']) / 100, 2)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error calculating optimal allocation: {e}")
            raise HTTPException(status_code=500, detail=f"Allocation calculation failed: {str(e)}")
    
    def calculate_portfolio_metrics(self, holdings: List[dict]) -> dict:
        """Calculate portfolio performance metrics"""
        try:
            total_value = sum(holding['current_value'] for holding in holdings)
            total_cost = sum(holding['cost_basis'] for holding in holdings)
            total_pnl = total_value - total_cost
            total_pnl_percentage = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            # Calculate individual stock performance
            stock_performance = []
            for holding in holdings:
                pnl = holding['current_value'] - holding['cost_basis']
                pnl_percentage = (pnl / holding['cost_basis']) * 100 if holding['cost_basis'] > 0 else 0
                
                stock_performance.append({
                    "ticker": holding['ticker'],
                    "pnl": round(pnl, 2),
                    "pnl_percentage": round(pnl_percentage, 2),
                    "current_value": holding['current_value'],
                    "cost_basis": holding['cost_basis']
                })
            
            # Calculate risk metrics
            returns = [stock['pnl_percentage'] for stock in stock_performance]
            portfolio_volatility = np.std(returns) if len(returns) > 1 else 0
            
            return {
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_percentage": round(total_pnl_percentage, 2),
                "portfolio_volatility": round(portfolio_volatility, 2),
                "stock_performance": stock_performance,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Portfolio metrics calculation failed: {str(e)}")

# Initialize optimizer
portfolio_optimizer = PortfolioOptimizer()

@router.post("/portfolio/analyze-volatility/{ticker}")
async def analyze_stock_volatility(ticker: str, days: int = 252):
    """Analyze volatility patterns for a specific stock"""
    try:
        result = portfolio_optimizer.analyze_volatility(ticker.upper(), days)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/calculate-allocation")
async def calculate_portfolio_allocation(
    tickers: List[str],
    capital: float,
    risk_tolerance: str = 'medium'
):
    """Calculate optimal portfolio allocation"""
    try:
        if not tickers or capital <= 0:
            raise HTTPException(status_code=400, detail="Invalid tickers or capital amount")
        
        if len(tickers) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 stocks allowed")
        
        result = portfolio_optimizer.calculate_optimal_allocation(
            [ticker.upper() for ticker in tickers],
            capital,
            risk_tolerance
        )
        
        return {
            "capital": capital,
            "risk_tolerance": risk_tolerance,
            "allocations": result,
            "total_percentage": round(sum(alloc['percentage'] for alloc in result), 2),
            "analysis_date": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/calculate-metrics")
async def calculate_portfolio_metrics(holdings: List[dict]):
    """Calculate portfolio performance metrics"""
    try:
        if not holdings:
            raise HTTPException(status_code=400, detail="No holdings provided")
        
        result = portfolio_optimizer.calculate_portfolio_metrics(holdings)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio/risk-analysis/{ticker}")
async def get_risk_analysis(ticker: str):
    """Get comprehensive risk analysis for a stock"""
    try:
        # Get volatility analysis
        vol_analysis = portfolio_optimizer.analyze_volatility(ticker.upper())
        
        # Get BSM fair value analysis
        bsm_analysis = bsm_analyzer.analyze_stock_valuation(ticker.upper())
        
        # Combine analyses
        risk_analysis = {
            "ticker": ticker.upper(),
            "volatility_analysis": vol_analysis,
            "fair_value_analysis": bsm_analysis,
            "risk_score": calculate_risk_score(vol_analysis, bsm_analysis),
            "analysis_date": datetime.now().isoformat()
        }
        
        return risk_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def calculate_risk_score(vol_analysis: dict, bsm_analysis: dict) -> dict:
    """Calculate overall risk score based on volatility and valuation"""
    try:
        # Volatility risk score (0-100, higher = more risky)
        vol_percentile = vol_analysis.get('volatility_percentile', 50)
        vol_risk_score = vol_percentile
        
        # Valuation risk score
        valuation_ratio = bsm_analysis.get('valuation_ratio', 1.0)
        if valuation_ratio > 1.5:
            val_risk_score = 80  # Overvalued = risky
        elif valuation_ratio < 0.8:
            val_risk_score = 20  # Undervalued = less risky
        else:
            val_risk_score = 50  # Fair value = medium risk
        
        # Combined risk score
        combined_risk_score = (vol_risk_score * 0.6) + (val_risk_score * 0.4)
        
        # Risk level classification
        if combined_risk_score > 70:
            risk_level = "high"
        elif combined_risk_score < 30:
            risk_level = "low"
        else:
            risk_level = "medium"
        
        return {
            "volatility_risk_score": round(vol_risk_score, 1),
            "valuation_risk_score": round(val_risk_score, 1),
            "combined_risk_score": round(combined_risk_score, 1),
            "risk_level": risk_level,
            "recommendation": get_risk_recommendation(risk_level, vol_analysis, bsm_analysis)
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        return {
            "volatility_risk_score": 50,
            "valuation_risk_score": 50,
            "combined_risk_score": 50,
            "risk_level": "medium",
            "recommendation": "Unable to calculate risk score"
        }

def get_risk_recommendation(risk_level: str, vol_analysis: dict, bsm_analysis: dict) -> str:
    """Generate risk-based investment recommendation"""
    vol_regime = vol_analysis.get('volatility_regime', 'medium')
    valuation_status = bsm_analysis.get('valuation_status', 'fair')
    
    if risk_level == "low":
        return f"Low risk stock. Volatility regime: {vol_regime}, Valuation: {valuation_status}. Consider larger position size."
    elif risk_level == "high":
        return f"High risk stock. Volatility regime: {vol_regime}, Valuation: {valuation_status}. Use smaller position size and tight stop loss."
    else:
        return f"Medium risk stock. Volatility regime: {vol_regime}, Valuation: {valuation_status}. Standard position sizing recommended."
