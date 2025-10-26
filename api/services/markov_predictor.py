"""
Markov Chain Predictor for Stock Price Predictions
Implements discrete state Markov chains for pattern-based predictions
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MarkovChainPredictor:
    """Markov Chain predictor for stock price patterns"""
    
    def __init__(self, num_states: int = 20, smoothing_factor: float = 0.1):
        """
        Initialize Markov Chain predictor
        
        Args:
            num_states: Number of discrete price states
            smoothing_factor: Laplace smoothing factor to avoid zero probabilities
        """
        self.num_states = num_states
        self.smoothing_factor = smoothing_factor
        self.transition_matrix = None
        self.price_states = None
        self.state_boundaries = None
        
    def create_price_states(self, prices: np.ndarray) -> np.ndarray:
        """
        Create discrete price states from continuous prices
        
        Args:
            prices: Array of historical prices
            
        Returns:
            Array of state center prices
        """
        # Use percentiles to create more meaningful states
        percentiles = np.linspace(0, 100, self.num_states + 1)
        boundaries = np.percentile(prices, percentiles)
        
        # Create state centers
        self.price_states = np.array([(boundaries[i] + boundaries[i+1]) / 2 
                                    for i in range(self.num_states)])
        self.state_boundaries = boundaries
        
        logger.info(f"Created {self.num_states} price states from ${self.price_states[0]:.2f} to ${self.price_states[-1]:.2f}")
        return self.price_states
    
    def price_to_state(self, price: float) -> int:
        """Convert price to state index"""
        if self.state_boundaries is None:
            raise ValueError("Price states not initialized")
        
        # Find which state the price belongs to
        state_idx = np.digitize(price, self.state_boundaries) - 1
        # Clamp to valid range
        state_idx = max(0, min(state_idx, self.num_states - 1))
        return state_idx
    
    def state_to_price(self, state_idx: int) -> float:
        """Convert state index to price"""
        if self.price_states is None:
            raise ValueError("Price states not initialized")
        
        if 0 <= state_idx < self.num_states:
            return self.price_states[state_idx]
        else:
            # Return closest valid state
            state_idx = max(0, min(state_idx, self.num_states - 1))
            return self.price_states[state_idx]
    
    def build_transition_matrix(self, price_sequence: np.ndarray) -> np.ndarray:
        """
        Build transition probability matrix from price sequence
        
        Args:
            price_sequence: Historical price data
            
        Returns:
            Transition probability matrix
        """
        logger.info(f"Building transition matrix from {len(price_sequence)} price points")
        
        # Create price states
        self.create_price_states(price_sequence)
        
        # Initialize transition matrix
        transition_matrix = np.zeros((self.num_states, self.num_states))
        
        # Map prices to states
        state_indices = [self.price_to_state(price) for price in price_sequence]
        
        # Count transitions
        for i in range(len(state_indices) - 1):
            current_state = state_indices[i]
            next_state = state_indices[i + 1]
            transition_matrix[current_state, next_state] += 1
        
        # Apply Laplace smoothing to avoid zero probabilities
        transition_matrix += self.smoothing_factor
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis])
        
        self.transition_matrix = transition_matrix
        
        logger.info(f"Transition matrix built: {self.num_states}x{self.num_states}")
        return transition_matrix
    
    def predict_next_states(self, current_price: float, steps: int = 30) -> Tuple[List[float], List[float]]:
        """
        Predict future states using Markov chain
        
        Args:
            current_price: Current stock price
            steps: Number of steps to predict
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not built")
        
        # Start with current state
        current_state = self.price_to_state(current_price)
        current_prob = np.zeros(self.num_states)
        current_prob[current_state] = 1.0
        
        predictions = []
        probabilities = []
        
        for step in range(steps):
            # Calculate next state probabilities
            current_prob = current_prob @ self.transition_matrix
            
            # Get most likely state
            most_likely_state = np.argmax(current_prob)
            predicted_price = self.state_to_price(most_likely_state)
            
            predictions.append(predicted_price)
            probabilities.append(current_prob[most_likely_state])
        
        return predictions, probabilities
    
    def predict_with_uncertainty(self, current_price: float, steps: int = 30) -> Tuple[List[float], List[float], List[float]]:
        """
        Predict with uncertainty bounds
        
        Args:
            current_price: Current stock price
            steps: Number of steps to predict
            
        Returns:
            Tuple of (predictions, upper_bounds, lower_bounds)
        """
        predictions, probabilities = self.predict_next_states(current_price, steps)
        
        # Calculate uncertainty based on probability distribution
        upper_bounds = []
        lower_bounds = []
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Uncertainty increases with lower probability confidence
            uncertainty_factor = (1 - prob) * 0.2  # Increased to 20% max uncertainty
            uncertainty = pred * uncertainty_factor
            
            upper_bounds.append(pred + uncertainty)
            lower_bounds.append(pred - uncertainty)
        
        return predictions, upper_bounds, lower_bounds
    
    def get_state_probabilities(self, current_price: float, steps: int = 1) -> np.ndarray:
        """
        Get probability distribution over all states
        
        Args:
            current_price: Current stock price
            steps: Number of steps ahead
            
        Returns:
            Probability distribution over states
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix not built")
        
        current_state = self.price_to_state(current_price)
        current_prob = np.zeros(self.num_states)
        current_prob[current_state] = 1.0
        
        for _ in range(steps):
            current_prob = current_prob @ self.transition_matrix
        
        return current_prob
    
    def analyze_patterns(self, price_sequence: np.ndarray) -> dict:
        """
        Analyze price patterns and return insights
        
        Args:
            price_sequence: Historical price data
            
        Returns:
            Dictionary with pattern analysis
        """
        if self.transition_matrix is None:
            self.build_transition_matrix(price_sequence)
        
        # Calculate stationary distribution
        eigenvals, eigenvecs = np.linalg.eig(self.transition_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvals - 1))
        stationary_dist = np.real(eigenvecs[:, stationary_idx])
        stationary_dist = stationary_dist / stationary_dist.sum()
        
        # Find most likely states
        most_likely_states = np.argsort(stationary_dist)[-3:][::-1]
        
        analysis = {
            "stationary_distribution": stationary_dist,
            "most_likely_states": most_likely_states,
            "most_likely_prices": [self.state_to_price(state) for state in most_likely_states],
            "transition_matrix": self.transition_matrix,
            "price_states": self.price_states
        }
        
        return analysis


def test_markov_predictor():
    """Test the Markov chain predictor"""
    # Generate sample data
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.normal(0, 2, 1000))
    
    # Create predictor
    predictor = MarkovChainPredictor(num_states=15)
    
    # Build transition matrix
    predictor.build_transition_matrix(prices)
    
    # Make prediction
    current_price = prices[-1]
    predictions, upper, lower = predictor.predict_with_uncertainty(current_price, 10)
    
    print(f"Current price: ${current_price:.2f}")
    print(f"Predictions: {[f'${p:.2f}' for p in predictions[:5]]}")
    print(f"Upper bounds: {[f'${u:.2f}' for u in upper[:5]]}")
    print(f"Lower bounds: {[f'${l:.2f}' for l in lower[:5]]}")


if __name__ == "__main__":
    test_markov_predictor()
