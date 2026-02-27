"""
Tests for Liquidity Engine (Layer 2).
"""

import unittest
import numpy as np
import pandas as pd

from probabilistic_market_engine.core.liquidity_engine.engine import LiquidityEngine, LiquidityState


class TestLiquidityEngine(unittest.TestCase):
    """Test cases for LiquidityEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = LiquidityEngine()
        
        # Create sample OHLCV data with clear pivot points
        np.random.seed(42)
        n = 200
        
        # Create oscillating data for pivot detection
        t = np.linspace(0, 10 * np.pi, n)
        closes = 100 + 5 * np.sin(t) + np.random.randn(n) * 0.5
        opens = closes + np.random.randn(n) * 0.2
        highs = np.maximum(opens, closes) + np.random.rand(n) * 0.5
        lows = np.minimum(opens, closes) - np.random.rand(n) * 0.5
        volumes = np.random.rand(n) * 1000 + 500
        
        self.sample_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(len(self.engine._pivots), 0)
    
    def test_compute_basic(self):
        """Test basic computation."""
        state = self.engine.compute(self.sample_data)
        
        self.assertIsInstance(state, LiquidityState)
        self.assertIsNotNone(state.timestamp)
    
    def test_liquidity_features(self):
        """Test liquidity feature computation."""
        state = self.engine.compute(self.sample_data)
        
        # Check all expected features exist
        self.assertIsNotNone(state.distance_to_support)
        self.assertIsNotNone(state.distance_to_resistance)
        self.assertIsNotNone(state.liquidity_density)
        self.assertIsNotNone(state.relative_position)
        self.assertIsNotNone(state.turnover_intensity)
    
    def test_distance_ranges(self):
        """Test that distances are in valid range."""
        state = self.engine.compute(self.sample_data)
        
        # Distances should be in [0, 1]
        self.assertGreaterEqual(state.distance_to_support, 0)
        self.assertLessEqual(state.distance_to_support, 1)
        self.assertGreaterEqual(state.distance_to_resistance, 0)
        self.assertLessEqual(state.distance_to_resistance, 1)
    
    def test_liquidity_density_range(self):
        """Test liquidity density is in valid range."""
        state = self.engine.compute(self.sample_data)
        
        # Density should be in [0, 1]
        self.assertGreaterEqual(state.liquidity_density, 0)
        self.assertLessEqual(state.liquidity_density, 1)
    
    def test_pivot_detection(self):
        """Test pivot detection."""
        # Compute on data
        for i in range(50, len(self.sample_data)):
            window = self.sample_data.iloc[i-50:i]
            self.engine.compute(window)
        
        # Should have detected some pivots
        self.assertGreater(len(self.engine._pivots), 0)
    
    def test_time_decay(self):
        """Test time decay of pivots."""
        # Add pivots
        for i in range(50, len(self.sample_data)):
            window = self.sample_data.iloc[i-50:i]
            self.engine.compute(window)
        
        initial_count = len(self.engine._pivots)
        initial_weights = [w for p, w in self.engine._pivots]
        
        # Apply decay multiple times
        for _ in range(100):
            self.engine._apply_time_decay()
        
        # Some pivots should have been removed
        self.assertLessEqual(len(self.engine._pivots), initial_count)
    
    def test_vector_conversion(self):
        """Test conversion to vector."""
        state = self.engine.compute(self.sample_data)
        vector = state.to_vector()
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertGreater(len(vector), 0)
    
    def test_reset(self):
        """Test reset functionality."""
        self.engine.compute(self.sample_data)
        self.assertGreater(len(self.engine._pivots), 0)
        
        self.engine.reset()
        self.assertEqual(len(self.engine._pivots), 0)


if __name__ == '__main__':
    unittest.main()
