"""
Tests for Physics Engine (Layer 1).
"""

import unittest
import numpy as np
import pandas as pd

from probabilistic_market_engine.core.physics_engine.engine import PhysicsEngine, PhysicsState


class TestPhysicsEngine(unittest.TestCase):
    """Test cases for PhysicsEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = PhysicsEngine()
        
        # Create sample OHLCV data
        np.random.seed(42)
        n = 100
        
        # Generate trending data
        closes = 100 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
        opens = closes + np.random.randn(n) * 0.3
        highs = np.maximum(opens, closes) + np.random.rand(n) * 1.5
        lows = np.minimum(opens, closes) - np.random.rand(n) * 1.5
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
        self.assertEqual(len(self.engine._returns_buffer), 0)
    
    def test_compute_basic(self):
        """Test basic computation."""
        state = self.engine.compute(self.sample_data)
        
        self.assertIsInstance(state, PhysicsState)
        self.assertIsNotNone(state.timestamp)
    
    def test_compute_features(self):
        """Test that all features are computed."""
        state = self.engine.compute(self.sample_data)
        
        # Check all expected features exist
        self.assertIsNotNone(state.pressure_norm)
        self.assertIsNotNone(state.flow_short)
        self.assertIsNotNone(state.flow_medium)
        self.assertIsNotNone(state.flow_long)
        self.assertIsNotNone(state.path_efficiency)
        self.assertIsNotNone(state.energy_total)
        self.assertIsNotNone(state.shock_index)
        self.assertIsNotNone(state.alignment_score)
    
    def test_vector_conversion(self):
        """Test conversion to vector."""
        state = self.engine.compute(self.sample_data)
        vector = state.to_vector()
        
        self.assertIsInstance(vector, np.ndarray)
        self.assertGreater(len(vector), 0)
    
    def test_pressure_normalization(self):
        """Test pressure normalization."""
        # Compute multiple times to build buffer
        for i in range(10, len(self.sample_data)):
            window = self.sample_data.iloc[:i]
            state = self.engine.compute(window)
        
        # Pressure should be normalized (z-score)
        self.assertIsNotNone(state.pressure_norm)
        self.assertIsInstance(state.pressure_norm, float)
    
    def test_multi_scale_flow(self):
        """Test multi-scale flow computation."""
        state = self.engine.compute(self.sample_data)
        
        # Flows should be finite numbers
        self.assertTrue(np.isfinite(state.flow_short))
        self.assertTrue(np.isfinite(state.flow_medium))
        self.assertTrue(np.isfinite(state.flow_long))
    
    def test_shock_detection(self):
        """Test shock detection."""
        # Create data with shock
        shocked_data = self.sample_data.copy()
        shocked_data.loc[shocked_data.index[-1], 'close'] *= 1.05  # 5% jump
        shocked_data.loc[shocked_data.index[-1], 'high'] *= 1.06
        
        state = self.engine.compute(shocked_data)
        
        # Shock index should be elevated
        self.assertGreaterEqual(state.shock_index, 0)
    
    def test_determinism(self):
        """Test that computations are deterministic."""
        state1 = self.engine.compute(self.sample_data)
        
        # Reset and recompute
        self.engine.reset()
        state2 = self.engine.compute(self.sample_data)
        
        # Should be identical
        vector1 = state1.to_vector()
        vector2 = state2.to_vector()
        
        np.testing.assert_array_almost_equal(vector1, vector2)
    
    def test_reset(self):
        """Test reset functionality."""
        self.engine.compute(self.sample_data)
        self.assertGreater(len(self.engine._returns_buffer), 0)
        
        self.engine.reset()
        self.assertEqual(len(self.engine._returns_buffer), 0)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        short_data = self.sample_data.iloc[:5]
        
        with self.assertRaises(ValueError):
            self.engine.compute(short_data)


if __name__ == '__main__':
    unittest.main()
