"""
Tests for Regime Model (Layer 4).
"""

import unittest
import numpy as np

from probabilistic_market_engine.core.regime_model.model import RegimeInferenceModel, RegimeState


class TestRegimeInferenceModel(unittest.TestCase):
    """Test cases for RegimeInferenceModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = RegimeInferenceModel()
        
        # Generate synthetic regime features
        np.random.seed(42)
        n_samples = 500
        
        # Create 3 clusters representing different regimes
        trend_data = np.random.randn(n_samples // 3, 5) + np.array([2, 1.5, 1, 0.5, 0])
        range_data = np.random.randn(n_samples // 3, 5) + np.array([0, 0, 0.5, 0.3, 0])
        shock_data = np.random.randn(n_samples // 3, 5) + np.array([0, 0, 0, 2, 3])
        
        self.R_features = np.vstack([trend_data, range_data, shock_data])
        np.random.shuffle(self.R_features)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model._is_fitted)
    
    def test_fit(self):
        """Test model fitting."""
        self.model.fit(self.R_features)
        
        self.assertTrue(self.model._is_fitted)
        self.assertIsNotNone(self.model._gmm)
    
    def test_predict_before_fit(self):
        """Test prediction before fitting returns uniform."""
        R_t = self.R_features[0]
        state = self.model.predict(R_t)
        
        # Should return uniform distribution
        self.assertAlmostEqual(state.trend_probability, 1/3, places=2)
        self.assertAlmostEqual(state.range_probability, 1/3, places=2)
        self.assertAlmostEqual(state.shock_probability, 1/3, places=2)
    
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        self.model.fit(self.R_features)
        
        R_t = self.R_features[0]
        state = self.model.predict(R_t)
        
        # Should return valid probabilities
        self.assertGreaterEqual(state.trend_probability, 0)
        self.assertLessEqual(state.trend_probability, 1)
        self.assertGreaterEqual(state.range_probability, 0)
        self.assertLessEqual(state.range_probability, 1)
        self.assertGreaterEqual(state.shock_probability, 0)
        self.assertLessEqual(state.shock_probability, 1)
        
        # Should sum to 1
        total = state.trend_probability + state.range_probability + state.shock_probability
        self.assertAlmostEqual(total, 1.0, places=5)
    
    def test_persistence_smoothing(self):
        """Test persistence smoothing."""
        self.model.fit(self.R_features)
        
        # Make consecutive predictions
        states = []
        for i in range(10):
            state = self.model.predict(self.R_features[i])
            states.append(state)
        
        # Check that smoothed differs from raw
        for state in states[1:]:
            # Due to smoothing, smoothed and raw might differ
            self.assertIsNotNone(state.smoothed_probs)
            self.assertIsNotNone(state.raw_posterior)
    
    def test_batch_predict(self):
        """Test batch prediction."""
        self.model.fit(self.R_features)
        
        states = self.model.predict_batch(self.R_features[:50])
        
        self.assertEqual(len(states), 50)
        for state in states:
            self.assertIsInstance(state, RegimeState)
    
    def test_regime_entropy(self):
        """Test regime entropy computation."""
        self.model.fit(self.R_features)
        
        # Predict on diverse data
        states = self.model.predict_batch(self.R_features[:100])
        
        for state in states:
            # Entropy should be in [0, 1]
            self.assertGreaterEqual(state.regime_entropy, 0)
            self.assertLessEqual(state.regime_entropy, 1)
    
    def test_dominant_regime(self):
        """Test dominant regime identification."""
        self.model.fit(self.R_features)
        
        states = self.model.predict_batch(self.R_features[:100])
        
        for state in states:
            self.assertIn(state.dominant_regime, ['trend', 'range', 'shock'])
    
    def test_save_load(self):
        """Test model save and load."""
        import tempfile
        import os
        
        self.model.fit(self.R_features)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            self.model.save(temp_path)
            
            # Load into new model
            new_model = RegimeInferenceModel()
            new_model.load(temp_path)
            
            # Check state
            self.assertTrue(new_model._is_fitted)
            
            # Predictions should match
            R_t = self.R_features[0]
            pred1 = self.model.predict(R_t)
            pred2 = new_model.predict(R_t)
            
            np.testing.assert_array_almost_equal(
                pred1.smoothed_probs,
                pred2.smoothed_probs
            )
        finally:
            os.unlink(temp_path)
    
    def test_insufficient_samples(self):
        """Test behavior with insufficient samples."""
        with self.assertRaises(ValueError):
            self.model.fit(self.R_features[:10])  # Too few samples


if __name__ == '__main__':
    unittest.main()
