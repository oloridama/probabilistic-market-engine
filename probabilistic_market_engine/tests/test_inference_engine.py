"""
Tests for Inference Engine (Layer 6).
"""

import unittest
import numpy as np

from probabilistic_market_engine.core.inference_engine.engine import InferenceEngine, InferenceResult, compute_mixture_prediction
from probabilistic_market_engine.core.outcome_models.experts import OutcomeExpertModels, OutcomePrediction
from probabilistic_market_engine.core.regime_model.model import RegimeState


class TestInferenceEngine(unittest.TestCase):
    """Test cases for InferenceEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock outcome models
        self.mock_experts = OutcomeExpertModels()
        
        # Mock the predict method
        def mock_predict(X_t, regime):
            pred = OutcomePrediction(regime=regime)
            if regime == 'trend':
                pred.probability = 0.7  # Trend tends to continue
            elif regime == 'range':
                pred.probability = 0.5  # Range is neutral
            elif regime == 'shock':
                pred.probability = 0.3  # Shock tends to reverse
            return pred
        
        self.mock_experts.predict = mock_predict
        
        # Mock predict_all
        def mock_predict_all(X_t):
            return {
                'trend': mock_predict(X_t, 'trend'),
                'range': mock_predict(X_t, 'range'),
                'shock': mock_predict(X_t, 'shock'),
            }
        
        self.mock_experts.predict_all = mock_predict_all
        
        self.engine = InferenceEngine(self.mock_experts)
        
        # Sample data
        self.X_t = np.random.randn(25)
        self.regime_state = RegimeState()
        self.regime_state.trend_probability = 0.5
        self.regime_state.range_probability = 0.3
        self.regime_state.shock_probability = 0.2
        self.regime_state.smoothed_probs = np.array([0.5, 0.3, 0.2])
        self.regime_state.regime_entropy = 0.5
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.expert_models, self.mock_experts)
    
    def test_predict(self):
        """Test prediction."""
        result = self.engine.predict(self.X_t, self.regime_state)
        
        self.assertIsInstance(result, InferenceResult)
        self.assertGreaterEqual(result.prob_continuation, 0)
        self.assertLessEqual(result.prob_continuation, 1)
    
    def test_mixture_computation(self):
        """Test mixture aggregation."""
        result = self.engine.predict(self.X_t, self.regime_state)
        
        # Expected: 0.7*0.5 + 0.5*0.3 + 0.3*0.2 = 0.35 + 0.15 + 0.06 = 0.56
        expected = 0.7 * 0.5 + 0.5 * 0.3 + 0.3 * 0.2
        self.assertAlmostEqual(result.prob_continuation, expected, places=5)
    
    def test_regime_probabilities_preserved(self):
        """Test that regime probabilities are preserved in output."""
        result = self.engine.predict(self.X_t, self.regime_state)
        
        self.assertEqual(result.regime_probabilities['trend'], 0.5)
        self.assertEqual(result.regime_probabilities['range'], 0.3)
        self.assertEqual(result.regime_probabilities['shock'], 0.2)
    
    def test_shock_probability_exposed(self):
        """Test that shock probability is exposed separately."""
        result = self.engine.predict(self.X_t, self.regime_state)
        
        self.assertEqual(result.shock_probability, 0.2)
    
    def test_prediction_confidence(self):
        """Test prediction confidence computation."""
        result = self.engine.predict(self.X_t, self.regime_state)
        
        # Confidence should be in [0, 1]
        self.assertGreaterEqual(result.prediction_confidence, 0)
        self.assertLessEqual(result.prediction_confidence, 1)
    
    def test_model_version(self):
        """Test model version setting."""
        self.engine.set_model_version('v2.0.0')
        result = self.engine.predict(self.X_t, self.regime_state)
        
        self.assertEqual(result.model_version, 'v2.0.0')
    
    def test_batch_predict(self):
        """Test batch prediction."""
        X_features = np.random.randn(10, 25)
        regime_states = [self.regime_state] * 10
        
        results = self.engine.predict_batch(X_features, regime_states)
        
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, InferenceResult)
    
    def test_compute_mixture_prediction_standalone(self):
        """Test standalone mixture prediction function."""
        expert_probs = {
            'trend': 0.7,
            'range': 0.5,
            'shock': 0.3
        }
        regime_probs = {
            'trend': 0.5,
            'range': 0.3,
            'shock': 0.2
        }
        
        result = compute_mixture_prediction(expert_probs, regime_probs)
        
        expected = 0.7 * 0.5 + 0.5 * 0.3 + 0.3 * 0.2
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_to_dict(self):
        """Test result dictionary conversion."""
        result = self.engine.predict(self.X_t, self.regime_state)
        d = result.to_dict()
        
        self.assertIn('prob_continuation', d)
        self.assertIn('regime_probabilities', d)
        self.assertIn('shock_probability', d)


if __name__ == '__main__':
    unittest.main()
