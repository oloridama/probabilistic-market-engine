"""
Tests for Risk Engine (Layer 7).
"""

import unittest

from probabilistic_market_engine.core.risk_engine.engine import RiskEngine, RiskAdjustment
from probabilistic_market_engine.core.inference_engine.engine import InferenceResult


class TestRiskEngine(unittest.TestCase):
    """Test cases for RiskEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = RiskEngine()
        
        # Create sample inference results
        self.neutral_result = InferenceResult(
            prob_continuation=0.5,
            shock_probability=0.0,
            regime_entropy=0.0
        )
        
        self.strong_long_result = InferenceResult(
            prob_continuation=0.9,
            shock_probability=0.0,
            regime_entropy=0.0
        )
        
        self.strong_short_result = InferenceResult(
            prob_continuation=0.1,
            shock_probability=0.0,
            regime_entropy=0.0
        )
        
        self.shock_result = InferenceResult(
            prob_continuation=0.7,
            shock_probability=0.8,
            regime_entropy=0.5
        )
        
        self.uncertain_result = InferenceResult(
            prob_continuation=0.7,
            shock_probability=0.0,
            regime_entropy=1.0  # Max entropy
        )
    
    def test_initialization(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
    
    def test_neutral_signal(self):
        """Test risk adjustment for neutral signal."""
        adjustment = self.engine.compute(self.neutral_result)
        
        # Neutral signal should have zero risk scaling
        self.assertEqual(adjustment.suggested_direction, 0)
        self.assertAlmostEqual(adjustment.signal_strength_factor, 0.0)
    
    def test_long_signal(self):
        """Test risk adjustment for long signal."""
        adjustment = self.engine.compute(self.strong_long_result)
        
        self.assertEqual(adjustment.suggested_direction, 1)
        self.assertGreater(adjustment.signal_strength_factor, 0.5)
    
    def test_short_signal(self):
        """Test risk adjustment for short signal."""
        adjustment = self.engine.compute(self.strong_short_result)
        
        self.assertEqual(adjustment.suggested_direction, -1)
        self.assertGreater(adjustment.signal_strength_factor, 0.5)
    
    def test_shock_penalty(self):
        """Test that shock probability reduces risk."""
        normal = self.engine.compute(self.strong_long_result)
        shock = self.engine.compute(self.shock_result)
        
        # Shock should reduce risk scaling
        self.assertLess(shock.risk_scaling_factor, normal.risk_scaling_factor)
        self.assertLess(shock.shock_penalty_factor, 1.0)
    
    def test_uncertainty_penalty(self):
        """Test that uncertainty reduces risk."""
        certain = self.engine.compute(self.strong_long_result)
        uncertain = self.engine.compute(self.uncertain_result)
        
        # Uncertainty should reduce risk scaling
        self.assertLess(uncertain.risk_scaling_factor, certain.risk_scaling_factor)
    
    def test_risk_scaling_limits(self):
        """Test that risk scaling is within limits."""
        adjustment = self.engine.compute(self.strong_long_result)
        
        self.assertGreaterEqual(adjustment.risk_scaling_factor, 
                               self.engine.config.min_risk_scale)
        self.assertLessEqual(adjustment.risk_scaling_factor,
                            self.engine.config.max_risk_scale)
    
    def test_compute_from_probs(self):
        """Test computation from raw probabilities."""
        adjustment = self.engine.compute_from_probs(
            prob=0.8,
            shock_prob=0.1,
            entropy=0.2
        )
        
        self.assertIsInstance(adjustment, RiskAdjustment)
        self.assertEqual(adjustment.suggested_direction, 1)
    
    def test_signal_confidence(self):
        """Test signal confidence computation."""
        weak_signal = InferenceResult(prob_continuation=0.55, shock_probability=0, regime_entropy=0)
        strong_signal = InferenceResult(prob_continuation=0.9, shock_probability=0, regime_entropy=0)
        
        weak_adj = self.engine.compute(weak_signal)
        strong_adj = self.engine.compute(strong_signal)
        
        self.assertLess(weak_adj.signal_confidence, strong_adj.signal_confidence)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        adjustment = self.engine.compute(self.strong_long_result)
        d = adjustment.to_dict()
        
        self.assertIn('risk_scaling_factor', d)
        self.assertIn('suggested_direction', d)
        self.assertIn('signal_confidence', d)


if __name__ == '__main__':
    unittest.main()
