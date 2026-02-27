"""
Integration tests for the full pipeline.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from probabilistic_market_engine.core.physics_engine.engine import PhysicsEngine
from probabilistic_market_engine.core.liquidity_engine.engine import LiquidityEngine
from probabilistic_market_engine.core.feature_pipeline.pipeline import FeaturePipeline
from probabilistic_market_engine.core.regime_model.model import RegimeInferenceModel
from probabilistic_market_engine.core.outcome_models.experts import OutcomeExpertModels
from probabilistic_market_engine.core.inference_engine.engine import InferenceEngine
from probabilistic_market_engine.core.risk_engine.engine import RiskEngine

from probabilistic_market_engine.training.regime_training.trainer import RegimeTrainer
from probabilistic_market_engine.training.outcome_training.trainer import OutcomeTrainer
from probabilistic_market_engine.training.calibration.evaluator import CalibrationEvaluator

from probabilistic_market_engine.monitoring.drift_detection.detector import DriftDetector
from probabilistic_market_engine.monitoring.regime_tracking.tracker import RegimeTracker

from probabilistic_market_engine.persistence.model_registry.registry import ModelRegistry


class TestFullPipeline(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Generate synthetic OHLCV data
        n = 1000
        t = np.linspace(0, 20 * np.pi, n)
        
        # Create data with regime changes
        trend = np.sin(t * 0.1) * 10
        noise = np.random.randn(n) * 0.5
        closes = 100 + np.cumsum(trend * 0.01) + noise
        
        opens = closes + np.random.randn(n) * 0.2
        highs = np.maximum(opens, closes) + np.random.rand(n) * 0.5
        lows = np.minimum(opens, closes) - np.random.rand(n) * 0.5
        volumes = np.random.rand(n) * 1000 + 500
        
        self.ohlcv_data = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
        }, index=pd.date_range('2024-01-01', periods=n, freq='15min'))
    
    def test_full_pipeline(self):
        """Test the complete pipeline from raw data to prediction."""
        # 1. Create engines
        physics_engine = PhysicsEngine()
        liquidity_engine = LiquidityEngine()
        
        # 2. Create feature pipeline
        feature_pipeline = FeaturePipeline(
            physics_engine=physics_engine,
            liquidity_engine=liquidity_engine
        )
        
        # 3. Fit pipeline and get features
        feature_sets = feature_pipeline.fit_transform(self.ohlcv_data)
        
        # Should have features
        self.assertGreater(len(feature_sets), 0)
        self.assertTrue(all(fs.is_valid for fs in feature_sets))
        
        # 4. Extract regime features
        R_features = np.array([fs.R_t for fs in feature_sets])
        X_features = np.array([fs.X_t for fs in feature_sets])
        
        # 5. Train regime model
        regime_trainer = RegimeTrainer()
        regime_model = regime_trainer.train(R_features)
        
        self.assertTrue(regime_model._is_fitted)
        
        # 6. Get regime predictions
        regime_states = regime_model.predict_batch(R_features)
        regime_probs = np.array([
            [s.trend_probability, s.range_probability, s.shock_probability]
            for s in regime_states
        ])
        
        # 7. Train outcome models
        outcome_trainer = OutcomeTrainer()
        labels = outcome_trainer.generate_labels(
            self.ohlcv_data['close'].values[feature_pipeline.physics_engine.config.lookback_windows[-1] + 1:]
        )
        
        # Filter valid labels
        valid_mask = ~np.isnan(labels)
        
        if np.sum(valid_mask) > 100:
            outcome_models = OutcomeExpertModels()
            outcome_models.fit(
                X_features[valid_mask],
                labels[valid_mask].astype(int),
                regime_probs[valid_mask],
                feature_names=feature_pipeline._feature_names
            )
            
            # 8. Create inference engine
            inference_engine = InferenceEngine(outcome_models)
            
            # 9. Make predictions
            results = inference_engine.predict_batch(
                X_features,
                regime_states
            )
            
            # Should have predictions
            self.assertEqual(len(results), len(feature_sets))
            
            # 10. Risk engine
            risk_engine = RiskEngine()
            risk_adjustments = [risk_engine.compute(r) for r in results]
            
            # Should have risk adjustments
            self.assertEqual(len(risk_adjustments), len(results))
            
            # Check predictions are valid probabilities
            for result in results:
                self.assertGreaterEqual(result.prob_continuation, 0)
                self.assertLessEqual(result.prob_continuation, 1)
    
    def test_model_registry(self):
        """Test model save and load through registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup registry
            registry_path = os.path.join(tmpdir, "registry")
            os.makedirs(registry_path, exist_ok=True)
            
            # Create a mock config
            from probabilistic_market_engine.config.settings import PersistenceConfig
            config = PersistenceConfig(model_registry_path=registry_path)
            registry = ModelRegistry(config)
            
            # Create and train simple models
            physics = PhysicsEngine()
            liquidity = LiquidityEngine()
            pipeline = FeaturePipeline(physics_engine=physics, liquidity_engine=liquidity)
            
            # Fit pipeline
            feature_sets = pipeline.fit_transform(self.ohlcv_data)
            
            R_features = np.array([fs.R_t for fs in feature_sets])
            
            # Train regime model
            regime_model = RegimeInferenceModel()
            regime_model.fit(R_features)
            
            # Create outcome models (untrained for this test)
            outcome_models = OutcomeExpertModels()
            
            # Register version
            version = registry.register_version(
                version="v1.0.0",
                regime_model=regime_model,
                outcome_models=outcome_models,
                feature_pipeline=pipeline,
                description="Test version"
            )
            
            self.assertEqual(version.version, "v1.0.0")
            
            # Load version
            loaded_regime, loaded_outcome, _ = registry.load_version("v1.0.0")
            
            self.assertIsNotNone(loaded_regime)
            self.assertIsNotNone(loaded_outcome)
    
    def test_monitoring(self):
        """Test monitoring components."""
        # Setup
        physics = PhysicsEngine()
        liquidity = LiquidityEngine()
        pipeline = FeaturePipeline(physics_engine=physics, liquidity_engine=liquidity)
        
        feature_sets = pipeline.fit_transform(self.ohlcv_data)
        
        # Train regime model
        R_features = np.array([fs.R_t for fs in feature_sets])
        regime_model = RegimeInferenceModel()
        regime_model.fit(R_features)
        
        # Create monitoring
        drift_detector = DriftDetector()
        regime_tracker = RegimeTracker()
        
        # Set reference
        regime_states = regime_model.predict_batch(R_features[:500])
        regime_probs = np.array([
            [s.trend_probability, s.range_probability, s.shock_probability]
            for s in regime_states
        ])
        
        drift_detector.set_reference(
            features=X_features[:500] if 'X_features' in dir() else np.random.randn(500, 10),
            regime_probs=regime_probs
        )
        
        # Simulate monitoring
        for i, (fs, state) in enumerate(zip(feature_sets[500:550], 
                                            regime_model.predict_batch(R_features[500:550]))):
            drift_detector.update(
                features=fs.X_t,
                regime_probs=state.smoothed_probs,
                prediction=0.5,
                timestamp=fs.timestamp
            )
            regime_tracker.update(state)
        
        # Get stats
        drift_stats = drift_detector.get_feature_statistics()
        regime_summary = regime_tracker.get_summary()
        
        self.assertIsNotNone(regime_summary)
    
    def test_calibration_evaluation(self):
        """Test calibration evaluation."""
        # Create synthetic predictions and outcomes
        np.random.seed(42)
        n = 500
        
        # Well-calibrated predictions
        true_probs = np.random.rand(n)
        outcomes = (np.random.rand(n) < true_probs).astype(int)
        
        # Add some noise to predictions (miscalibration)
        predictions = true_probs + np.random.randn(n) * 0.1
        predictions = np.clip(predictions, 0.01, 0.99)
        
        # Evaluate
        evaluator = CalibrationEvaluator()
        result = evaluator.evaluate(outcomes, predictions)
        
        # Check results
        self.assertGreaterEqual(result.brier_score, 0)
        self.assertLessEqual(result.brier_score, 0.25)  # Max for binary
        self.assertGreaterEqual(result.expected_calibration_error, 0)


if __name__ == '__main__':
    unittest.main()
