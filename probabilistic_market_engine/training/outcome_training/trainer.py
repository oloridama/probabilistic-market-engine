"""
Outcome Model Training

Trains regime-conditional outcome experts using soft regime weights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

from probabilistic_market_engine.config.settings import OutcomeModelConfig
from probabilistic_market_engine.core.outcome_models.experts import OutcomeExpertModels
from probabilistic_market_engine.core.regime_model.model import RegimeInferenceModel, RegimeState


class OutcomeTrainer:
    """
    Trainer for regime-conditional outcome experts.
    
    Handles:
    - Label generation (continuation targets)
    - Soft-weighted training
    - Feature importance tracking
    """
    
    def __init__(self, config: Optional[OutcomeModelConfig] = None):
        self.config = config or OutcomeModelConfig()
        self.logger = logging.getLogger(__name__)
    
    def generate_labels(self, closes: np.ndarray, 
                       prediction_horizon: Optional[int] = None) -> np.ndarray:
        """
        Generate binary continuation labels.
        
        Label Y=1 if price moves in same direction as recent trend.
        
        Args:
            closes: Closing prices
            prediction_horizon: Bars to look ahead
        
        Returns:
            Binary labels (0 or 1), with NaN for unlabelable points
        """
        horizon = prediction_horizon or self.config.prediction_horizon
        threshold = self.config.continuation_threshold
        
        n = len(closes)
        labels = np.full(n, np.nan)
        
        # Need at least 2 bars for trend + horizon bars for prediction
        if n < horizon + 2:
            return labels
        
        for i in range(n - horizon):
            # Recent trend (last 4 bars)
            recent_start = max(0, i - 3)
            recent_return = (closes[i] - closes[recent_start]) / closes[recent_start]
            
            # Future return
            future_return = (closes[i + horizon] - closes[i]) / closes[i]
            
            # Label = 1 if future move is in same direction as recent trend
            # AND magnitude exceeds threshold
            if abs(recent_return) < 1e-10:
                # No clear trend - skip
                continue
            
            same_direction = np.sign(recent_return) == np.sign(future_return)
            exceeds_threshold = abs(future_return) >= threshold
            
            if same_direction and exceeds_threshold:
                labels[i] = 1
            else:
                labels[i] = 0
        
        return labels
    
    def train(self, X: np.ndarray, 
              closes: np.ndarray,
              regime_model: RegimeInferenceModel,
              R_features: np.ndarray,
              feature_names: Optional[List[str]] = None) -> OutcomeExpertModels:
        """
        Train outcome expert models.
        
        Args:
            X: Full feature matrix
            closes: Closing prices for label generation
            regime_model: Trained regime model
            R_features: Regime features for regime probabilities
            feature_names: Feature names for interpretability
        
        Returns:
            Trained OutcomeExpertModels
        """
        self.logger.info(f"Training outcome models on {len(X)} samples")
        
        # Generate labels
        labels = self.generate_labels(closes)
        
        # Get regime probabilities
        regime_states = regime_model.predict_batch(R_features)
        regime_probs = np.array([
            [s.trend_probability, s.range_probability, s.shock_probability]
            for s in regime_states
        ])
        
        # Filter valid samples (non-NaN labels)
        valid_mask = ~np.isnan(labels)
        X_valid = X[valid_mask]
        y_valid = labels[valid_mask].astype(int)
        probs_valid = regime_probs[valid_mask]
        
        self.logger.info(f"Valid samples: {len(y_valid)} (positive rate: {np.mean(y_valid):.2%})")
        
        # Train models
        experts = OutcomeExpertModels(self.config)
        experts.fit(X_valid, y_valid, probs_valid, feature_names)
        
        # Log feature importance
        self._log_feature_importance(experts)
        
        return experts
    
    def _log_feature_importance(self, experts: OutcomeExpertModels):
        """Log feature importance for each regime."""
        self.logger.info("=== Feature Importance ===")
        
        for regime in ['trend', 'range', 'shock']:
            importance = experts.get_feature_importance(regime)
            if importance:
                self.logger.info(f"\n{regime.upper()} Regime:")
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, imp in top_features:
                    self.logger.info(f"  {feat}: {imp:.3f}")


def train_outcome_models(ohlcv_data: pd.DataFrame,
                        feature_pipeline,
                        regime_model: RegimeInferenceModel,
                        config: Optional[OutcomeModelConfig] = None) -> OutcomeExpertModels:
    """
    Convenience function to train outcome models.
    
    Args:
        ohlcv_data: OHLCV DataFrame
        feature_pipeline: Fitted feature pipeline
        regime_model: Trained regime model
        config: Optional outcome model config
    
    Returns:
        Trained OutcomeExpertModels
    """
    # Get feature sets
    feature_sets = feature_pipeline.fit_transform(ohlcv_data)
    
    # Extract features
    X = np.array([fs.X_t for fs in feature_sets])
    R = np.array([fs.R_t for fs in feature_sets])
    
    # Align with closes (need to skip initial bars used for feature computation)
    min_window = max(feature_pipeline.config.rolling_window,
                    max(feature_pipeline.physics_engine.config.lookback_windows) + 1)
    closes = ohlcv_data['close'].values[min_window:min_window + len(X)]
    
    # Train models
    trainer = OutcomeTrainer(config)
    experts = trainer.train(X, closes, regime_model, R, feature_pipeline._feature_names)
    
    return experts
