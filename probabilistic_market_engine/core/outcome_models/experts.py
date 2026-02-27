"""
Layer 5: Regime-Conditional Outcome Experts

For each regime k: Train logistic regression Pr(Y=1 | X, Z=k)
Training uses soft weights = Pr(Z=k)
Must be walk-forward validated
Supports recalibration

No regime inference logic allowed here.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _sigmoid_calibration
import pickle

from probabilistic_market_engine.config.settings import OutcomeModelConfig


@dataclass
class OutcomePrediction:
    """Container for outcome prediction from a single expert."""
    regime: str = 'unknown'
    
    # Raw prediction probability
    probability: float = 0.5
    
    # Calibrated probability (if calibration enabled)
    calibrated_probability: Optional[float] = None
    
    # Model confidence
    confidence: float = 0.0
    
    # Feature contribution (for interpretability)
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    
    def get_probability(self, use_calibrated: bool = True) -> float:
        """Get prediction probability, preferring calibrated if available."""
        if use_calibrated and self.calibrated_probability is not None:
            return self.calibrated_probability
        return self.probability
    
    def to_dict(self) -> Dict:
        """Convert prediction to dictionary."""
        return {
            'regime': self.regime,
            'probability': self.probability,
            'calibrated_probability': self.calibrated_probability,
            'confidence': self.confidence,
            'feature_contributions': self.feature_contributions,
        }


class OutcomeExpertModels:
    """
    Regime-conditional outcome experts using logistic regression.
    
    One expert per regime: Trend, Range, Shock
    Training uses soft weights = Pr(Z=k) for regime k
    Supports probability calibration
    """
    
    def __init__(self, config: Optional[OutcomeModelConfig] = None):
        self.config = config or OutcomeModelConfig()
        
        # Expert models for each regime
        self._experts: Dict[str, Optional[LogisticRegression]] = {
            'trend': None,
            'range': None,
            'shock': None
        }
        
        # Calibration models for each regime
        self._calibrators: Dict[str, Optional[IsotonicRegression]] = {
            'trend': None,
            'range': None,
            'shock': None
        }
        
        # Feature names
        self._feature_names: List[str] = []
        
        # Training state
        self._is_fitted: Dict[str, bool] = {
            'trend': False,
            'range': False,
            'shock': False
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            regime_probs: np.ndarray,
            feature_names: Optional[List[str]] = None):
        """
        Fit all regime-conditional experts.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary outcomes (n_samples,)
            regime_probs: Soft regime probabilities (n_samples, 3) 
                         [trend_prob, range_prob, shock_prob]
            feature_names: Optional feature names for interpretability
        """
        if feature_names is not None:
            self._feature_names = feature_names
        
        regimes = ['trend', 'range', 'shock']
        
        for i, regime in enumerate(regimes):
            # Extract soft weights for this regime
            weights = regime_probs[:, i]
            
            # Only train on samples with meaningful weight for this regime
            mask = weights > 0.1
            if np.sum(mask) < self.config.min_train_samples:
                print(f"Warning: Not enough samples for {regime} regime ({np.sum(mask)} < {self.config.min_train_samples})")
                continue
            
            X_regime = X[mask]
            y_regime = y[mask]
            w_regime = weights[mask]
            
            # Fit logistic regression with sample weights
            expert = LogisticRegression(
                max_iter=self.config.max_iter,
                class_weight='balanced',
                solver='lbfgs'
            )
            expert.fit(X_regime, y_regime, sample_weight=w_regime)
            
            self._experts[regime] = expert
            self._is_fitted[regime] = True
            
            # Fit calibration model if enabled
            if self.config.calibration_method is not None:
                self._fit_calibration(regime, X_regime, y_regime, w_regime)
        
        return self
    
    def _fit_calibration(self, regime: str, X: np.ndarray, y: np.ndarray, 
                         sample_weight: np.ndarray):
        """Fit calibration model for a regime expert."""
        expert = self._experts[regime]
        if expert is None:
            return
        
        # Get uncalibrated predictions
        probs = expert.predict_proba(X)[:, 1]
        
        if self.config.calibration_method == 'isotonic':
            calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            calibrator.fit(probs, y, sample_weight=sample_weight)
            self._calibrators[regime] = calibrator
        
        elif self.config.calibration_method == 'platt':
            # Platt scaling (sigmoid calibration)
            # sklearn's _sigmoid_calibration returns (a, b) for 1/(1 + exp(a*x + b))
            ab = _sigmoid_calibration(probs, y, sample_weight=sample_weight)
            self._platt_params[regime] = ab
    
    def predict(self, X_t: np.ndarray, regime: str) -> OutcomePrediction:
        """
        Predict outcome probability for a specific regime.
        
        Args:
            X_t: Feature vector
            regime: Regime label ('trend', 'range', 'shock')
        
        Returns:
            OutcomePrediction
        """
        prediction = OutcomePrediction(regime=regime)
        
        expert = self._experts.get(regime)
        if expert is None or not self._is_fitted.get(regime, False):
            # Return neutral prediction if expert not available
            prediction.probability = 0.5
            prediction.confidence = 0.0
            return prediction
        
        # Reshape for sklearn
        X_reshaped = X_t.reshape(1, -1)
        
        # Get probability
        prob = expert.predict_proba(X_reshaped)[0, 1]
        prediction.probability = prob
        
        # Get confidence (distance from 0.5)
        prediction.confidence = abs(prob - 0.5) * 2
        
        # Apply calibration if available
        calibrator = self._calibrators.get(regime)
        if calibrator is not None:
            calibrated = calibrator.predict([prob])[0]
            prediction.calibrated_probability = float(calibrated)
        
        # Compute feature contributions (simplified linear approximation)
        prediction.feature_contributions = self._compute_feature_contributions(
            X_t, expert
        )
        
        return prediction
    
    def predict_all(self, X_t: np.ndarray) -> Dict[str, OutcomePrediction]:
        """
        Predict outcome probabilities for all regimes.
        
        Args:
            X_t: Feature vector
        
        Returns:
            Dictionary mapping regime to OutcomePrediction
        """
        regimes = ['trend', 'range', 'shock']
        predictions = {}
        
        for regime in regimes:
            predictions[regime] = self.predict(X_t, regime)
        
        return predictions
    
    def _compute_feature_contributions(self, X_t: np.ndarray, 
                                       expert: LogisticRegression) -> Dict[str, float]:
        """
        Compute approximate feature contributions using log-odds.
        
        Contribution_i = beta_i * x_i
        """
        if len(self._feature_names) == 0:
            return {}
        
        # Get coefficients
        coefs = expert.coef_[0]
        intercept = expert.intercept_[0]
        
        # Compute contributions
        contributions = {}
        for i, name in enumerate(self._feature_names):
            if i < len(coefs):
                contributions[name] = float(coefs[i] * X_t[i])
        
        # Add intercept
        contributions['intercept'] = float(intercept)
        
        return contributions
    
    def get_feature_importance(self, regime: str) -> Dict[str, float]:
        """Get feature importance for a regime expert."""
        expert = self._experts.get(regime)
        if expert is None:
            return {}
        
        coefs = expert.coef_[0]
        importance = {name: abs(coefs[i]) for i, name in enumerate(self._feature_names) 
                     if i < len(coefs)}
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def save(self, path: str):
        """Save models to disk."""
        data = {
            'experts': self._experts,
            'calibrators': self._calibrators,
            'is_fitted': self._is_fitted,
            'feature_names': self._feature_names,
            'config': self.config
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load models from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self._experts = data['experts']
        self._calibrators = data['calibrators']
        self._is_fitted = data['is_fitted']
        self._feature_names = data['feature_names']
        self.config = data['config']


# Platt scaling parameters storage
OutcomeExpertModels._platt_params = {}
