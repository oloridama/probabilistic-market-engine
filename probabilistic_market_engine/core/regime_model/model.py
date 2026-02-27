"""
Layer 4: Bayesian Regime Inference

Gaussian Mixture Model for latent regime classification.
Input: R_t subset only: [S_t, alignment_score, path_efficiency, volatility_ratio, shock_index]
Output: Regime probability vector: [Pr(Trend), Pr(Range), Pr(Shock)]

This module must never access outcome labels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
from scipy.special import softmax
import pickle

from probabilistic_market_engine.config.settings import RegimeModelConfig


@dataclass
class RegimeState:
    """Container for regime inference output."""
    timestamp: Optional[pd.Timestamp] = None
    
    # Raw posterior probabilities from GMM
    raw_posterior: np.ndarray = field(default_factory=lambda: np.ones(3) / 3)
    
    # Smoothed probabilities with persistence
    smoothed_probs: np.ndarray = field(default_factory=lambda: np.ones(3) / 3)
    
    # Labeled probabilities
    trend_probability: float = 0.33
    range_probability: float = 0.33
    shock_probability: float = 0.34
    
    # Regime uncertainty (entropy)
    regime_entropy: float = 1.0
    
    # Most likely regime
    dominant_regime: str = 'unknown'
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert regime state to dictionary."""
        return {
            'timestamp': self.timestamp,
            'raw_posterior': self.raw_posterior.tolist(),
            'smoothed_probs': self.smoothed_probs.tolist(),
            'trend_probability': self.trend_probability,
            'range_probability': self.range_probability,
            'shock_probability': self.shock_probability,
            'regime_entropy': self.regime_entropy,
            'dominant_regime': self.dominant_regime,
            'confidence': self.confidence,
        }


class RegimeInferenceModel:
    """
    Bayesian regime inference using Gaussian Mixture Model.
    
    Key constraints:
    - Uses only R_t features (regime subset)
    - No outcome labels used
    - Applies persistence smoothing to regime probabilities
    - K=3 regimes: Trend, Range, Shock
    """
    
    def __init__(self, config: Optional[RegimeModelConfig] = None):
        self.config = config or RegimeModelConfig()
        
        # GMM model
        self._gmm: Optional[GaussianMixture] = None
        self._is_fitted: bool = False
        
        # Regime persistence state
        self._prev_probs: Optional[np.ndarray] = None
        
        # Regime label mapping (determined during fitting)
        self._regime_labels: List[str] = self.config.regime_labels
        self._regime_mapping: Dict[int, str] = {}
    
    def fit(self, R_features: np.ndarray) -> 'RegimeInferenceModel':
        """
        Fit GMM on regime features.
        
        Args:
            R_features: Array of shape (n_samples, n_regime_features)
        
        Returns:
            self for method chaining
        """
        if len(R_features) < self.config.n_regimes * 10:
            raise ValueError(f"Need at least {self.config.n_regimes * 10} samples for GMM fitting")
        
        # Standardize features
        self._feature_means = np.mean(R_features, axis=0)
        self._feature_stds = np.std(R_features, axis=0) + 1e-10
        R_std = (R_features - self._feature_means) / self._feature_stds
        
        # Fit GMM
        self._gmm = GaussianMixture(
            n_components=self.config.n_regimes,
            random_state=self.config.random_state,
            max_iter=self.config.max_iter,
            n_init=self.config.n_init,
            covariance_type='full'
        )
        
        self._gmm.fit(R_std)
        self._is_fitted = True
        
        # Determine regime labels based on component characteristics
        self._determine_regime_labels(R_std)
        
        # Reset persistence
        self._prev_probs = None
        
        return self
    
    def predict(self, R_t: np.ndarray, timestamp: Optional[pd.Timestamp] = None) -> RegimeState:
        """
        Predict regime probabilities for a single observation.
        
        Args:
            R_t: Regime feature vector
            timestamp: Optional timestamp for tracking
        
        Returns:
            RegimeState with probabilities
        """
        state = RegimeState(timestamp=timestamp)
        
        if not self._is_fitted or self._gmm is None:
            # Return uniform distribution if not fitted
            uniform = np.ones(self.config.n_regimes) / self.config.n_regimes
            state.raw_posterior = uniform
            state.smoothed_probs = uniform
            self._update_state_from_probs(state, uniform)
            return state
        
        # Standardize input
        R_std = (R_t - self._feature_means) / self._feature_stds
        R_std = R_std.reshape(1, -1)
        
        # Get posterior probabilities
        posterior = self._gmm.predict_proba(R_std)[0]
        state.raw_posterior = posterior.copy()
        
        # Apply persistence smoothing
        smoothed = self._apply_persistence(posterior)
        state.smoothed_probs = smoothed.copy()
        
        # Update labeled probabilities
        self._update_state_from_probs(state, smoothed)
        
        return state
    
    def predict_batch(self, R_features: np.ndarray, 
                     timestamps: Optional[List[pd.Timestamp]] = None) -> List[RegimeState]:
        """
        Predict regime probabilities for batch of observations.
        
        Args:
            R_features: Array of shape (n_samples, n_regime_features)
            timestamps: Optional list of timestamps
        
        Returns:
            List of RegimeState objects
        """
        states = []
        
        for i, R_t in enumerate(R_features):
            ts = timestamps[i] if timestamps is not None else None
            state = self.predict(R_t, ts)
            states.append(state)
        
        return states
    
    def _apply_persistence(self, posterior: np.ndarray) -> np.ndarray:
        """
        Apply regime persistence smoothing.
        
        Pr(Z_t) = alpha * Pr(Z_{t-1}) + (1 - alpha) * posterior
        """
        alpha = self.config.persistence_alpha
        
        if self._prev_probs is None:
            # First prediction
            smoothed = posterior
        else:
            # Apply smoothing
            smoothed = alpha * self._prev_probs + (1 - alpha) * posterior
        
        # Normalize to ensure probabilities sum to 1
        smoothed = smoothed / (smoothed.sum() + 1e-10)
        
        # Store for next prediction
        self._prev_probs = smoothed.copy()
        
        return smoothed
    
    def _determine_regime_labels(self, R_std: np.ndarray):
        """
        Determine which GMM component corresponds to which regime label.
        
        Uses heuristic based on component characteristics:
        - Shock: high volatility, high shock index
        - Trend: high alignment, high path efficiency
        - Range: low alignment, low volatility
        """
        if self._gmm is None:
            return
        
        n_components = self.config.n_regimes
        labels = self.config.regime_labels
        
        # Compute component statistics
        component_scores = {}
        
        for k in range(n_components):
            # Get samples assigned to this component
            probs = self._gmm.predict_proba(R_std)[:, k]
            weighted_samples = R_std * probs[:, np.newaxis]
            
            # Compute weighted means
            mean_features = np.sum(weighted_samples, axis=0) / (np.sum(probs) + 1e-10)
            
            # Heuristic scoring (assuming features are in order):
            # [pressure_norm, alignment_score, path_efficiency, volatility_ratio, shock_index]
            
            shock_score = mean_features[3] + mean_features[4]  # volatility + shock
            trend_score = mean_features[1] + mean_features[2]   # alignment + efficiency
            range_score = -mean_features[3] - abs(mean_features[1])  # low volatility, neutral alignment
            
            component_scores[k] = {
                'shock': shock_score,
                'trend': trend_score,
                'range': range_score
            }
        
        # Assign labels greedily
        assigned = set()
        label_to_component = {}
        
        # First assign shock (highest shock score)
        shock_comp = max(component_scores.keys(), 
                        key=lambda k: component_scores[k]['shock'])
        label_to_component['shock'] = shock_comp
        assigned.add(shock_comp)
        
        # Then assign trend (highest trend score among remaining)
        remaining = [k for k in component_scores.keys() if k not in assigned]
        trend_comp = max(remaining, key=lambda k: component_scores[k]['trend'])
        label_to_component['trend'] = trend_comp
        assigned.add(trend_comp)
        
        # Remaining is range
        remaining = [k for k in component_scores.keys() if k not in assigned]
        if remaining:
            label_to_component['range'] = remaining[0]
        
        # Create mapping from component index to label
        self._regime_mapping = {v: k for k, v in label_to_component.items()}
        
        # Store ordered labels
        self._ordered_labels = [self._regime_mapping.get(k, f'component_{k}') 
                               for k in range(n_components)]
    
    def _update_state_from_probs(self, state: RegimeState, probs: np.ndarray):
        """Update state object from probability vector."""
        # Map probabilities to labeled regimes
        for i, label in enumerate(self._ordered_labels):
            if label == 'trend':
                state.trend_probability = probs[i]
            elif label == 'range':
                state.range_probability = probs[i]
            elif label == 'shock':
                state.shock_probability = probs[i]
        
        # Compute entropy (uncertainty)
        # Higher entropy = more uncertain
        probs_safe = probs + 1e-10
        entropy = -np.sum(probs_safe * np.log(probs_safe))
        state.regime_entropy = entropy / np.log(self.config.n_regimes)  # Normalize to [0, 1]
        
        # Determine dominant regime
        max_idx = np.argmax(probs)
        state.dominant_regime = self._ordered_labels[max_idx]
        state.confidence = probs[max_idx]
    
    def reset_persistence(self):
        """Reset persistence state (e.g., for new trading day)."""
        self._prev_probs = None
    
    def save(self, path: str):
        """Save model to disk."""
        data = {
            'gmm': self._gmm,
            'feature_means': getattr(self, '_feature_means', None),
            'feature_stds': getattr(self, '_feature_stds', None),
            'regime_mapping': self._regime_mapping,
            'ordered_labels': getattr(self, '_ordered_labels', self._regime_labels),
            'config': self.config,
            'is_fitted': self._is_fitted
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self._gmm = data['gmm']
        self._feature_means = data['feature_means']
        self._feature_stds = data['feature_stds']
        self._regime_mapping = data['regime_mapping']
        self._ordered_labels = data['ordered_labels']
        self.config = data['config']
        self._is_fitted = data['is_fitted']
        
        # Reset persistence on load
        self._prev_probs = None
