"""
Drift Detection Module

Detects:
- Feature mean/variance drift
- Regime distribution drift
- Prediction distribution drift

Alerts are programmatically detectable.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats
from collections import deque
import logging

from probabilistic_market_engine.config.settings import MonitoringConfig


@dataclass
class DriftAlert:
    """Drift alert container."""
    
    alert_type: str  # 'feature_drift', 'regime_drift', 'prediction_drift'
    severity: str  # 'low', 'medium', 'high'
    timestamp: Optional[pd.Timestamp] = None
    
    # Details
    feature_name: Optional[str] = None
    drift_score: float = 0.0
    threshold: float = 0.0
    description: str = ""
    
    def is_critical(self) -> bool:
        """Check if alert is critical."""
        return self.severity == 'high'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'alert_type': self.alert_type,
            'severity': self.severity,
            'timestamp': str(self.timestamp) if self.timestamp else None,
            'feature_name': self.feature_name,
            'drift_score': self.drift_score,
            'threshold': self.threshold,
            'description': self.description,
        }


class DriftDetector:
    """
    Detects various types of drift in the system.
    
    Drift types:
    - Feature drift: Changes in feature distributions
    - Regime drift: Changes in regime frequencies
    - Prediction drift: Changes in prediction distributions
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Buffers for tracking
        self._feature_buffer: deque = deque(maxlen=self.config.drift_window_long)
        self._regime_buffer: deque = deque(maxlen=self.config.drift_window_long)
        self._prediction_buffer: deque = deque(maxlen=self.config.drift_window_long)
        
        # Reference statistics (from training)
        self._reference_feature_stats: Optional[Dict] = None
        self._reference_regime_dist: Optional[np.ndarray] = None
        
        # Alert cooldown
        self._last_alert_period: Dict[str, int] = {}
        self._current_period: int = 0
    
    def set_reference(self, features: np.ndarray, 
                     regime_probs: np.ndarray):
        """
        Set reference statistics from training data.
        
        Args:
            features: Reference feature matrix
            regime_probs: Reference regime probabilities
        """
        # Feature statistics
        self._reference_feature_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0) + 1e-10,
        }
        
        # Regime distribution
        self._reference_regime_dist = np.mean(regime_probs, axis=0)
        
        self.logger.info("Reference statistics set for drift detection")
    
    def update(self, features: np.ndarray,
               regime_probs: np.ndarray,
               prediction: float,
               timestamp: Optional[pd.Timestamp] = None) -> List[DriftAlert]:
        """
        Update drift detector with new observation.
        
        Args:
            features: Feature vector
            regime_probs: Regime probability vector
            prediction: Final prediction probability
            timestamp: Optional timestamp
        
        Returns:
            List of DriftAlert objects (empty if no drift detected)
        """
        self._current_period += 1
        
        # Add to buffers
        self._feature_buffer.append(features)
        self._regime_buffer.append(regime_probs)
        self._prediction_buffer.append(prediction)
        
        # Check for drift
        alerts = []
        
        # Feature drift
        if len(self._feature_buffer) >= self.config.drift_window_short:
            feature_alerts = self._check_feature_drift(timestamp)
            alerts.extend(feature_alerts)
        
        # Regime drift
        if len(self._regime_buffer) >= self.config.drift_window_short:
            regime_alert = self._check_regime_drift(timestamp)
            if regime_alert:
                alerts.append(regime_alert)
        
        # Prediction drift
        if len(self._prediction_buffer) >= self.config.drift_window_short:
            pred_alert = self._check_prediction_drift(timestamp)
            if pred_alert:
                alerts.append(pred_alert)
        
        return alerts
    
    def _check_feature_drift(self, timestamp: Optional[pd.Timestamp]) -> List[DriftAlert]:
        """Check for feature drift using z-scores."""
        alerts = []
        
        if self._reference_feature_stats is None:
            return alerts
        
        # Get recent features
        recent = np.array(list(self._feature_buffer)[-self.config.drift_window_short:])
        recent_mean = np.mean(recent, axis=0)
        
        # Compute z-scores
        ref_mean = self._reference_feature_stats['mean']
        ref_std = self._reference_feature_stats['std']
        
        z_scores = np.abs((recent_mean - ref_mean) / ref_std)
        
        # Check for significant drift
        for i, z in enumerate(z_scores):
            if z > self.config.feature_drift_threshold:
                # Check cooldown
                alert_key = f'feature_{i}'
                if self._can_alert(alert_key):
                    severity = 'high' if z > self.config.feature_drift_threshold * 2 else 'medium'
                    alert = DriftAlert(
                        alert_type='feature_drift',
                        severity=severity,
                        timestamp=timestamp,
                        feature_name=f'feature_{i}',
                        drift_score=float(z),
                        threshold=self.config.feature_drift_threshold,
                        description=f'Feature {i} z-score: {z:.2f}'
                    )
                    alerts.append(alert)
                    self._record_alert(alert_key)
        
        return alerts
    
    def _check_regime_drift(self, timestamp: Optional[pd.Timestamp]) -> Optional[DriftAlert]:
        """Check for regime distribution drift."""
        if self._reference_regime_dist is None:
            return None
        
        # Get recent regime distribution
        recent = np.array(list(self._regime_buffer)[-self.config.drift_window_short:])
        recent_dist = np.mean(recent, axis=0)
        
        # Compute KL divergence
        kl_div = self._kl_divergence(self._reference_regime_dist, recent_dist)
        
        # Threshold for regime drift
        threshold = 0.1
        
        if kl_div > threshold and self._can_alert('regime_drift'):
            self._record_alert('regime_drift')
            return DriftAlert(
                alert_type='regime_drift',
                severity='high' if kl_div > 0.2 else 'medium',
                timestamp=timestamp,
                drift_score=float(kl_div),
                threshold=threshold,
                description=f'Regime distribution KL divergence: {kl_div:.3f}'
            )
        
        return None
    
    def _check_prediction_drift(self, timestamp: Optional[pd.Timestamp]) -> Optional[DriftAlert]:
        """Check for prediction distribution drift."""
        if len(self._prediction_buffer) < self.config.drift_window_long:
            return None
        
        # Compare short-term vs long-term prediction distributions
        recent = list(self._prediction_buffer)[-self.config.drift_window_short:]
        historical = list(self._prediction_buffer)[-self.config.drift_window_long:]
        
        # KS test
        try:
            ks_stat, p_value = stats.ks_2samp(recent, historical)
            
            # Threshold for significant drift
            if ks_stat > 0.2 and p_value < 0.05 and self._can_alert('prediction_drift'):
                self._record_alert('prediction_drift')
                return DriftAlert(
                    alert_type='prediction_drift',
                    severity='medium',
                    timestamp=timestamp,
                    drift_score=float(ks_stat),
                    threshold=0.2,
                    description=f'Prediction distribution KS statistic: {ks_stat:.3f}'
                )
        except Exception:
            pass
        
        return None
    
    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions."""
        # Add small epsilon to avoid log(0)
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        
        return np.sum(p * np.log(p / q))
    
    def _can_alert(self, alert_key: str) -> bool:
        """Check if alert can be issued (cooldown check)."""
        last_period = self._last_alert_period.get(alert_key, -self.config.alert_cooldown_periods)
        return (self._current_period - last_period) >= self.config.alert_cooldown_periods
    
    def _record_alert(self, alert_key: str):
        """Record that an alert was issued."""
        self._last_alert_period[alert_key] = self._current_period
    
    def get_feature_statistics(self) -> Dict:
        """Get current feature statistics."""
        if len(self._feature_buffer) < self.config.drift_window_short:
            return {}
        
        recent = np.array(list(self._feature_buffer)[-self.config.drift_window_short:])
        
        return {
            'mean': recent.mean(axis=0).tolist(),
            'std': recent.std(axis=0).tolist(),
            'min': recent.min(axis=0).tolist(),
            'max': recent.max(axis=0).tolist(),
        }
    
    def get_regime_statistics(self) -> Dict:
        """Get current regime statistics."""
        if len(self._regime_buffer) < self.config.drift_window_short:
            return {}
        
        recent = np.array(list(self._regime_buffer)[-self.config.drift_window_short:])
        
        return {
            'mean_distribution': recent.mean(axis=0).tolist(),
            'entropy': float(-np.sum(recent.mean(axis=0) * np.log(recent.mean(axis=0) + 1e-10))),
        }


def detect_concept_drift(reference_data: np.ndarray,
                        current_data: np.ndarray,
                        method: str = 'ks') -> Tuple[float, float]:
    """
    Standalone function for concept drift detection.
    
    Args:
        reference_data: Reference distribution
        current_data: Current distribution
        method: 'ks' for Kolmogorov-Smirnov, 'psi' for Population Stability Index
    
    Returns:
        (drift_score, p_value)
    """
    if method == 'ks':
        ks_stat, p_value = stats.ks_2samp(reference_data, current_data)
        return ks_stat, p_value
    
    elif method == 'psi':
        # Population Stability Index
        # Bin the data
        combined = np.concatenate([reference_data, current_data])
        bins = np.percentile(combined, np.linspace(0, 100, 11))
        bins[0] -= 1e-10  # Ensure all data is included
        bins[-1] += 1e-10
        
        ref_hist, _ = np.histogram(reference_data, bins=bins)
        curr_hist, _ = np.histogram(current_data, bins=bins)
        
        # Convert to probabilities
        ref_pct = ref_hist / len(reference_data) + 1e-10
        curr_pct = curr_hist / len(current_data) + 1e-10
        
        # PSI formula
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return psi, 0.0  # No p-value for PSI
    
    else:
        raise ValueError(f"Unknown method: {method}")
