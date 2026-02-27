"""
Calibration Tracking Module

Tracks rolling calibration metrics over time.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

from probabilistic_market_engine.config.settings import MonitoringConfig
from probabilistic_market_engine.training.calibration.evaluator import CalibrationEvaluator, CalibrationResult


class CalibrationTracker:
    """
    Tracks model calibration over time.
    
    Maintains rolling window of predictions and outcomes
    to compute calibration metrics.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Buffers
        self._predictions: deque = deque(maxlen=self.config.brier_score_window)
        self._outcomes: deque = deque(maxlen=self.config.brier_score_window)
        self._timestamps: deque = deque(maxlen=self.config.brier_score_window)
        
        # History of metrics
        self._brier_history: List[float] = []
        self._logloss_history: List[float] = []
        
        # Calibration evaluator
        self._evaluator = CalibrationEvaluator(n_bins=10)
    
    def update(self, prediction: float, outcome: int,
               timestamp: Optional[pd.Timestamp] = None):
        """
        Update tracker with new prediction and outcome.
        
        Args:
            prediction: Predicted probability
            outcome: Actual outcome (0 or 1)
            timestamp: Optional timestamp
        """
        self._predictions.append(prediction)
        self._outcomes.append(outcome)
        self._timestamps.append(timestamp)
        
        # Compute rolling metrics if enough data
        if len(self._predictions) >= self.config.drift_window_short:
            self._compute_rolling_metrics()
    
    def _compute_rolling_metrics(self):
        """Compute metrics for current window."""
        predictions = np.array(list(self._predictions))
        outcomes = np.array(list(self._outcomes))
        
        # Brier score
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(outcomes, predictions)
        self._brier_history.append(brier)
        
        # Log loss
        clipped = np.clip(predictions, 1e-10, 1 - 1e-10)
        from sklearn.metrics import log_loss
        ll = log_loss(outcomes, clipped)
        self._logloss_history.append(ll)
    
    def get_current_calibration(self) -> Optional[CalibrationResult]:
        """
        Get calibration metrics for current window.
        
        Returns:
            CalibrationResult or None if not enough data
        """
        if len(self._predictions) < self.config.drift_window_short:
            return None
        
        predictions = np.array(list(self._predictions))
        outcomes = np.array(list(self._outcomes))
        
        return self._evaluator.evaluate(outcomes, predictions)
    
    def get_rolling_brier_score(self, window: Optional[int] = None) -> float:
        """
        Get rolling Brier score.
        
        Args:
            window: Lookback window (default: brier_score_window)
        
        Returns:
            Brier score
        """
        window = window or self.config.brier_score_window
        
        if len(self._predictions) < window:
            return 0.25  # Max Brier score
        
        predictions = np.array(list(self._predictions)[-window:])
        outcomes = np.array(list(self._outcomes)[-window:])
        
        from sklearn.metrics import brier_score_loss
        return brier_score_loss(outcomes, predictions)
    
    def get_brier_trend(self, short_window: Optional[int] = None,
                       long_window: Optional[int] = None) -> float:
        """
        Get trend in Brier score (negative is good).
        
        Returns:
            Difference between short and long window Brier scores
        """
        short = short_window or self.config.drift_window_short
        long = long_window or self.config.brier_score_window
        
        if len(self._predictions) < long:
            return 0.0
        
        short_brier = self.get_rolling_brier_score(short)
        long_brier = self.get_rolling_brier_score(long)
        
        return short_brier - long_brier  # Positive means degrading
    
    def is_calibration_degrading(self, threshold: float = 0.05) -> bool:
        """
        Check if calibration is degrading.
        
        Args:
            threshold: Threshold for degradation alert
        
        Returns:
            True if calibration is degrading
        """
        trend = self.get_brier_trend()
        return trend > threshold
    
    def get_calibration_report(self) -> Dict:
        """Get calibration tracking report."""
        current = self.get_current_calibration()
        
        report = {
            'n_samples': len(self._predictions),
            'rolling_brier': self.get_rolling_brier_score(),
            'brier_trend': self.get_brier_trend(),
            'calibration_degrading': self.is_calibration_degrading(),
        }
        
        if current:
            report.update({
                'expected_calibration_error': current.expected_calibration_error,
                'maximum_calibration_error': current.maximum_calibration_error,
                'log_loss': current.log_loss,
            })
        
        # History
        if self._brier_history:
            report['brier_history_latest'] = self._brier_history[-10:]
        
        return report
    
    def reset(self):
        """Reset tracker state."""
        self._predictions.clear()
        self._outcomes.clear()
        self._timestamps.clear()
        self._brier_history.clear()
        self._logloss_history.clear()


def compute_rolling_calibration(predictions: np.ndarray,
                                outcomes: np.ndarray,
                                window: int = 100) -> List[CalibrationResult]:
    """
    Compute rolling calibration metrics.
    
    Args:
        predictions: Array of predictions
        outcomes: Array of outcomes
        window: Rolling window size
    
    Returns:
        List of CalibrationResult objects
    """
    evaluator = CalibrationEvaluator()
    results = []
    
    for i in range(window, len(predictions)):
        pred_window = predictions[i-window:i]
        out_window = outcomes[i-window:i]
        
        result = evaluator.evaluate(out_window, pred_window)
        results.append(result)
    
    return results
