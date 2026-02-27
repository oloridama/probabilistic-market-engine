"""
Calibration Evaluation

Evaluates probability calibration using:
- Brier score
- Reliability diagram
- Log-loss
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import logging


@dataclass
class CalibrationResult:
    """Result of calibration evaluation."""
    
    # Scoring metrics
    brier_score: float = 0.0
    log_loss: float = 0.0
    
    # Reliability diagram data
    prob_true: np.ndarray = field(default_factory=lambda: np.array([]))
    prob_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Calibration error
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    
    # Perfect calibration would have ECE = 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'brier_score': self.brier_score,
            'log_loss': self.log_loss,
            'prob_true': self.prob_true.tolist(),
            'prob_pred': self.prob_pred.tolist(),
            'bin_counts': self.bin_counts.tolist(),
            'expected_calibration_error': self.expected_calibration_error,
            'maximum_calibration_error': self.maximum_calibration_error,
        }


class CalibrationEvaluator:
    """
    Evaluates probability calibration of predictions.
    
    A well-calibrated model should have:
    - Brier score close to 0
    - Reliability curve close to diagonal
    - Low Expected Calibration Error (ECE)
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self, y_true: np.ndarray, 
                 y_prob: np.ndarray) -> CalibrationResult:
        """
        Evaluate calibration of predictions.
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
        
        Returns:
            CalibrationResult with metrics and reliability diagram data
        """
        result = CalibrationResult()
        
        # Basic metrics
        result.brier_score = float(brier_score_loss(y_true, y_prob))
        
        # Clip for log loss
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        result.log_loss = float(log_loss(y_true, y_prob_clipped))
        
        # Reliability diagram
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=self.n_bins)
        result.prob_true = prob_true
        result.prob_pred = prob_pred
        
        # Compute bin counts
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        result.bin_counts = np.bincount(bin_indices, minlength=self.n_bins)
        
        # Compute Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(y_true)
        
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_acc = np.mean(y_true[mask])
                bin_conf = np.mean(y_prob[mask])
                bin_weight = np.sum(mask) / total_samples
                ece += bin_weight * abs(bin_acc - bin_conf)
        
        result.expected_calibration_error = ece
        
        # Compute Maximum Calibration Error (MCE)
        if len(prob_true) > 0:
            result.maximum_calibration_error = float(np.max(np.abs(prob_true - prob_pred)))
        
        return result
    
    def evaluate_by_regime(self, y_true: np.ndarray,
                          y_prob: np.ndarray,
                          regime_probs: np.ndarray,
                          regime_labels: List[str] = None) -> Dict[str, CalibrationResult]:
        """
        Evaluate calibration separately for each dominant regime.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            regime_probs: Regime probabilities (n_samples, n_regimes)
            regime_labels: List of regime names
        
        Returns:
            Dict mapping regime to CalibrationResult
        """
        if regime_labels is None:
            regime_labels = ['trend', 'range', 'shock']
        
        results = {}
        
        # Get dominant regime for each sample
        dominant_regimes = np.argmax(regime_probs, axis=1)
        
        for i, regime in enumerate(regime_labels):
            mask = dominant_regimes == i
            if np.sum(mask) > 50:  # Need enough samples
                regime_y_true = y_true[mask]
                regime_y_prob = y_prob[mask]
                results[regime] = self.evaluate(regime_y_true, regime_y_prob)
            else:
                self.logger.warning(f"Not enough samples for {regime} regime calibration")
        
        return results
    
    def is_well_calibrated(self, result: CalibrationResult,
                          brier_threshold: float = 0.25,
                          ece_threshold: float = 0.1) -> bool:
        """
        Check if model is well-calibrated.
        
        Args:
            result: CalibrationResult
            brier_threshold: Maximum acceptable Brier score
            ece_threshold: Maximum acceptable ECE
        
        Returns:
            True if well-calibrated
        """
        return (result.brier_score <= brier_threshold and 
                result.expected_calibration_error <= ece_threshold)
    
    def print_report(self, result: CalibrationResult):
        """Print calibration report."""
        self.logger.info("=== Calibration Report ===")
        self.logger.info(f"Brier Score: {result.brier_score:.4f}")
        self.logger.info(f"Log Loss: {result.log_loss:.4f}")
        self.logger.info(f"Expected Calibration Error: {result.expected_calibration_error:.4f}")
        self.logger.info(f"Maximum Calibration Error: {result.maximum_calibration_error:.4f}")
        
        self.logger.info("\nReliability Diagram:")
        self.logger.info("Predicted | Actual | Count")
        self.logger.info("-" * 30)
        for pred, true, count in zip(result.prob_pred, result.prob_true, result.bin_counts):
            self.logger.info(f"{pred:8.3f} | {true:6.3f} | {count:5d}")


def compute_reliability_diagram(y_true: np.ndarray, 
                                 y_prob: np.ndarray,
                                 n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability diagram data.
    
    Returns:
        (bin_centers, bin_accuracies, bin_counts)
    """
    evaluator = CalibrationEvaluator(n_bins=n_bins)
    result = evaluator.evaluate(y_true, y_prob)
    
    return result.prob_pred, result.prob_true, result.bin_counts
