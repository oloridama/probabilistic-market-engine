"""
Walk-Forward Validation Framework

Rolling training window, rolling forward test window.
No leakage guaranteed.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Tuple, Any
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from collections import defaultdict
import logging

from probabilistic_market_engine.config.settings import TrainingConfig


@dataclass
class WalkForwardResult:
    """Result of walk-forward validation."""
    
    # Predictions and actuals
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    
    # Window tracking
    train_windows: List[Tuple[int, int]] = field(default_factory=list)
    test_windows: List[Tuple[int, int]] = field(default_factory=list)
    
    # Metrics by window
    window_metrics: List[Dict] = field(default_factory=list)
    
    # Overall metrics
    overall_brier: float = 0.0
    overall_logloss: float = 0.0
    overall_auc: float = 0.5
    
    # Regime-specific metrics
    regime_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'predictions': self.predictions.tolist(),
            'actuals': self.actuals.tolist(),
            'timestamps': [str(ts) for ts in self.timestamps],
            'n_windows': len(self.train_windows),
            'overall_brier': self.overall_brier,
            'overall_logloss': self.overall_logloss,
            'overall_auc': self.overall_auc,
            'regime_metrics': self.regime_metrics,
        }


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    Ensures no lookahead bias by strictly separating train and test periods.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.logger = logging.getLogger(__name__)
    
    def validate(self, 
                 data: pd.DataFrame,
                 train_fn: Callable,
                 predict_fn: Callable,
                 min_train_size: Optional[int] = None,
                 test_size: Optional[int] = None,
                 step_size: Optional[int] = None) -> WalkForwardResult:
        """
        Run walk-forward validation.
        
        Args:
            data: DataFrame with features and target
            train_fn: Function(train_data) -> model that trains on data
            predict_fn: Function(model, test_data) -> predictions that makes predictions
            min_train_size: Minimum training samples (default from config)
            test_size: Test window size (default from config)
            step_size: Step size between windows (default from config)
        
        Returns:
            WalkForwardResult with all predictions and metrics
        """
        result = WalkForwardResult()
        
        # Get parameters
        min_train = min_train_size or self.config.walkforward_train_size
        test_sz = test_size or self.config.walkforward_test_size
        step_sz = step_size or self.config.walkforward_step_size
        
        n_samples = len(data)
        
        if n_samples < min_train + test_sz:
            raise ValueError(f"Not enough data ({n_samples}) for walk-forward validation")
        
        all_predictions = []
        all_actuals = []
        all_timestamps = []
        
        # Generate windows
        current_idx = min_train
        window_idx = 0
        
        while current_idx + test_sz <= n_samples:
            # Define windows
            train_start = max(0, current_idx - min_train)
            train_end = current_idx
            test_start = current_idx
            test_end = min(current_idx + test_sz, n_samples)
            
            # Store window info
            result.train_windows.append((train_start, train_end))
            result.test_windows.append((test_start, test_end))
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            self.logger.info(f"Window {window_idx}: Train [{train_start}:{train_end}], "
                           f"Test [{test_start}:{test_end}]")
            
            try:
                # Train model
                model = train_fn(train_data)
                
                # Make predictions
                predictions = predict_fn(model, test_data)
                
                # Store results
                all_predictions.extend(predictions.tolist())
                all_actuals.extend(test_data['target'].values.tolist())
                all_timestamps.extend(test_data.index.tolist())
                
                # Compute window metrics
                window_metrics = self._compute_window_metrics(
                    np.array(predictions),
                    test_data['target'].values
                )
                result.window_metrics.append(window_metrics)
                
            except Exception as e:
                self.logger.error(f"Error in window {window_idx}: {e}")
                # Fill with neutral predictions
                neutral_preds = [0.5] * len(test_data)
                all_predictions.extend(neutral_preds)
                all_actuals.extend(test_data['target'].values.tolist())
                all_timestamps.extend(test_data.index.tolist())
            
            # Move window
            current_idx += step_sz
            window_idx += 1
        
        # Store all results
        result.predictions = np.array(all_predictions)
        result.actuals = np.array(all_actuals)
        result.timestamps = all_timestamps
        
        # Compute overall metrics
        overall_metrics = self._compute_window_metrics(result.predictions, result.actuals)
        result.overall_brier = overall_metrics['brier_score']
        result.overall_logloss = overall_metrics['log_loss']
        result.overall_auc = overall_metrics['auc_roc']
        
        self.logger.info(f"Walk-forward validation complete: {window_idx} windows")
        self.logger.info(f"Overall Brier: {result.overall_brier:.4f}")
        self.logger.info(f"Overall LogLoss: {result.overall_logloss:.4f}")
        self.logger.info(f"Overall AUC: {result.overall_auc:.4f}")
        
        return result
    
    def _compute_window_metrics(self, predictions: np.ndarray, 
                                actuals: np.ndarray) -> Dict:
        """Compute metrics for a single window."""
        metrics = {}
        
        # Basic counts
        metrics['n_samples'] = len(predictions)
        metrics['positive_rate'] = float(np.mean(actuals))
        
        # Brier score (proper scoring rule)
        try:
            metrics['brier_score'] = float(brier_score_loss(actuals, predictions))
        except Exception:
            metrics['brier_score'] = 0.25  # Maximum for binary case
        
        # Log loss
        try:
            # Clip predictions to avoid log(0)
            clipped = np.clip(predictions, 1e-10, 1 - 1e-10)
            metrics['log_loss'] = float(log_loss(actuals, clipped))
        except Exception:
            metrics['log_loss'] = float('inf')
        
        # AUC-ROC
        try:
            if len(np.unique(actuals)) > 1:
                metrics['auc_roc'] = float(roc_auc_score(actuals, predictions))
            else:
                metrics['auc_roc'] = 0.5
        except Exception:
            metrics['auc_roc'] = 0.5
        
        # Calibration metrics
        metrics['mean_prediction'] = float(np.mean(predictions))
        metrics['prediction_std'] = float(np.std(predictions))
        
        return metrics
    
    def generate_windows(self, n_samples: int, 
                         min_train_size: Optional[int] = None,
                         test_size: Optional[int] = None,
                         step_size: Optional[int] = None) -> List[Tuple[slice, slice]]:
        """
        Generate walk-forward window slices without running validation.
        
        Returns:
            List of (train_slice, test_slice) tuples
        """
        min_train = min_train_size or self.config.walkforward_train_size
        test_sz = test_size or self.config.walkforward_test_size
        step_sz = step_size or self.config.walkforward_step_size
        
        windows = []
        current_idx = min_train
        
        while current_idx + test_sz <= n_samples:
            train_start = max(0, current_idx - min_train)
            train_end = current_idx
            test_start = current_idx
            test_end = min(current_idx + test_sz, n_samples)
            
            windows.append((
                slice(train_start, train_end),
                slice(test_start, test_end)
            ))
            
            current_idx += step_sz
        
        return windows
