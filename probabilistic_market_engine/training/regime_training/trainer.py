"""
Regime Model Training

Trains the Bayesian regime inference model (GMM) on regime features.
Includes diagnostics for regime stability and transition analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

from probabilistic_market_engine.config.settings import RegimeModelConfig
from probabilistic_market_engine.core.regime_model.model import RegimeInferenceModel, RegimeState
from probabilistic_market_engine.core.feature_pipeline.pipeline import FeaturePipeline


class RegimeTrainer:
    """
    Trainer for regime inference model.
    
    Handles:
    - GMM fitting on regime features
    - Regime diagnostics (duration, transitions, stability)
    - Walk-forward regime validation
    """
    
    def __init__(self, config: Optional[RegimeModelConfig] = None):
        self.config = config or RegimeModelConfig()
        self.logger = logging.getLogger(__name__)
    
    def train(self, R_features: np.ndarray, 
              timestamps: Optional[List[pd.Timestamp]] = None) -> RegimeInferenceModel:
        """
        Train regime inference model.
        
        Args:
            R_features: Regime feature matrix (n_samples, n_features)
            timestamps: Optional timestamps for diagnostics
        
        Returns:
            Trained RegimeInferenceModel
        """
        self.logger.info(f"Training regime model on {len(R_features)} samples")
        
        model = RegimeInferenceModel(self.config)
        model.fit(R_features)
        
        # Run diagnostics
        diagnostics = self.compute_diagnostics(model, R_features, timestamps)
        self._log_diagnostics(diagnostics)
        
        return model
    
    def compute_diagnostics(self, model: RegimeInferenceModel,
                           R_features: np.ndarray,
                           timestamps: Optional[List[pd.Timestamp]] = None) -> Dict:
        """
        Compute regime diagnostics.
        
        Returns:
            Dict with diagnostic metrics
        """
        # Get predictions for all samples
        states = model.predict_batch(R_features, timestamps)
        
        diagnostics = {}
        
        # 1. Regime distribution
        regime_counts = Counter([s.dominant_regime for s in states])
        total = len(states)
        diagnostics['regime_distribution'] = {
            regime: count / total 
            for regime, count in regime_counts.items()
        }
        
        # 2. Regime duration statistics
        durations = self._compute_regime_durations(states)
        diagnostics['regime_durations'] = {
            regime: {
                'mean': float(np.mean(durs)) if durs else 0,
                'std': float(np.std(durs)) if durs else 0,
                'min': int(np.min(durs)) if durs else 0,
                'max': int(np.max(durs)) if durs else 0,
                'median': float(np.median(durs)) if durs else 0,
            }
            for regime, durs in durations.items()
        }
        
        # 3. Transition frequency
        transitions = self._compute_transitions(states)
        diagnostics['transition_matrix'] = transitions
        
        # 4. Regime stability (average confidence)
        diagnostics['regime_stability'] = {
            'mean_confidence': float(np.mean([s.confidence for s in states])),
            'mean_entropy': float(np.mean([s.regime_entropy for s in states])),
        }
        
        # 5. Shock detection rate
        shock_probs = [s.shock_probability for s in states]
        diagnostics['shock_stats'] = {
            'mean_shock_prob': float(np.mean(shock_probs)),
            'shock_periods': int(np.sum(np.array(shock_probs) > 0.5)),
            'shock_rate': float(np.mean(np.array(shock_probs) > 0.5)),
        }
        
        return diagnostics
    
    def _compute_regime_durations(self, states: List[RegimeState]) -> Dict[str, List[int]]:
        """Compute duration statistics for each regime."""
        durations = defaultdict(list)
        
        if not states:
            return dict(durations)
        
        current_regime = states[0].dominant_regime
        current_duration = 1
        
        for state in states[1:]:
            if state.dominant_regime == current_regime:
                current_duration += 1
            else:
                durations[current_regime].append(current_duration)
                current_regime = state.dominant_regime
                current_duration = 1
        
        # Don't forget the last regime
        durations[current_regime].append(current_duration)
        
        return dict(durations)
    
    def _compute_transitions(self, states: List[RegimeState]) -> Dict[str, Dict[str, float]]:
        """Compute transition matrix between regimes."""
        regimes = self.config.regime_labels
        
        # Initialize transition counts
        transitions = {r: {r2: 0 for r2 in regimes} for r in regimes}
        regime_counts = {r: 0 for r in regimes}
        
        if len(states) < 2:
            return transitions
        
        for i in range(len(states) - 1):
            current = states[i].dominant_regime
            next_regime = states[i + 1].dominant_regime
            
            if current in transitions and next_regime in transitions[current]:
                transitions[current][next_regime] += 1
                regime_counts[current] += 1
        
        # Convert to probabilities
        for r in regimes:
            total = regime_counts[r]
            if total > 0:
                for r2 in regimes:
                    transitions[r][r2] = transitions[r][r2] / total
        
        return transitions
    
    def _log_diagnostics(self, diagnostics: Dict):
        """Log diagnostic results."""
        self.logger.info("=== Regime Model Diagnostics ===")
        
        self.logger.info("Regime Distribution:")
        for regime, pct in diagnostics['regime_distribution'].items():
            self.logger.info(f"  {regime}: {pct:.2%}")
        
        self.logger.info("\nRegime Durations (periods):")
        for regime, stats in diagnostics['regime_durations'].items():
            self.logger.info(f"  {regime}: mean={stats['mean']:.1f}, "
                           f"median={stats['median']:.1f}, max={stats['max']}")
        
        self.logger.info("\nTransition Matrix:")
        for from_regime, to_regimes in diagnostics['transition_matrix'].items():
            self.logger.info(f"  {from_regime}: {to_regimes}")
        
        self.logger.info("\nStability Metrics:")
        self.logger.info(f"  Mean Confidence: {diagnostics['regime_stability']['mean_confidence']:.3f}")
        self.logger.info(f"  Mean Entropy: {diagnostics['regime_stability']['mean_entropy']:.3f}")
        
        self.logger.info("\nShock Statistics:")
        self.logger.info(f"  Mean Shock Prob: {diagnostics['shock_stats']['mean_shock_prob']:.3f}")
        self.logger.info(f"  Shock Rate: {diagnostics['shock_stats']['shock_rate']:.2%}")


def train_regime_model(ohlcv_data: pd.DataFrame,
                       feature_pipeline: FeaturePipeline,
                       config: Optional[RegimeModelConfig] = None) -> Tuple[RegimeInferenceModel, Dict]:
    """
    Convenience function to train regime model from OHLCV data.
    
    Args:
        ohlcv_data: OHLCV DataFrame
        feature_pipeline: Fitted feature pipeline
        config: Optional regime model config
    
    Returns:
        (trained_model, diagnostics)
    """
    # Get feature sets
    feature_sets = feature_pipeline.fit_transform(ohlcv_data)
    
    # Extract regime features
    R_features = np.array([fs.R_t for fs in feature_sets])
    timestamps = [fs.timestamp for fs in feature_sets]
    
    # Train model
    trainer = RegimeTrainer(config)
    model = trainer.train(R_features, timestamps)
    
    # Get diagnostics
    diagnostics = trainer.compute_diagnostics(model, R_features, timestamps)
    
    return model, diagnostics
