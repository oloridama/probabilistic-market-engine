"""
Layer 6: Mixture Aggregation Inference Engine

Final prediction: Pr(Y=1) = Σ Pr(Y=1 | X, Z=k) × Pr(Z=k)

This is the only prediction exposed to API.
Must be deterministic and reproducible.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from probabilistic_market_engine.core.regime_model.model import RegimeState
from probabilistic_market_engine.core.outcome_models.experts import OutcomeExpertModels, OutcomePrediction


@dataclass
class InferenceResult:
    """Container for final inference result."""
    timestamp: Optional[pd.Timestamp] = None
    
    # Final aggregated probability
    prob_continuation: float = 0.5
    
    # Component predictions
    regime_predictions: Dict[str, OutcomePrediction] = field(default_factory=dict)
    
    # Regime probabilities
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Shock probability (exposed separately)
    shock_probability: float = 0.0
    
    # Model confidence
    prediction_confidence: float = 0.0
    regime_entropy: float = 1.0
    
    # Component breakdown
    mixture_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    model_version: str = 'unknown'
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            'timestamp': self.timestamp,
            'prob_continuation': self.prob_continuation,
            'regime_predictions': {k: v.to_dict() for k, v in self.regime_predictions.items()},
            'regime_probabilities': self.regime_probabilities,
            'shock_probability': self.shock_probability,
            'prediction_confidence': self.prediction_confidence,
            'regime_entropy': self.regime_entropy,
            'mixture_contributions': self.mixture_contributions,
            'model_version': self.model_version,
        }


class InferenceEngine:
    """
    Mixture aggregation inference engine.
    
    Computes final prediction as weighted average of expert predictions:
    Pr(Y=1) = Σ_k Pr(Y=1 | X, Z=k) × Pr(Z=k)
    
    This is the ONLY layer that should be exposed to the API.
    """
    
    def __init__(self, expert_models: OutcomeExpertModels):
        self.expert_models = expert_models
        self._model_version: str = 'v1.0.0'
    
    def set_model_version(self, version: str):
        """Set model version for tracking."""
        self._model_version = version
    
    def predict(self, X_t: np.ndarray, regime_state: RegimeState,
                timestamp: Optional[pd.Timestamp] = None) -> InferenceResult:
        """
        Compute final prediction using mixture aggregation.
        
        Args:
            X_t: Full feature vector
            regime_state: Current regime state with probabilities
            timestamp: Optional timestamp
        
        Returns:
            InferenceResult with final probability
        """
        result = InferenceResult(timestamp=timestamp)
        result.model_version = self._model_version
        
        # Get regime probabilities
        regime_probs = {
            'trend': regime_state.trend_probability,
            'range': regime_state.range_probability,
            'shock': regime_state.shock_probability
        }
        result.regime_probabilities = regime_probs
        result.shock_probability = regime_state.shock_probability
        result.regime_entropy = regime_state.regime_entropy
        
        # Get predictions from all experts
        expert_preds = self.expert_models.predict_all(X_t)
        result.regime_predictions = expert_preds
        
        # Compute mixture: weighted average
        total_prob = 0.0
        contributions = {}
        
        for regime, pred in expert_preds.items():
            weight = regime_probs.get(regime, 0.0)
            prob = pred.get_probability(use_calibrated=True)
            
            contribution = prob * weight
            contributions[regime] = contribution
            total_prob += contribution
        
        # Normalize (in case weights don't sum to 1 due to missing experts)
        total_weight = sum(regime_probs.values())
        if total_weight > 0:
            result.prob_continuation = total_prob / total_weight
        else:
            result.prob_continuation = 0.5
        
        result.mixture_contributions = contributions
        
        # Compute prediction confidence
        # Higher when: 
        # 1. Experts agree (low variance in predictions)
        # 2. Regime is certain (low entropy)
        probs = [pred.get_probability(use_calibrated=True) for pred in expert_preds.values()]
        prob_variance = np.var(probs) if len(probs) > 0 else 0
        agreement = 1 - 4 * prob_variance  # 1 when all agree, 0 when max disagreement
        
        regime_certainty = 1 - regime_state.regime_entropy
        
        result.prediction_confidence = (agreement + regime_certainty) / 2
        
        return result
    
    def predict_batch(self, X_features: np.ndarray, 
                     regime_states: List[RegimeState],
                     timestamps: Optional[List[pd.Timestamp]] = None) -> List[InferenceResult]:
        """
        Compute predictions for a batch.
        
        Args:
            X_features: Array of feature vectors (n_samples, n_features)
            regime_states: List of regime states
            timestamps: Optional list of timestamps
        
        Returns:
            List of InferenceResult objects
        """
        results = []
        
        for i, (X_t, regime_state) in enumerate(zip(X_features, regime_states)):
            ts = timestamps[i] if timestamps is not None else None
            result = self.predict(X_t, regime_state, ts)
            results.append(result)
        
        return results


def compute_mixture_prediction(expert_probs: Dict[str, float],
                               regime_probs: Dict[str, float]) -> float:
    """
    Standalone function for mixture computation.
    
    Args:
        expert_probs: Dict mapping regime to expert prediction probability
        regime_probs: Dict mapping regime to regime probability
    
    Returns:
        Aggregated probability
    """
    total = 0.0
    total_weight = 0.0
    
    for regime, prob in expert_probs.items():
        weight = regime_probs.get(regime, 0.0)
        total += prob * weight
        total_weight += weight
    
    if total_weight > 0:
        return total / total_weight
    return 0.5
