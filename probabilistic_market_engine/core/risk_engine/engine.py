"""
Layer 7: Risk Adjustment Engine

Risk scaling depends on:
- Final continuation probability
- Shock probability
- Regime uncertainty (entropy of regime distribution)

Formula: Risk_scale ∝ (Pr(Y) - 0.5) × (1 - Pr(Shock))
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

from probabilistic_market_engine.config.settings import RiskEngineConfig
from probabilistic_market_engine.core.inference_engine.engine import InferenceResult


@dataclass
class RiskAdjustment:
    """Container for risk adjustment output."""
    
    # Final risk scaling factor
    risk_scaling_factor: float = 0.0
    
    # Component factors
    signal_strength_factor: float = 0.0
    shock_penalty_factor: float = 1.0
    uncertainty_penalty_factor: float = 1.0
    
    # Position direction (based on probability)
    suggested_direction: int = 0  # -1, 0, 1
    
    # Confidence metrics
    signal_confidence: float = 0.0
    execution_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert risk adjustment to dictionary."""
        return {
            'risk_scaling_factor': self.risk_scaling_factor,
            'signal_strength_factor': self.signal_strength_factor,
            'shock_penalty_factor': self.shock_penalty_factor,
            'uncertainty_penalty_factor': self.uncertainty_penalty_factor,
            'suggested_direction': self.suggested_direction,
            'signal_confidence': self.signal_confidence,
            'execution_confidence': self.execution_confidence,
        }


class RiskEngine:
    """
    Risk adjustment engine for position sizing.
    
    Computes risk scaling factor based on:
    1. Signal strength: |Pr(Y) - 0.5| - how far from neutral
    2. Shock penalty: (1 - Pr(Shock)) - reduce exposure in volatile periods
    3. Uncertainty penalty: (1 - entropy) - reduce when regime unclear
    
    Formula: scale = base × signal_strength × (1 - shock) × (1 - uncertainty)
    """
    
    def __init__(self, config: Optional[RiskEngineConfig] = None):
        self.config = config or RiskEngineConfig()
    
    def compute(self, inference_result: InferenceResult) -> RiskAdjustment:
        """
        Compute risk adjustment from inference result.
        
        Args:
            inference_result: Result from inference engine
        
        Returns:
            RiskAdjustment with scaling factors
        """
        adjustment = RiskAdjustment()
        
        # Extract probabilities
        prob = inference_result.prob_continuation
        shock_prob = inference_result.shock_probability
        entropy = inference_result.regime_entropy
        
        # 1. Signal strength factor
        # Scales from 0 at p=0.5 to 1 at p=0 or p=1
        signal_strength = abs(prob - 0.5) * 2
        adjustment.signal_strength_factor = signal_strength
        
        # 2. Shock penalty factor
        # Reduce exposure when shock probability is high
        shock_penalty = max(0, 1 - shock_prob * self.config.shock_penalty_factor)
        adjustment.shock_penalty_factor = shock_penalty
        
        # 3. Uncertainty penalty factor
        # Reduce exposure when regime is uncertain
        uncertainty_penalty = max(0, 1 - entropy * self.config.uncertainty_penalty_factor)
        adjustment.uncertainty_penalty_factor = uncertainty_penalty
        
        # Compute final scaling factor
        raw_scale = (
            self.config.base_risk_scale *
            signal_strength *
            shock_penalty *
            uncertainty_penalty
        )
        
        # Clip to limits
        adjustment.risk_scaling_factor = np.clip(
            raw_scale,
            self.config.min_risk_scale,
            self.config.max_risk_scale
        )
        
        # Determine direction
        if prob > 0.55:
            adjustment.suggested_direction = 1  # Long
        elif prob < 0.45:
            adjustment.suggested_direction = -1  # Short
        else:
            adjustment.suggested_direction = 0  # Neutral
        
        # Compute confidence metrics
        adjustment.signal_confidence = signal_strength
        adjustment.execution_confidence = (
            inference_result.prediction_confidence * shock_penalty
        )
        
        return adjustment
    
    def compute_from_probs(self, prob: float, shock_prob: float, 
                          entropy: float) -> RiskAdjustment:
        """
        Compute risk adjustment from raw probabilities.
        
        Args:
            prob: Continuation probability
            shock_prob: Shock regime probability
            entropy: Regime entropy
        
        Returns:
            RiskAdjustment
        """
        # Create dummy inference result
        dummy_result = InferenceResult(
            prob_continuation=prob,
            shock_probability=shock_prob,
            regime_entropy=entropy
        )
        
        return self.compute(dummy_result)
