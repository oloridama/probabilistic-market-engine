"""
Layer 4: Bayesian Regime Inference

Gaussian Mixture Model for latent regime classification.
Input: R_t subset only
Output: Soft posterior probabilities [Pr(Trend), Pr(Range), Pr(Shock)]
"""

from .model import RegimeInferenceModel, RegimeState

__all__ = ['RegimeInferenceModel', 'RegimeState']
