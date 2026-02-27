"""
Layer 5: Regime-Conditional Outcome Experts

Logistic regression experts trained for each regime.
Pr(Y=1 | X, Z=k) using soft weights = Pr(Z=k)
"""

from .experts import OutcomeExpertModels, OutcomePrediction

__all__ = ['OutcomeExpertModels', 'OutcomePrediction']
