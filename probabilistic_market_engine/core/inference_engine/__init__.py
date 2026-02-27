"""
Layer 6: Mixture Aggregation Inference Engine

Final prediction: Pr(Y=1) = Σ Pr(Y=1 | X, Z=k) × Pr(Z=k)
Deterministic and reproducible.
Only prediction exposed to API.
"""

from .engine import InferenceEngine, InferenceResult

__all__ = ['InferenceEngine', 'InferenceResult']
