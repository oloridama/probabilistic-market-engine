"""
Layer 3: Feature Pipeline

Prepares features safely for modeling.
Responsibilities: rolling window isolation, standardization, no lookahead bias.
"""

from .pipeline import FeaturePipeline, FeatureSet

__all__ = ['FeaturePipeline', 'FeatureSet']
