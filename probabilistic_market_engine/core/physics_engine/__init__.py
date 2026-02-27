"""
Layer 1: Deterministic Physics Engine

Transforms OHLCV data into structured state features.
NO ML. NO probability modeling. Pure deterministic transformations.
"""

from .engine import PhysicsEngine, PhysicsState

__all__ = ['PhysicsEngine', 'PhysicsState']
