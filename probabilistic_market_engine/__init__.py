"""
Probabilistic Market Engine

A production-grade nonlinear probabilistic state-space trading engine
operating on 15-minute OHLCV data.
"""

__version__ = "1.0.0"
__author__ = "Quantitative Systems Team"

# Expose main components for easy import
from probabilistic_market_engine.core import (
    PhysicsEngine,
    PhysicsState,
    LiquidityEngine,
    LiquidityState,
    FeaturePipeline,
    FeatureSet,
    RegimeInferenceModel,
    RegimeState,
    OutcomeExpertModels,
    OutcomePrediction,
    InferenceEngine,
    InferenceResult,
    RiskEngine,
    RiskAdjustment,
)

from probabilistic_market_engine.training import (
    WalkForwardValidator,
    RegimeTrainer,
    OutcomeTrainer,
    CalibrationEvaluator,
)

from probabilistic_market_engine.monitoring import (
    DriftDetector,
    DriftAlert,
    RegimeTracker,
    CalibrationTracker,
)

from probabilistic_market_engine.persistence import (
    ModelRegistry,
    FeatureStore,
)

__all__ = [
    # Core engines
    'PhysicsEngine',
    'PhysicsState',
    'LiquidityEngine',
    'LiquidityState',
    'FeaturePipeline',
    'FeatureSet',
    'RegimeInferenceModel',
    'RegimeState',
    'OutcomeExpertModels',
    'OutcomePrediction',
    'InferenceEngine',
    'InferenceResult',
    'RiskEngine',
    'RiskAdjustment',
    # Training
    'WalkForwardValidator',
    'RegimeTrainer',
    'OutcomeTrainer',
    'CalibrationEvaluator',
    # Monitoring
    'DriftDetector',
    'DriftAlert',
    'RegimeTracker',
    'CalibrationTracker',
    # Persistence
    'ModelRegistry',
    'FeatureStore',
]
