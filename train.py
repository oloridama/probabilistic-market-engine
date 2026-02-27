"""
Training script for the probabilistic state-space trading engine.

Usage:
    python train.py --data data.csv --output models/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from probabilistic_market_engine.config.settings import settings, Settings
from probabilistic_market_engine.core.physics_engine.engine import PhysicsEngine
from probabilistic_market_engine.core.liquidity_engine.engine import LiquidityEngine
from probabilistic_market_engine.core.feature_pipeline.pipeline import FeaturePipeline
from probabilistic_market_engine.core.regime_model.model import RegimeInferenceModel
from probabilistic_market_engine.core.outcome_models.experts import OutcomeExpertModels
from probabilistic_market_engine.core.inference_engine.engine import InferenceEngine
from probabilistic_market_engine.core.risk_engine.engine import RiskEngine

from probabilistic_market_engine.training.regime_training.trainer import RegimeTrainer
from probabilistic_market_engine.training.outcome_training.trainer import OutcomeTrainer
from probabilistic_market_engine.training.calibration.evaluator import CalibrationEvaluator
from probabilistic_market_engine.training.walkforward.validator import WalkForwardValidator

from probabilistic_market_engine.persistence.model_registry.registry import ModelRegistry


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(data_path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Add volume if missing
    if 'volume' not in df.columns:
        df['volume'] = 0
    
    # Parse timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def train_pipeline(df: pd.DataFrame, validate: bool = True):
    """Train the complete pipeline."""
    
    logger.info("=== Starting Training Pipeline ===")
    
    # 1. Initialize engines
    logger.info("Initializing engines...")
    physics_engine = PhysicsEngine(settings.physics)
    liquidity_engine = LiquidityEngine(settings.liquidity)
    
    # 2. Create and fit feature pipeline
    logger.info("Computing features...")
    feature_pipeline = FeaturePipeline(
        config=settings.feature_pipeline,
        physics_engine=physics_engine,
        liquidity_engine=liquidity_engine
    )
    
    feature_sets = feature_pipeline.fit_transform(df)
    logger.info(f"Computed {len(feature_sets)} feature sets")
    
    # Extract features
    R_features = np.array([fs.R_t for fs in feature_sets])
    X_features = np.array([fs.X_t for fs in feature_sets])
    timestamps = [fs.timestamp for fs in feature_sets]
    
    # 3. Train regime model
    logger.info("Training regime model...")
    regime_trainer = RegimeTrainer(settings.regime_model)
    regime_model = regime_trainer.train(R_features, timestamps)
    
    # Get regime predictions for all samples
    regime_states = regime_model.predict_batch(R_features, timestamps)
    regime_probs = np.array([
        [s.trend_probability, s.range_probability, s.shock_probability]
        for s in regime_states
    ])
    
    # 4. Train outcome models
    logger.info("Training outcome models...")
    
    # Align closes with features
    min_window = max(feature_pipeline.config.rolling_window,
                    max(feature_pipeline.physics_engine.config.lookback_windows) + 1)
    closes = df['close'].values[min_window:min_window + len(X_features)]
    
    outcome_trainer = OutcomeTrainer(settings.outcome_model)
    labels = outcome_trainer.generate_labels(closes)
    
    # Filter valid labels
    valid_mask = ~np.isnan(labels)
    n_valid = np.sum(valid_mask)
    
    logger.info(f"Valid labels: {n_valid} (positive rate: {np.mean(labels[valid_mask]):.2%})")
    
    if n_valid < settings.training.min_train_samples:
        raise ValueError(f"Not enough valid samples ({n_valid} < {settings.training.min_train_samples})")
    
    outcome_models = OutcomeExpertModels(settings.outcome_model)
    outcome_models.fit(
        X_features[valid_mask],
        labels[valid_mask].astype(int),
        regime_probs[valid_mask],
        feature_names=feature_pipeline._feature_names
    )
    
    # 5. Create inference engine
    inference_engine = InferenceEngine(outcome_models)
    inference_engine.set_model_version(settings.model_version)
    
    # 6. Evaluate calibration
    logger.info("Evaluating calibration...")
    
    # Make predictions on training set
    train_results = inference_engine.predict_batch(
        X_features[valid_mask],
        [regime_states[i] for i in np.where(valid_mask)[0]],
        [timestamps[i] for i in np.where(valid_mask)[0]]
    )
    
    predictions = np.array([r.prob_continuation for r in train_results])
    outcomes = labels[valid_mask].astype(int)
    
    evaluator = CalibrationEvaluator()
    calibration_result = evaluator.evaluate(outcomes, predictions)
    
    logger.info(f"Brier Score: {calibration_result.brier_score:.4f}")
    logger.info(f"Expected Calibration Error: {calibration_result.expected_calibration_error:.4f}")
    
    # 7. Walk-forward validation (optional)
    if validate and len(df) > settings.training.walkforward_train_size * 2:
        logger.info("Running walk-forward validation...")
        
        # Prepare data for walk-forward
        wf_data = pd.DataFrame({
            'X': list(X_features[valid_mask]),
            'R': list(R_features[valid_mask]),
            'target': outcomes,
            'timestamp': [timestamps[i] for i in np.where(valid_mask)[0]]
        })
        
        # This is a simplified walk-forward - in production would need more complex logic
        logger.info("Walk-forward validation would run here (simplified for demo)")
    
    # 8. Evaluate on training data
    train_metrics = {
        'brier_score': calibration_result.brier_score,
        'expected_calibration_error': calibration_result.expected_calibration_error,
        'max_calibration_error': calibration_result.maximum_calibration_error,
        'mean_prediction': float(np.mean(predictions)),
        'positive_rate': float(np.mean(outcomes)),
        'n_samples': int(n_valid),
    }
    
    logger.info("=== Training Complete ===")
    
    return {
        'regime_model': regime_model,
        'outcome_models': outcome_models,
        'feature_pipeline': feature_pipeline,
        'metrics': train_metrics,
    }


def save_models(results: dict, output_path: str):
    """Save trained models to registry."""
    logger.info(f"Saving models to {output_path}")
    
    registry = ModelRegistry()
    
    version = registry.register_version(
        version=settings.model_version,
        regime_model=results['regime_model'],
        outcome_models=results['outcome_models'],
        feature_pipeline=results['feature_pipeline'],
        description=f"Trained model version {settings.model_version}",
        metrics=results['metrics'],
        config=settings
    )
    
    registry.set_active_version(version.version)
    
    logger.info(f"Models saved as version {version.version}")
    
    return version


def main():
    parser = argparse.ArgumentParser(description='Train trading engine models')
    parser.add_argument('--data', type=str, required=True, help='Path to OHLCV CSV file')
    parser.add_argument('--output', type=str, default='models', help='Output directory')
    parser.add_argument('--no-validate', action='store_true', help='Skip walk-forward validation')
    parser.add_argument('--version', type=str, default=None, help='Model version override')
    
    args = parser.parse_args()
    
    # Override version if specified
    if args.version:
        settings.model_version = args.version
    
    # Load data
    df = load_data(args.data)
    
    # Train
    results = train_pipeline(df, validate=not args.no_validate)
    
    # Save
    version = save_models(results, args.output)
    
    # Print summary
    print("\n=== Training Summary ===")
    print(f"Model Version: {version.version}")
    print(f"Samples Used: {results['metrics']['n_samples']}")
    print(f"Brier Score: {results['metrics']['brier_score']:.4f}")
    print(f"Calibration Error: {results['metrics']['expected_calibration_error']:.4f}")
    print(f"\nModels saved to: {args.output}")


if __name__ == '__main__':
    main()
