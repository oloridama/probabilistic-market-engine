"""
Demo script showing end-to-end usage of the trading engine.
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_periods=500, seed=42):
    """Generate synthetic OHLCV data with regime changes."""
    np.random.seed(seed)
    
    # Generate price series with regime changes
    regimes = []
    prices = [100.0]
    
    for i in range(n_periods):
        # Cycle through regimes
        regime_idx = (i // 100) % 3
        regimes.append(['trend_up', 'range', 'trend_down'][regime_idx])
        
        if regime_idx == 0:  # Trend up
            ret = np.random.randn() * 0.005 + 0.002
        elif regime_idx == 1:  # Range
            ret = np.random.randn() * 0.008
        else:  # Trend down
            ret = np.random.randn() * 0.005 - 0.002
        
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices + np.random.randn(n_periods) * 0.1,
        'high': prices + np.abs(np.random.randn(n_periods)) * 0.5 + 0.1,
        'low': prices - np.abs(np.random.randn(n_periods)) * 0.5 - 0.1,
        'close': prices,
        'volume': np.random.rand(n_periods) * 1000 + 500,
    }, index=pd.date_range('2024-01-01', periods=n_periods, freq='15min'))
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df


def demo_full_pipeline():
    """Demonstrate the full pipeline."""
    logger.info("=" * 60)
    logger.info("PROBABILISTIC STATE-SPACE TRADING ENGINE DEMO")
    logger.info("=" * 60)
    
    # Import components
    from probabilistic_market_engine.core.physics_engine import PhysicsEngine
    from probabilistic_market_engine.core.liquidity_engine import LiquidityEngine
    from probabilistic_market_engine.core.feature_pipeline import FeaturePipeline
    from probabilistic_market_engine.core.regime_model import RegimeInferenceModel
    from probabilistic_market_engine.core.outcome_models import OutcomeExpertModels
    from probabilistic_market_engine.core.inference_engine import InferenceEngine
    from probabilistic_market_engine.core.risk_engine import RiskEngine
    
    # 1. Generate data
    logger.info("\n1. Generating synthetic OHLCV data...")
    df = generate_synthetic_data(n_periods=800)
    logger.info(f"   Generated {len(df)} periods")
    logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # 2. Initialize engines
    logger.info("\n2. Initializing engines...")
    physics_engine = PhysicsEngine()
    liquidity_engine = LiquidityEngine()
    logger.info("   ✓ Physics Engine")
    logger.info("   ✓ Liquidity Engine")
    
    # 3. Feature pipeline
    logger.info("\n3. Computing features...")
    feature_pipeline = FeaturePipeline(
        physics_engine=physics_engine,
        liquidity_engine=liquidity_engine
    )
    feature_sets = feature_pipeline.fit_transform(df)
    logger.info(f"   ✓ Computed {len(feature_sets)} feature sets")
    logger.info(f"   ✓ Feature dimensions: X_t={feature_sets[0].X_t.shape}, R_t={feature_sets[0].R_t.shape}")
    
    # 4. Train regime model
    logger.info("\n4. Training regime inference model (GMM)...")
    R_features = np.array([fs.R_t for fs in feature_sets])
    regime_model = RegimeInferenceModel()
    regime_model.fit(R_features)
    logger.info("   ✓ Regime model trained")
    
    # Get regime predictions
    regime_states = regime_model.predict_batch(R_features)
    regime_probs = np.array([
        [s.trend_probability, s.range_probability, s.shock_probability]
        for s in regime_states
    ])
    
    # Show regime distribution
    trend_count = sum(1 for s in regime_states if s.dominant_regime == 'trend')
    range_count = sum(1 for s in regime_states if s.dominant_regime == 'range')
    shock_count = sum(1 for s in regime_states if s.dominant_regime == 'shock')
    logger.info(f"   Regime distribution: Trend={trend_count}, Range={range_count}, Shock={shock_count}")
    
    # 5. Train outcome models
    logger.info("\n5. Training outcome expert models...")
    X_features = np.array([fs.X_t for fs in feature_sets])
    
    # Generate labels
    min_window = max(feature_pipeline.config.rolling_window,
                    max(feature_pipeline.physics_engine.config.lookback_windows) + 1)
    closes = df['close'].values[min_window:min_window + len(X_features)]
    
    # Simple label generation: next return direction
    future_returns = np.diff(closes) / closes[:-1]
    labels = (future_returns > 0).astype(int)
    labels = np.append(labels, 0)  # Last bar has no label
    
    # Filter valid
    valid_mask = ~np.isnan(labels)
    if np.sum(valid_mask) > 100:
        outcome_models = OutcomeExpertModels()
        outcome_models.fit(
            X_features[valid_mask],
            labels[valid_mask],
            regime_probs[valid_mask],
            feature_names=feature_pipeline._feature_names
        )
        logger.info("   ✓ Outcome models trained")
    else:
        logger.warning("   ! Not enough valid labels, using placeholder models")
        outcome_models = OutcomeExpertModels()
    
    # 6. Inference engine
    logger.info("\n6. Running inference...")
    inference_engine = InferenceEngine(outcome_models)
    
    # Make predictions on last 50 samples
    test_indices = slice(-50, None)
    test_results = inference_engine.predict_batch(
        X_features[test_indices],
        regime_states[test_indices]
    )
    logger.info(f"   ✓ Generated {len(test_results)} predictions")
    
    # Show sample predictions
    logger.info("\n   Sample predictions (last 5):")
    for i, result in enumerate(test_results[-5:]):
        logger.info(f"   [{i+1}] Prob={result.prob_continuation:.3f}, "
                   f"Regime={result.regime_probabilities}")
    
    # 7. Risk engine
    logger.info("\n7. Computing risk adjustments...")
    risk_engine = RiskEngine()
    risk_adjustments = [risk_engine.compute(r) for r in test_results]
    logger.info("   ✓ Risk adjustments computed")
    
    # Show sample risk adjustments
    logger.info("\n   Sample risk adjustments (last 5):")
    for i, adj in enumerate(risk_adjustments[-5:]):
        direction = "LONG" if adj.suggested_direction == 1 else "SHORT" if adj.suggested_direction == -1 else "NEUTRAL"
        logger.info(f"   [{i+1}] Direction={direction}, Scale={adj.risk_scaling_factor:.3f}, "
                   f"Confidence={adj.signal_confidence:.3f}")
    
    # 8. Summary statistics
    logger.info("\n8. Summary Statistics:")
    probs = [r.prob_continuation for r in test_results]
    logger.info(f"   Mean prediction: {np.mean(probs):.3f}")
    logger.info(f"   Std prediction: {np.std(probs):.3f}")
    logger.info(f"   Min prediction: {np.min(probs):.3f}")
    logger.info(f"   Max prediction: {np.max(probs):.3f}")
    
    entropies = [r.regime_entropy for r in test_results]
    logger.info(f"   Mean regime entropy: {np.mean(entropies):.3f}")
    
    risk_scales = [a.risk_scaling_factor for a in risk_adjustments]
    logger.info(f"   Mean risk scale: {np.mean(risk_scales):.3f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE")
    logger.info("=" * 60)
    
    return {
        'df': df,
        'feature_pipeline': feature_pipeline,
        'regime_model': regime_model,
        'outcome_models': outcome_models,
        'inference_results': test_results,
        'risk_adjustments': risk_adjustments,
    }


def demo_api():
    """Demonstrate API usage (requires running server)."""
    import requests
    
    logger.info("\nAPI Demo")
    logger.info("-" * 40)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health")
        logger.info(f"Health check: {response.json()}")
    except requests.exceptions.ConnectionError:
        logger.warning("Server not running. Start with: uvicorn app.api.main:app")
        return
    
    # Generate test data
    df = generate_synthetic_data(n_periods=100)
    
    # Convert to API format
    ohlcv_window = []
    for idx, row in df.iterrows():
        ohlcv_window.append({
            'timestamp': idx.isoformat(),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
        })
    
    # Make prediction request
    payload = {
        'ohlcv_window': ohlcv_window,
        'symbol': 'DEMO-BTC'
    }
    
    logger.info("Sending prediction request...")
    response = requests.post("http://localhost:8000/predict", json=payload)
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Prediction: {result['prob_continuation']:.3f}")
        logger.info(f"Regimes: {result['regime_probabilities']}")
        logger.info(f"Risk scale: {result['risk_scaling_factor']:.3f}")
    else:
        logger.error(f"Error: {response.status_code} - {response.text}")


if __name__ == '__main__':
    # Run full pipeline demo
    results = demo_full_pipeline()
    
    # Optionally run API demo
    # demo_api()
