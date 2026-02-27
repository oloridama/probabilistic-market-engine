# Usage Guide: Probabilistic Market Engine

This guide shows how to use the `probabilistic-market-engine` package in your XAU/USD project.

## Option 1: Install as a Package (Recommended)

### Step 1: Install the package

From your XAU/USD project directory:

```bash
# Install directly from GitHub
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0

# Or install in editable mode for development
pip install -e git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0#egg=probabilistic-market-engine
```

### Step 2: Use in your XAU/USD code

```python
import pandas as pd
from probabilistic_market_engine import (
    PhysicsEngine,
    LiquidityEngine,
    FeaturePipeline,
    RegimeInferenceModel,
    OutcomeExpertModels,
    InferenceEngine,
    RiskEngine,
    RegimeTrainer,
    OutcomeTrainer,
)

# Load your XAU/USD data
df = pd.read_csv('xauusd_15m.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# Ensure columns are named correctly
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Initialize engines
physics_engine = PhysicsEngine()
liquidity_engine = LiquidityEngine()

# Create feature pipeline
feature_pipeline = FeaturePipeline(
    physics_engine=physics_engine,
    liquidity_engine=liquidity_engine
)

# Compute features
feature_sets = feature_pipeline.fit_transform(df)
print(f"Computed {len(feature_sets)} feature sets")

# Train regime model
import numpy as np
R_features = np.array([fs.R_t for fs in feature_sets])
regime_model = RegimeInferenceModel()
regime_model.fit(R_features)

# Get regime predictions
regime_states = regime_model.predict_batch(R_features)

# Train outcome models (simplified example)
X_features = np.array([fs.X_t for fs in feature_sets])
closes = df['close'].values[65:]  # Align with features

# Generate labels (example: next bar direction)
future_returns = np.diff(closes) / closes[:-1]
labels = (future_returns > 0).astype(int)
labels = np.append(labels, 0)

regime_probs = np.array([
    [s.trend_probability, s.range_probability, s.shock_probability]
    for s in regime_states
])

outcome_models = OutcomeExpertModels()
outcome_models.fit(
    X_features[:len(labels)],
    labels,
    regime_probs[:len(labels)]
)

# Create inference engine
inference_engine = InferenceEngine(outcome_models)
risk_engine = RiskEngine()

# Make predictions
results = inference_engine.predict_batch(X_features[-100:], regime_states[-100:])
for result in results[-5:]:
    risk_adj = risk_engine.compute(result)
    print(f"Prob: {result.prob_continuation:.3f}, "
          f"Risk: {risk_adj.risk_scaling_factor:.3f}, "
          f"Direction: {risk_adj.suggested_direction}")
```

## Option 2: Git Submodule (Alternative)

If you need to modify the engine while working on your XAU/USD project:

```bash
# In your XAU/USD project
cd your-xauusd-project
git submodule add https://github.com/YOUR_USERNAME/probabilistic-market-engine.git
git submodule update --init --recursive

# Install in editable mode
pip install -e ./probabilistic-market-engine
```

## Option 3: Local Copy (Simplest for testing)

```bash
# Copy the package folder to your project
cp -r /path/to/probabilistic_market_engine ./

# Install dependencies
pip install -r probabilistic_market_engine/requirements.txt

# Add to Python path in your script
import sys
sys.path.insert(0, './probabilistic_market_engine')
```

## Training Script for XAU/USD

Create a script in your XAU/USD project:

```python
#!/usr/bin/env python
"""Train the engine on XAU/USD data."""

import pandas as pd
from probabilistic_market_engine import (
    FeaturePipeline,
    RegimeInferenceModel,
    OutcomeExpertModels,
    InferenceEngine,
    PhysicsEngine,
    LiquidityEngine,
)
from probabilistic_market_engine.training import RegimeTrainer, OutcomeTrainer
from probabilistic_market_engine.persistence import ModelRegistry

def train_on_xauusd(data_path: str, version: str = "v1.0.0"):
    # Load data
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    print(f"Training on {len(df)} bars of XAU/USD data")
    
    # Initialize
    physics = PhysicsEngine()
    liquidity = LiquidityEngine()
    pipeline = FeaturePipeline(physics_engine=physics, liquidity_engine=liquidity)
    
    # Compute features
    feature_sets = pipeline.fit_transform(df)
    
    # Train regime model
    import numpy as np
    R_features = np.array([fs.R_t for fs in feature_sets])
    regime_trainer = RegimeTrainer()
    regime_model = regime_trainer.train(R_features)
    
    # Train outcome models
    X_features = np.array([fs.X_t for fs in feature_sets])
    closes = df['close'].values[65:65+len(X_features)]
    
    outcome_trainer = OutcomeTrainer()
    labels = outcome_trainer.generate_labels(closes)
    
    valid_mask = ~np.isnan(labels)
    regime_states = regime_model.predict_batch(R_features)
    regime_probs = np.array([[s.trend_probability, s.range_probability, s.shock_probability] 
                              for s in regime_states])
    
    outcome_models = OutcomeExpertModels()
    outcome_models.fit(
        X_features[valid_mask],
        labels[valid_mask].astype(int),
        regime_probs[valid_mask]
    )
    
    # Save models
    registry = ModelRegistry()
    registry.register_version(
        version=version,
        regime_model=regime_model,
        outcome_models=outcome_models,
        feature_pipeline=pipeline,
        description=f"XAU/USD model trained on {len(df)} bars"
    )
    registry.set_active_version(version)
    
    print(f"Model saved as version {version}")

if __name__ == '__main__':
    import sys
    train_on_xauusd(sys.argv[1])
```

## Running the API Server

```bash
# After installing the package
python -m uvicorn probabilistic_market_engine.api.main:app --host 0.0.0.0 --port 8000

# Or from your XAU/USD project
cd your-xauusd-project
python -c "
import uvicorn
from probabilistic_market_engine.api.main import app
uvicorn.run(app, host='0.0.0.0', port=8000)
"
```

## Quick Test

```python
import numpy as np
import pandas as pd
from probabilistic_market_engine import PhysicsEngine

# Create sample OHLCV data
dates = pd.date_range('2024-01-01', periods=100, freq='15min')
np.random.seed(42)
df = pd.DataFrame({
    'open': 2000 + np.random.randn(100).cumsum() * 2,
    'high': 2000 + np.random.randn(100).cumsum() * 2 + 1,
    'low': 2000 + np.random.randn(100).cumsum() * 2 - 1,
    'close': 2000 + np.random.randn(100).cumsum() * 2,
    'volume': np.random.rand(100) * 1000
}, index=dates)

# Ensure OHLC consistency
df['high'] = df[['open', 'close', 'high']].max(axis=1)
df['low'] = df[['open', 'close', 'low']].min(axis=1)

# Compute physics features
engine = PhysicsEngine()
state = engine.compute(df)
print(f"Physics features computed: {len(state.to_vector())} features")
print(f"Sample features: pressure={state.pressure_norm:.3f}, flow={state.flow_short:.3f}")
```

## Requirements

Add to your XAU/USD project's `requirements.txt`:

```
# Your existing requirements
pandas
numpy

# Probabilistic Market Engine
probabilistic-market-engine @ git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0
```

Or install manually:
```bash
pip install numpy pandas scipy scikit-learn fastapi uvicorn pyyaml
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0
```
