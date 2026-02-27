# Probabilistic Market Engine

A production-grade nonlinear probabilistic state-space trading engine operating on 15-minute OHLCV data, producing probabilistic continuation forecasts under latent regime uncertainty.

## Architecture Overview

The system follows a strict 7-layer architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 7: Risk Adjustment Engine                                  │
│   Risk_scale = f(Pr(Y), Pr(Shock), Entropy)                     │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 6: Mixture Aggregation Inference Engine                    │
│   Pr(Y=1) = Σ Pr(Y=1 | X, Z=k) × Pr(Z=k)                        │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: Regime-Conditional Outcome Experts                      │
│   For each k: Logistic Regression Pr(Y=1 | X, Z=k)              │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Bayesian Regime Inference (GMM, K=3)                    │
│   Input: R_t → [Pr(Trend), Pr(Range), Pr(Shock)]                │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 3: Feature Pipeline                                        │
│   Rolling window, standardization, no lookahead bias            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 2: Liquidity Topology Engine                               │
│   KDE of turning points, time-decayed structural density        │
└─────────────────────────────────────────────────────────────────┘
                              ▲
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: Deterministic Physics Engine                            │
│   OHLCV → [pressure, flow, energy, shock metrics]               │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
probabilistic_market_engine/
├── config/
│   ├── settings.py          # Configuration management
│   └── model_config.yaml    # Hyperparameters
├── core/
│   ├── physics_engine/      # Layer 1: Deterministic features
│   ├── liquidity_engine/    # Layer 2: Liquidity topology
│   ├── feature_pipeline/    # Layer 3: Feature preparation
│   ├── regime_model/        # Layer 4: GMM regime inference
│   ├── outcome_models/      # Layer 5: Expert models
│   ├── inference_engine/    # Layer 6: Mixture aggregation
│   └── risk_engine/         # Layer 7: Risk adjustment
├── training/
│   ├── regime_training/     # Regime model training
│   ├── outcome_training/    # Outcome model training
│   ├── walkforward/         # Walk-forward validation
│   └── calibration/         # Calibration evaluation
├── monitoring/
│   ├── drift_detection/     # Feature/regime drift detection
│   ├── regime_tracking/     # Regime statistics tracking
│   └── calibration_tracking/# Calibration monitoring
├── persistence/
│   ├── model_registry/      # Model versioning and storage
│   └── feature_store/       # Feature storage and retrieval
├── api/
│   └── main.py              # FastAPI application
└── tests/                   # Comprehensive test suite
```

## Installation

```bash
# Clone repository
git clone <repo-url>
cd probabilistic-market-engine

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train models on historical data
python train.py --data data/btc_15m.csv --version v1.0.0

# Skip validation for faster training
python train.py --data data/btc_15m.csv --no-validate
```

### API Server

```bash
# Start API server
uvicorn probabilistic_market_engine.api.main:app --host 0.0.0.0 --port 8000

# With auto-reload (development)
uvicorn probabilistic_market_engine.api.main:app --reload
```

### API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
payload = {
    "ohlcv_window": [
        {"open": 45000, "high": 45100, "low": 44900, "close": 45050, "volume": 100},
        # ... more bars (minimum 64)
    ],
    "symbol": "BTC-USD"
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()

print(f"Continuation Probability: {result['prob_continuation']}")
print(f"Regime: {result['regime_probabilities']}")
print(f"Risk Scaling: {result['risk_scaling_factor']}")
```

## Configuration

All hyperparameters are centralized in `probabilistic_market_engine/config/model_config.yaml`:

```yaml
physics:
  lookback_windows: [4, 16, 64]
  pressure_normalization_lookback: 256
  shock_detection_threshold: 2.5

regime_model:
  n_regimes: 3
  persistence_alpha: 0.7

outcome_model:
  prediction_horizon: 4
  calibration_method: isotonic
```

Environment variables can override YAML values:

```bash
export REGIME_MODEL_N_REGIMES=4
export OUTCOME_MODEL_PREDICTION_HORIZON=8
```

## Key Design Principles

### 1. Strict Layer Separation
- No layer bypassing
- No mixing of regime and outcome logic
- Clear data flow: L1 → L2 → L3 → L4 → L5 → L6 → L7

### 2. No Lookahead Bias
- All statistics computed on expanding windows
- Feature standardization uses only past data
- Walk-forward validation enforced

### 3. Stateless Inference
- No hidden state except controlled regime persistence
- Reproducible predictions given same inputs
- No reliance on global variables

### 4. Production-First
- Comprehensive monitoring and alerting
- Model versioning and registry
- Calibration tracking and drift detection

## Feature Engineering

### Layer 1: Physics Features
- **Normalized Pressure**: Volume-weighted buying/selling pressure
- **Multi-scale Flow**: Returns at 1h, 4h, 16h timescales
- **Path Efficiency**: Net displacement / total distance
- **Acceleration/Convexity**: Second-order path derivatives
- **Energy Metrics**: Kinetic and potential energy analogs
- **Shock Index**: Extreme move detection

### Layer 2: Liquidity Features
- **Distance to Support/Resistance**: KDE-based proximity to pivot clusters
- **Liquidity Density**: Probability density at current price
- **Turnover Intensity**: Time-decayed volume at price level

## Regime Model

Uses Gaussian Mixture Model (K=3) on regime features:
- `pressure_norm`: Normalized pressure
- `alignment_score`: Multi-scale alignment
- `path_efficiency`: Directional efficiency
- `volatility_ratio`: Short/long volatility ratio
- `shock_index`: Shock magnitude

Output: Soft probabilities for Trend/Range/Shock regimes with persistence smoothing.

## Outcome Models

Three logistic regression experts, one per regime:
- Training uses soft weights = Pr(Z=k)
- Supports Platt scaling or isotonic calibration
- Feature importance tracked per regime

## Risk Engine

Risk scaling formula:
```
scale = base × |Pr(Y) - 0.5| × 2 × (1 - shock_penalty) × (1 - uncertainty_penalty)
```

Where:
- `shock_penalty = Pr(Shock) × shock_penalty_factor`
- `uncertainty_penalty = Entropy × uncertainty_penalty_factor`

## Monitoring

### Drift Detection
- Feature mean/variance drift (z-score based)
- Regime distribution drift (KL divergence)
- Prediction distribution drift (KS test)

### Regime Tracking
- Regime flip frequency
- Average regime entropy
- Regime duration statistics

### Calibration Tracking
- Rolling Brier score
- Expected calibration error
- Reliability diagrams

## Testing

```bash
# Run all tests
python -m pytest probabilistic_market_engine/tests/

# Run specific test module
python -m pytest probabilistic_market_engine/tests/test_physics_engine.py -v

# Run with coverage
python -m pytest --cov=probabilistic_market_engine probabilistic_market_engine/tests/
```

## Development Strategy

Following the strategic development rule:

1. **Phase 1 (Current)**: Correct architecture
   - Modular, layered design
   - Clear interfaces
   - Comprehensive testing

2. **Phase 2**: Signal validation
   - Walk-forward backtesting
   - Calibration analysis
   - Regime stability metrics

3. **Phase 3**: Optimization
   - Performance tuning
   - Caching strategies
   - Parallelization

**Premature optimization is prohibited until signal validity is confirmed.**

## License

Proprietary - All rights reserved.

## Contact

For questions or issues, contact the quantitative systems team.
