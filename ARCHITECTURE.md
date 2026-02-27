# System Architecture Documentation

## Layer-by-Layer Specification

### Layer 1: Deterministic Physics Engine

**Purpose**: Transform OHLCV data into structured state features.

**Constraints**:
- No ML
- No probability modeling
- Pure deterministic transformations
- Stateless except controlled flow memory

**Computations**:

| Feature | Description | Formula |
|---------|-------------|---------|
| `pressure_norm` | Z-score normalized buying/selling pressure | `(current_pressure - mean) / std` |
| `flow_short` | Fast timescale momentum (4 bars) | `sum(returns) / std(returns) * sqrt(4)` |
| `flow_medium` | Medium timescale momentum (16 bars) | `sum(returns) / std(returns) * sqrt(16)` |
| `flow_long` | Slow timescale momentum (64 bars) | `sum(returns) / std(returns) * sqrt(64)` |
| `flow_alignment` | Alignment across timescales | `sign(flows[0]) * min(abs(flows)) / max(abs(flows))` |
| `path_efficiency` | Net displacement / total distance | `abs(sum(returns)) / sum(abs(returns))` |
| `directional_inertia` | Tendency to continue direction | Count of consecutive same-sign returns |
| `acceleration` | Change in returns | Second derivative approximation |
| `convexity` | Curvature of price path | Second derivative of cumulative returns |
| `compression_index` | Range compression | `1 - current_range / historical_range` |
| `energy_kinetic` | Velocity^2 * mass | `(mean_return)^2 * sign * sqrt(volume)` |
| `energy_potential` | Stored energy | `price_range / (volatility * sqrt(window))` |
| `volatility_ratio` | Short / long volatility | `std(short) / std(long)` |
| `shock_index` | Extreme move detection | `abs(z_score) * vol_ratio - threshold` |
| `alignment_score` | Overall feature alignment | Weighted average of component alignments |

**Output**: `PhysicsState` object with 18 features.

---

### Layer 2: Liquidity Topology Engine

**Purpose**: Maintain rolling structural density map of price interaction zones.

**Computations**:

| Feature | Description | Method |
|---------|-------------|--------|
| `distance_to_support` | Normalized distance to nearest support cluster | `abs(price - support) / price_range` |
| `distance_to_resistance` | Normalized distance to nearest resistance cluster | `abs(resistance - price) / price_range` |
| `liquidity_density` | KDE probability at current price | Gaussian KDE evaluation |
| `relative_position` | Position within historical range | `(price - min) / (max - min)` |
| `support_strength` | Weight of nearest support cluster | Pivot weight from prominence |
| `resistance_strength` | Weight of nearest resistance cluster | Pivot weight from prominence |
| `turnover_intensity` | Time-decayed volume at level | Exponentially weighted volume |

**Methods**:
- Pivot detection using `scipy.signal.argrelextrema`
- KDE using `scipy.stats.gaussian_kde`
- Time decay: `weight *= decay_factor` per period

**Output**: `LiquidityState` object with 7 features.

---

### Layer 3: Feature Pipeline

**Purpose**: Prepare features safely for modeling.

**Responsibilities**:
- Rolling window isolation
- Standardization
- No lookahead bias
- Train vs inference separation

**Process**:
1. Compute physics features via `PhysicsEngine`
2. Compute liquidity features via `LiquidityEngine`
3. Extract regime subset `R_t` from physics features
4. Standardize using expanding window statistics

**Regime Feature Subset (R_t)**:
- `pressure_norm`
- `alignment_score`
- `path_efficiency`
- `volatility_ratio`
- `shock_index`

**Output**: `FeatureSet` containing `X_t` (25 features) and `R_t` (5 features).

---

### Layer 4: Bayesian Regime Inference

**Model**: Gaussian Mixture Model (K=3)

**Input**: `R_t` only - 5 regime features

**Process**:
1. Standardize regime features
2. Fit GMM offline: `sklearn.mixture.GaussianMixture`
3. Save model parameters
4. Inference: `posterior = gmm.predict_proba(R_t)`
5. Persistence smoothing: `Pr(Z_t) = α Pr(Z_{t-1}) + (1-α) posterior`

**Regime Label Assignment**:
- **Trend**: High alignment + efficiency
- **Range**: Low volatility + neutral alignment
- **Shock**: High volatility + shock index

**Output**: `RegimeState` with probabilities:
- `trend_probability`
- `range_probability`
- `shock_probability`
- `regime_entropy` (uncertainty)

---

### Layer 5: Regime-Conditional Outcome Experts

**Model**: Logistic Regression per regime

**Formula**: `Pr(Y=1 | X, Z=k) = 1 / (1 + exp(-(β_k · X + b_k)))`

**Training**:
- Soft weights = `Pr(Z=k)`
- Sample weights for class balance
- Walk-forward validation

**Calibration** (optional):
- Isotonic regression
- Platt scaling (sigmoid)

**Output**: `OutcomePrediction` per regime:
- `probability`: Raw logistic output
- `calibrated_probability`: Post-calibration
- `feature_contributions`: Linear decomposition

---

### Layer 6: Mixture Aggregation Inference Engine

**Formula**:
```
Pr(Y=1) = Σ_k Pr(Y=1 | X, Z=k) × Pr(Z=k)
        = Pr(Y|Trend)×Pr(Trend) + Pr(Y|Range)×Pr(Range) + Pr(Y|Shock)×Pr(Shock)
```

**Properties**:
- Deterministic
- Reproducible
- Only layer exposed to API

**Output**: `InferenceResult`:
- `prob_continuation`: Final aggregated probability
- `regime_probabilities`: Component regime probs
- `shock_probability`: Extracted for risk
- `prediction_confidence`: Model confidence metric
- `regime_entropy`: Regime uncertainty

---

### Layer 7: Risk Adjustment Engine

**Formula**:
```
Risk_scale = base × signal_strength × (1 - shock_penalty) × (1 - uncertainty_penalty)

where:
  signal_strength = |Pr(Y) - 0.5| × 2
  shock_penalty = Pr(Shock) × shock_penalty_factor
  uncertainty_penalty = Entropy × uncertainty_penalty_factor
```

**Direction Determination**:
- `Pr(Y) > 0.55`: Long (1)
- `Pr(Y) < 0.45`: Short (-1)
- Otherwise: Neutral (0)

**Output**: `RiskAdjustment`:
- `risk_scaling_factor`: Final position size multiplier
- `suggested_direction`: -1, 0, or 1
- `signal_confidence`: Strength of signal
- `execution_confidence`: Confidence adjusted for risk

---

## Data Flow Diagram

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  OHLCV      │────▶│  Physics Engine │────▶│  PhysicsState   │
│  (15min)    │     │  (Layer 1)      │     │  (18 features)  │
└─────────────┘     └─────────────────┘     └────────┬────────┘
       │                                             │
       │         ┌─────────────────┐                │
       │         │  Liquidity Eng. │                │
       └────────▶│  (Layer 2)      │───────────────▶│
                 │  (7 features)   │                │
                 └─────────────────┘                │
                                                    ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Pipeline (Layer 3)               │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Full Features  │    │  Regime Subset  │                │
│  │  X_t (25 dims)  │    │  R_t (5 dims)   │                │
│  └────────┬────────┘    └────────┬────────┘                │
│           │                      │                          │
└───────────┼──────────────────────┼──────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────┐       ┌─────────────────┐
│ Outcome Experts │       │  Regime Model   │
│ (Layer 5)       │       │  (Layer 4)      │
│ Logistic Reg.   │       │  GMM (K=3)      │
└────────┬────────┘       └────────┬────────┘
         │                         │
         │    Pr(Y|X,Z=k)          │ Pr(Z=k|R)
         │                         │
         └───────────┬─────────────┘
                     ▼
         ┌───────────────────────┐
         │  Inference Engine     │
         │  (Layer 6)            │
         │  Mixture Aggregation  │
         └───────────┬───────────┘
                     │ Pr(Y=1)
                     ▼
         ┌───────────────────────┐
         │  Risk Engine          │
         │  (Layer 7)            │
         │  Position Sizing      │
         └───────────┬───────────┘
                     │ Risk_scale
                     ▼
         ┌───────────────────────┐
         │  API Response         │
         └───────────────────────┘
```

---

## Monitoring Architecture

### Drift Detection

| Metric | Method | Threshold | Action |
|--------|--------|-----------|--------|
| Feature Drift | Z-score vs reference | 2.0 std | Alert |
| Regime Drift | KL divergence | 0.1 | Alert |
| Prediction Drift | KS test | 0.2 | Alert |

### Regime Tracking

| Metric | Description |
|--------|-------------|
| Flip Frequency | Regime changes per 100 periods |
| Average Entropy | Mean regime uncertainty |
| Regime Duration | Persistence statistics per regime |

### Calibration Tracking

| Metric | Target | Warning |
|--------|--------|---------|
| Brier Score | < 0.20 | > 0.25 |
| ECE | < 0.05 | > 0.10 |
| Log Loss | < 0.69 | > 0.80 |

---

## API Specification

### POST /predict

**Request**:
```json
{
  "ohlcv_window": [
    {"timestamp": "2024-01-01T00:00:00", "open": 45000, "high": 45100, "low": 44900, "close": 45050, "volume": 100},
    // ... min 64 bars
  ],
  "symbol": "BTC-USD"
}
```

**Response**:
```json
{
  "prob_continuation": 0.6234,
  "regime_probabilities": {
    "trend": 0.45,
    "range": 0.30,
    "shock": 0.25
  },
  "shock_probability": 0.25,
  "risk_scaling_factor": 0.85,
  "model_version": "v1.0.0",
  "timestamp": "2024-01-01T12:00:00",
  "prediction_confidence": 0.72,
  "regime_entropy": 0.45
}
```

### GET /health

**Response**:
```json
{
  "status": "healthy",
  "model_version": "v1.0.0",
  "model_loaded": true
}
```

### GET /monitoring/status

**Response**:
```json
{
  "drift_detection": {
    "feature_stats": {...},
    "regime_stats": {...}
  },
  "regime_tracking": {
    "flip_frequency": 0.05,
    "average_entropy": 0.42,
    "current_regime": {...}
  },
  "calibration": {
    "rolling_brier": 0.18,
    "calibration_degrading": false
  }
}
```

---

## Configuration Hierarchy

1. **Default values** (code)
2. **YAML config** (`model_config.yaml`)
3. **Environment variables** (override YAML)
4. **Runtime parameters** (highest priority)

Environment variable naming:
```
{SECTION}_{KEY}
Examples:
  PHYSICS_LOOKBACK_WINDOWS
  REGIME_MODEL_N_REGIMES
  OUTCOME_MODEL_PREDICTION_HORIZON
```

---

## Testing Strategy

| Level | Scope | Tools |
|-------|-------|-------|
| Unit | Individual layers | pytest |
| Integration | Layer interactions | pytest |
| End-to-End | Full pipeline | demo.py |
| Performance | Latency, throughput | locust |

---

## Deployment Checklist

- [ ] Models trained and validated
- [ ] Calibration verified (ECE < 0.10)
- [ ] Drift detection baselines set
- [ ] Model version registered
- [ ] API health checks passing
- [ ] Monitoring dashboards configured
- [ ] Alert thresholds configured
- [ ] Rollback plan documented
