# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-02-28

### Fixed
- **GMM Covariance Collapse**: Fixed critical issue where regime model would classify all samples as "shock" with 100% confidence
  - Changed default `covariance_type` from `'full'` to `'spherical'` to prevent component collapse
  - Added `reg_covar` parameter (default: 0.01) for covariance regularization
  - Components now properly separate into trend/range/shock regimes

### Added
- **Configurable GMM Parameters**: `RegimeModelConfig` now supports:
  - `covariance_type`: Choice of 'full', 'tied', 'diag', or 'spherical' (default: 'spherical')
  - `reg_covar`: Regularization parameter to prevent singular covariance matrices (default: 0.01)
- Environment variable support for new parameters:
  - `REGIME_MODEL_COVARIANCE_TYPE`
  - `REGIME_MODEL_REG_COVAR`

### Technical Details
- **Problem**: With `covariance_type='full'`, components collapsed to near-zero variance on some features (volatility_ratio std ≈ 0.001), causing one component to capture 100% of data
- **Solution**: `spherical` covariance allows different variances per component while preventing collapse through regularization
- **Result**: Regime distribution now balanced (Range ~60%, Shock ~20%, Trend ~20%) with healthy uncertainty (entropy 0.02-0.21)

## [1.0.0] - 2026-02-27

### Added
- Initial release of Probabilistic Market Engine
- 7-layer architecture implementation:
  - Layer 1: Physics Engine (18 deterministic features)
  - Layer 2: Liquidity Engine (KDE-based topology)
  - Layer 3: Feature Pipeline (rolling standardization)
  - Layer 4: Regime Inference (GMM with persistence smoothing)
  - Layer 5: Outcome Experts (regime-conditional models)
  - Layer 6: Inference Engine (mixture aggregation)
  - Layer 7: Risk Engine (position sizing with penalties)
- Persistence layer with ModelRegistry and FeatureStore
- Training pipeline with walk-forward validation
- Calibration evaluation (Brier score, ECE)
- Monitoring framework with drift detection
- FastAPI server for real-time inference
