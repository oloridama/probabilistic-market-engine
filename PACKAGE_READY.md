# Package Ready for GitHub

The project has been restructured as a proper Python package named `probabilistic-market-engine`.

## What Was Changed

1. **Package Name**: `app/` → `probabilistic_market_engine/`
2. **Package Structure**: Added `pyproject.toml` and `setup.py` for pip installability
3. **Imports**: All internal imports updated to use new package name
4. **Config Paths**: Updated to reference new package structure
5. **Documentation**: Created USAGE.md and GIT_COMMANDS.md

## File Structure

```
probabilistic-market-engine/
├── probabilistic_market_engine/     # Main package (was: app/)
│   ├── config/
│   ├── core/                        # 7 layers
│   ├── training/
│   ├── monitoring/
│   ├── persistence/
│   ├── api/
│   └── tests/
├── pyproject.toml                   # Modern Python packaging
├── setup.py                         # Backwards compatibility
├── requirements.txt                 # Dependencies
├── .gitignore                       # Git ignore rules
├── README.md                        # Main documentation
├── ARCHITECTURE.md                  # Detailed architecture
├── USAGE.md                         # How to use in other projects
├── GIT_COMMANDS.md                  # Git commands for publishing
├── PACKAGE_READY.md                 # This file
├── train.py                         # Training script
└── demo.py                          # Demo script
```

## How to Use in Your XAU/USD Project

### Option 1: Install from GitHub (Recommended)

```bash
# In your XAU/USD project directory
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0

# Then in your Python code:
from probabilistic_market_engine import PhysicsEngine, FeaturePipeline
```

### Option 2: Git Submodule

```bash
# Add as submodule
cd your-xauusd-project
git submodule add https://github.com/YOUR_USERNAME/probabilistic-market-engine.git
pip install -e ./probabilistic-market-engine
```

### Option 3: Copy Package

```bash
# Copy package to your project
cp -r /path/to/probabilistic_market_engine ./
pip install -r probabilistic_market_engine/requirements.txt
```

## Quick Start Example

```python
import pandas as pd
from probabilistic_market_engine import (
    PhysicsEngine,
    LiquidityEngine, 
    FeaturePipeline,
    RegimeInferenceModel,
    InferenceEngine,
    RiskEngine,
)

# Load your XAU/USD data
df = pd.read_csv('xauusd_15m.csv', parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
df.columns = ['open', 'high', 'low', 'close', 'volume']

# Initialize pipeline
feature_pipeline = FeaturePipeline(
    physics_engine=PhysicsEngine(),
    liquidity_engine=LiquidityEngine()
)

# Compute features and train (see USAGE.md for full example)
feature_sets = feature_pipeline.fit_transform(df)
```

## Next Steps to Publish

1. **Create GitHub Repository**: Go to GitHub and create a new repo named `probabilistic-market-engine`

2. **Run Git Commands** (from GIT_COMMANDS.md):
```bash
cd "/home/christian/Desktop/Untitled Folder"
git init
git add .
git commit -m "Initial commit: Probabilistic Market Engine v1.0.0"
git remote add origin https://github.com/YOUR_USERNAME/probabilistic-market-engine.git
git branch -M main
git push -u origin main
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

3. **Verify Installation**:
```bash
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0
python -c "from probabilistic_market_engine import PhysicsEngine; print('OK')"
```

## Version Management

Future releases:
```bash
# Update version in:
# - pyproject.toml: version = "1.1.0"
# - setup.py: version="1.1.0"
# - probabilistic_market_engine/__init__.py: __version__ = "1.1.0"

git add .
git commit -m "feat: Add new features"
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin main
git push origin v1.1.0
```

## Package Exports

Main components available from package root:

```python
from probabilistic_market_engine import (
    # Core engines
    PhysicsEngine, PhysicsState,
    LiquidityEngine, LiquidityState,
    FeaturePipeline, FeatureSet,
    RegimeInferenceModel, RegimeState,
    OutcomeExpertModels, OutcomePrediction,
    InferenceEngine, InferenceResult,
    RiskEngine, RiskAdjustment,
    
    # Training
    WalkForwardValidator,
    RegimeTrainer,
    OutcomeTrainer,
    CalibrationEvaluator,
    
    # Monitoring
    DriftDetector, DriftAlert,
    RegimeTracker,
    CalibrationTracker,
    
    # Persistence
    ModelRegistry,
    FeatureStore,
)
```

## Important Notes

- **Data Storage**: Model registry and feature store paths are set to `probabilistic_market_engine/persistence/` subdirectories
- **Config**: YAML config is bundled with the package at `probabilistic_market_engine/config/model_config.yaml`
- **Tests**: Run with `pytest probabilistic_market_engine/tests/`
- **API**: Start with `uvicorn probabilistic_market_engine.api.main:app`

## Running with Your XAU/USD Data

See `USAGE.md` for complete examples including:
- Training on XAU/USD data
- Making predictions
- Running the API server
- Integration patterns
