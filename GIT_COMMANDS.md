# Git Commands for Publishing

Follow these steps to commit the project to GitHub with version tags.

## Initial Setup

```bash
# Navigate to your project directory
cd "/home/christian/Desktop/Untitled Folder"

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Probabilistic Market Engine v1.0.0

Production-grade nonlinear probabilistic state-space trading engine:
- 7-layer strict architecture
- GMM-based regime inference (K=3)
- Regime-conditional logistic experts
- Walk-forward validation
- Calibration monitoring
- Drift detection
- FastAPI server
- Model registry and versioning"

# Add remote repository (replace with your actual repo URL)
git remote add origin https://github.com/YOUR_USERNAME/probabilistic-market-engine.git

# Push to main branch
git branch -M main
git push -u origin main
```

## Create Version Tag

```bash
# Create annotated tag for v1.0.0
git tag -a v1.0.0 -m "Release v1.0.0 - Initial production release

Features:
- Full 7-layer architecture implementation
- Physics engine with 18 deterministic features
- Liquidity topology engine with KDE
- Bayesian regime inference (GMM)
- Regime-conditional outcome experts
- Mixture aggregation inference
- Risk adjustment engine
- Walk-forward validation framework
- Calibration evaluation (Brier, ECE)
- Drift detection and monitoring
- Model registry with versioning
- FastAPI prediction endpoint
- Comprehensive test suite"

# Push tags to remote
git push origin v1.0.0
```

## Verify Installation from GitHub

```bash
# Test installing from GitHub in a fresh environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from GitHub
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0

# Test import
python -c "from probabilistic_market_engine import PhysicsEngine; print('Success!')"

# Cleanup
 deactivate
cd ..
rm -rf test_env
```

## Future Version Updates

```bash
# Make changes to code...

# Update version in files
# - pyproject.toml: version = "1.1.0"
# - setup.py: version="1.1.0"
# - probabilistic_market_engine/__init__.py: __version__ = "1.1.0"

# Commit changes
git add .
git commit -m "feat: Add feature X, fix bug Y

- Description of changes
- More details"

# Create new tag
git tag -a v1.1.0 -m "Release v1.1.0 - Description"

# Push
git push origin main
git push origin v1.1.0
```

## Useful Git Commands

```bash
# Check status
git status

# View tags
git tag -l

# View tag details
git show v1.0.0

# Delete local tag (if needed)
git tag -d v1.0.0

# Delete remote tag (if needed)
git push --delete origin v1.0.0

# List all remote branches and tags
git ls-remote

# View commit history
git log --oneline --graph --all

# Check what will be pushed
git diff --stat origin/main
```

## Installing Specific Versions

Users can install specific versions:

```bash
# Latest main branch
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git

# Specific version
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0

# Specific commit
pip install git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@abc123

# For development (editable)
pip install -e git+https://github.com/YOUR_USERNAME/probabilistic-market-engine.git@v1.0.0#egg=probabilistic-market-engine
```

## GitHub Actions (Optional)

Create `.github/workflows/python-package.yml` for CI/CD:

```yaml
name: Python Package

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    - name: Test with pytest
      run: |
        pytest probabilistic_market_engine/tests/ -v
```
