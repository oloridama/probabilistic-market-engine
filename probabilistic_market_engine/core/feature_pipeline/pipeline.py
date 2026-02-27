"""
Layer 3: Feature Pipeline

Prepares features safely for modeling.
Responsibilities:
- Rolling window isolation
- Standardization
- No lookahead bias
- Clear timestamp alignment
- Train vs inference pipeline separation
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

from probabilistic_market_engine.config.settings import FeaturePipelineConfig
from probabilistic_market_engine.core.physics_engine.engine import PhysicsEngine, PhysicsState
from probabilistic_market_engine.core.liquidity_engine.engine import LiquidityEngine, LiquidityState


@dataclass
class FeatureSet:
    """
    Container for prepared features.
    
    Contains:
    - X_t: Full feature vector for outcome models
    - R_t: Regime subset for regime inference
    - metadata: Timestamp and other tracking info
    """
    timestamp: Optional[pd.Timestamp] = None
    
    # Full feature vector
    X_t: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Regime inference subset
    R_t: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Feature names
    feature_names: List[str] = field(default_factory=list)
    regime_feature_names: List[str] = field(default_factory=list)
    
    # Metadata
    is_valid: bool = False
    missing_features: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert feature set to dictionary."""
        return {
            'timestamp': self.timestamp,
            'X_t': self.X_t.tolist() if len(self.X_t) > 0 else [],
            'R_t': self.R_t.tolist() if len(self.R_t) > 0 else [],
            'feature_names': self.feature_names,
            'regime_feature_names': self.regime_feature_names,
            'is_valid': self.is_valid,
            'missing_features': self.missing_features,
        }


class FeaturePipeline:
    """
    Feature pipeline for safe feature preparation.
    
    Key design principles:
    - No lookahead bias: only uses past data for statistics
    - Rolling window isolation: computations use fixed windows
    - Clear separation between training and inference modes
    """
    
    def __init__(self, config: Optional[FeaturePipelineConfig] = None,
                 physics_engine: Optional[PhysicsEngine] = None,
                 liquidity_engine: Optional[LiquidityEngine] = None):
        self.config = config or FeaturePipelineConfig()
        
        # Engines
        self.physics_engine = physics_engine or PhysicsEngine()
        self.liquidity_engine = liquidity_engine or LiquidityEngine()
        
        # Statistics for standardization (training mode)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._regime_feature_means: Optional[np.ndarray] = None
        self._regime_feature_stds: Optional[np.ndarray] = None
        
        # Rolling statistics buffers (inference mode)
        self._rolling_buffer: deque = deque(maxlen=self.config.standardization_lookback)
        self._regime_rolling_buffer: deque = deque(maxlen=self.config.standardization_lookback)
        
        # Mode flag
        self._is_training_mode: bool = False
        
        # Feature names (established during fit)
        self._feature_names: List[str] = []
        self._regime_feature_indices: Dict[str, int] = {}
    
    def set_training_mode(self, is_training: bool):
        """Set training mode flag."""
        self._is_training_mode = is_training
    
    def fit(self, ohlcv_data: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit standardization parameters on historical data.
        
        This should be called once on training data.
        Uses expanding window statistics to avoid lookahead bias.
        
        Args:
            ohlcv_data: Historical OHLCV data for fitting
        
        Returns:
            self for method chaining
        """
        self._is_training_mode = True
        
        # Compute features for entire dataset
        features_list = []
        regime_features_list = []
        
        min_window = max(self.config.rolling_window, 
                        max(self.physics_engine.config.lookback_windows) + 1)
        
        for i in range(min_window, len(ohlcv_data)):
            window = ohlcv_data.iloc[i-min_window:i]
            
            # Compute physics and liquidity features
            physics_state = self.physics_engine.compute(window)
            liquidity_state = self.liquidity_engine.compute(window)
            
            # Combine features
            features = np.concatenate([
                physics_state.to_vector(),
                liquidity_state.to_vector()
            ])
            
            # Extract regime features
            regime_features = self._extract_regime_features(physics_state, liquidity_state)
            
            features_list.append(features)
            regime_features_list.append(regime_features)
        
        # Compute statistics
        features_array = np.array(features_list)
        regime_features_array = np.array(regime_features_list)
        
        self._feature_means = np.mean(features_array, axis=0)
        self._feature_stds = np.std(features_array, axis=0) + 1e-10
        self._regime_feature_means = np.mean(regime_features_array, axis=0)
        self._regime_feature_stds = np.std(regime_features_array, axis=0) + 1e-10
        
        # Store feature names
        self._feature_names = self._get_feature_names()
        self._regime_feature_indices = {name: i for i, name in enumerate(self.config.feature_subset_regime)}
        
        # Store fitted statistics for inference
        self._is_training_mode = False
        
        return self
    
    def transform(self, ohlcv_window: pd.DataFrame) -> FeatureSet:
        """
        Transform OHLCV window to feature set.
        
        Args:
            ohlcv_window: DataFrame with OHLCV data
        
        Returns:
            FeatureSet containing X_t and R_t
        """
        feature_set = FeatureSet()
        feature_set.timestamp = ohlcv_window.index[-1] if isinstance(ohlcv_window.index, pd.DatetimeIndex) else None
        
        try:
            # Compute physics and liquidity features
            physics_state = self.physics_engine.compute(ohlcv_window)
            liquidity_state = self.liquidity_engine.compute(ohlcv_window)
            
            # Combine features
            features_raw = np.concatenate([
                physics_state.to_vector(),
                liquidity_state.to_vector()
            ])
            
            # Extract regime features
            regime_features_raw = self._extract_regime_features(physics_state, liquidity_state)
            
            # Standardize
            if self._is_training_mode or self._feature_means is None:
                # Use rolling statistics
                features_std = self._standardize_rolling(features_raw, is_regime=False)
                regime_features_std = self._standardize_rolling(regime_features_raw, is_regime=True)
            else:
                # Use fitted statistics
                features_std = (features_raw - self._feature_means) / self._feature_stds
                regime_features_std = (regime_features_raw - self._regime_feature_means) / self._regime_feature_stds
            
            feature_set.X_t = features_std
            feature_set.R_t = regime_features_std
            feature_set.feature_names = self._feature_names
            feature_set.regime_feature_names = self.config.feature_subset_regime
            feature_set.is_valid = True
            
        except Exception as e:
            feature_set.is_valid = False
            feature_set.missing_features = [str(e)]
        
        return feature_set
    
    def fit_transform(self, ohlcv_data: pd.DataFrame) -> List[FeatureSet]:
        """
        Fit on data and return transformed features.
        
        Args:
            ohlcv_data: Historical OHLCV data
        
        Returns:
            List of FeatureSet objects
        """
        self.fit(ohlcv_data)
        
        feature_sets = []
        min_window = max(self.config.rolling_window, 
                        max(self.physics_engine.config.lookback_windows) + 1)
        
        for i in range(min_window, len(ohlcv_data)):
            window = ohlcv_data.iloc[i-min_window:i]
            feature_set = self.transform(window)
            feature_sets.append(feature_set)
        
        return feature_sets
    
    def _extract_regime_features(self, physics_state: PhysicsState, 
                                  liquidity_state: LiquidityState) -> np.ndarray:
        """
        Extract regime inference subset from state objects.
        
        Regime features: [pressure_norm, alignment_score, path_efficiency, 
                         volatility_ratio, shock_index]
        """
        # Map from physics state
        physics_dict = physics_state.to_dict()
        
        regime_features = []
        for feature_name in self.config.feature_subset_regime:
            if feature_name in physics_dict:
                regime_features.append(physics_dict[feature_name])
            else:
                # Default to 0 if not found
                regime_features.append(0.0)
        
        return np.array(regime_features)
    
    def _standardize_rolling(self, features: np.ndarray, is_regime: bool = False) -> np.ndarray:
        """Standardize using rolling window statistics."""
        buffer = self._regime_rolling_buffer if is_regime else self._rolling_buffer
        
        # Add current features to buffer
        buffer.append(features)
        
        # Compute statistics from buffer
        if len(buffer) < self.config.min_samples_for_std:
            # Not enough data - return centered features
            return features - np.mean(list(buffer), axis=0)
        
        buffer_array = np.array(buffer)
        mean = np.mean(buffer_array, axis=0)
        std = np.std(buffer_array, axis=0) + 1e-10
        
        return (features - mean) / std
    
    def _get_feature_names(self) -> List[str]:
        """Generate feature names based on physics and liquidity features."""
        physics_names = [
            'pressure_norm', 'flow_short', 'flow_medium', 'flow_long',
            'flow_alignment', 'path_efficiency', 'directional_inertia',
            'acceleration', 'convexity', 'compression_index', 'range_efficiency',
            'energy_kinetic', 'energy_potential', 'energy_total',
            'volatility_ratio', 'shock_index', 'flow_discontinuity',
            'alignment_score'
        ]
        
        liquidity_names = [
            'dist_to_support', 'dist_to_resistance', 'liquidity_density',
            'relative_position', 'support_strength', 'resistance_strength',
            'turnover_intensity'
        ]
        
        return physics_names + liquidity_names
    
    def get_feature_dim(self) -> int:
        """Get dimension of full feature vector."""
        return len(self._get_feature_names())
    
    def get_regime_feature_dim(self) -> int:
        """Get dimension of regime feature vector."""
        return len(self.config.feature_subset_regime)
    
    def save_stats(self, path: str):
        """Save fitted statistics to file."""
        np.savez(
            path,
            feature_means=self._feature_means,
            feature_stds=self._feature_stds,
            regime_feature_means=self._regime_feature_means,
            regime_feature_stds=self._regime_feature_stds,
            feature_names=self._feature_names
        )
    
    def load_stats(self, path: str):
        """Load fitted statistics from file."""
        data = np.load(path, allow_pickle=True)
        self._feature_means = data['feature_means']
        self._feature_stds = data['feature_stds']
        self._regime_feature_means = data['regime_feature_means']
        self._regime_feature_stds = data['regime_feature_stds']
        self._feature_names = data['feature_names'].tolist()
