"""
Application settings and configuration management.
Production-grade configuration with environment variable support.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class PhysicsConfig:
    """Configuration for Physics Engine."""
    lookback_windows: List[int] = field(default_factory=lambda: [4, 16, 64])  # 1h, 4h, 16h
    pressure_normalization_lookback: int = 256
    energy_accumulation_window: int = 16
    shock_detection_threshold: float = 2.5
    path_memory_window: int = 8


@dataclass
class LiquidityConfig:
    """Configuration for Liquidity Topology Engine."""
    kde_bandwidth: float = 0.02
    historical_lookback: int = 1000
    time_decay_factor: float = 0.995
    min_pivots_required: int = 10
    pivot_prominence: float = 0.01


@dataclass
class FeaturePipelineConfig:
    """Configuration for Feature Pipeline."""
    rolling_window: int = 64
    standardization_lookback: int = 256
    min_samples_for_std: int = 30
    feature_subset_regime: List[str] = field(default_factory=lambda: [
        'pressure_norm', 'alignment_score', 'path_efficiency', 
        'volatility_ratio', 'shock_index'
    ])


@dataclass
class RegimeModelConfig:
    """Configuration for Bayesian Regime Inference."""
    n_regimes: int = 3
    regime_labels: List[str] = field(default_factory=lambda: ['trend', 'range', 'shock'])
    persistence_alpha: float = 0.7
    random_state: int = 42
    max_iter: int = 100
    n_init: int = 10


@dataclass
class OutcomeModelConfig:
    """Configuration for Regime-Conditional Outcome Experts."""
    prediction_horizon: int = 4  # 1 hour forward (4 * 15min)
    continuation_threshold: float = 0.001  # 0.1% move
    max_iter: int = 1000
    calibration_method: str = 'isotonic'  # 'isotonic', 'platt', None


@dataclass
class RiskEngineConfig:
    """Configuration for Risk Adjustment Engine."""
    base_risk_scale: float = 1.0
    max_risk_scale: float = 2.0
    min_risk_scale: float = 0.0
    shock_penalty_factor: float = 2.0
    uncertainty_penalty_factor: float = 1.5


@dataclass
class TrainingConfig:
    """Configuration for Training Framework."""
    walkforward_train_size: int = 2000  # ~20 days of 15min bars
    walkforward_test_size: int = 500   # ~5 days
    walkforward_step_size: int = 250
    min_train_samples: int = 500
    calibration_window: int = 500


@dataclass
class MonitoringConfig:
    """Configuration for Monitoring Framework."""
    drift_window_short: int = 100
    drift_window_long: int = 500
    brier_score_window: int = 250
    regime_flip_threshold: int = 5  # flips per window
    feature_drift_threshold: float = 2.0  # standard deviations
    alert_cooldown_periods: int = 10


@dataclass
class PersistenceConfig:
    """Configuration for Persistence Layer."""
    model_registry_path: str = "probabilistic_market_engine/persistence/model_registry"
    feature_store_path: str = "app/persistence/feature_store"
    logs_path: str = "app/persistence/logs"
    max_models_to_keep: int = 10


@dataclass
class APIConfig:
    """Configuration for API Server."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    request_timeout: int = 30
    max_ohlcv_lookback: int = 256


class Settings:
    """
    Centralized settings manager.
    Loads from YAML config file and environment variables.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load YAML config if exists
        if config_path is None:
            config_path = self.project_root / "probabilistic_market_engine" / "config" / "model_config.yaml"
        
        self._config_data = self._load_yaml(config_path)
        
        # Initialize all config sections
        self.physics = self._init_physics_config()
        self.liquidity = self._init_liquidity_config()
        self.feature_pipeline = self._init_feature_pipeline_config()
        self.regime_model = self._init_regime_model_config()
        self.outcome_model = self._init_outcome_model_config()
        self.risk_engine = self._init_risk_engine_config()
        self.training = self._init_training_config()
        self.monitoring = self._init_monitoring_config()
        self.persistence = self._init_persistence_config()
        self.api = self._init_api_config()
        
        # Model versioning
        self.model_version = os.getenv("MODEL_VERSION", "v1.0.0")
        self.environment = os.getenv("ENVIRONMENT", "development")
    
    def _load_yaml(self, path: Path) -> dict:
        """Load YAML configuration file."""
        if path.exists():
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _get_env_or_config(self, key: str, default=None, section: str = None):
        """Get value from environment variable or config dict."""
        env_key = key.upper() if section is None else f"{section.upper()}_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Try to convert to appropriate type
            if env_value.lower() in ('true', 'false'):
                return env_value.lower() == 'true'
            try:
                if '.' in env_value:
                    return float(env_value)
                return int(env_value)
            except ValueError:
                return env_value
        
        if section and section in self._config_data:
            return self._config_data[section].get(key, default)
        return self._config_data.get(key, default)
    
    def _init_physics_config(self) -> PhysicsConfig:
        return PhysicsConfig(
            lookback_windows=self._get_env_or_config('lookback_windows', [4, 16, 64], 'physics'),
            pressure_normalization_lookback=self._get_env_or_config('pressure_normalization_lookback', 256, 'physics'),
            energy_accumulation_window=self._get_env_or_config('energy_accumulation_window', 16, 'physics'),
            shock_detection_threshold=self._get_env_or_config('shock_detection_threshold', 2.5, 'physics'),
            path_memory_window=self._get_env_or_config('path_memory_window', 8, 'physics'),
        )
    
    def _init_liquidity_config(self) -> LiquidityConfig:
        return LiquidityConfig(
            kde_bandwidth=self._get_env_or_config('kde_bandwidth', 0.02, 'liquidity'),
            historical_lookback=self._get_env_or_config('historical_lookback', 1000, 'liquidity'),
            time_decay_factor=self._get_env_or_config('time_decay_factor', 0.995, 'liquidity'),
            min_pivots_required=self._get_env_or_config('min_pivots_required', 10, 'liquidity'),
            pivot_prominence=self._get_env_or_config('pivot_prominence', 0.01, 'liquidity'),
        )
    
    def _init_feature_pipeline_config(self) -> FeaturePipelineConfig:
        return FeaturePipelineConfig(
            rolling_window=self._get_env_or_config('rolling_window', 64, 'feature_pipeline'),
            standardization_lookback=self._get_env_or_config('standardization_lookback', 256, 'feature_pipeline'),
            min_samples_for_std=self._get_env_or_config('min_samples_for_std', 30, 'feature_pipeline'),
            feature_subset_regime=self._get_env_or_config('feature_subset_regime', [
                'pressure_norm', 'alignment_score', 'path_efficiency',
                'volatility_ratio', 'shock_index'
            ], 'feature_pipeline'),
        )
    
    def _init_regime_model_config(self) -> RegimeModelConfig:
        return RegimeModelConfig(
            n_regimes=self._get_env_or_config('n_regimes', 3, 'regime_model'),
            regime_labels=self._get_env_or_config('regime_labels', ['trend', 'range', 'shock'], 'regime_model'),
            persistence_alpha=self._get_env_or_config('persistence_alpha', 0.7, 'regime_model'),
            random_state=self._get_env_or_config('random_state', 42, 'regime_model'),
            max_iter=self._get_env_or_config('max_iter', 100, 'regime_model'),
            n_init=self._get_env_or_config('n_init', 10, 'regime_model'),
        )
    
    def _init_outcome_model_config(self) -> OutcomeModelConfig:
        return OutcomeModelConfig(
            prediction_horizon=self._get_env_or_config('prediction_horizon', 4, 'outcome_model'),
            continuation_threshold=self._get_env_or_config('continuation_threshold', 0.001, 'outcome_model'),
            max_iter=self._get_env_or_config('max_iter', 1000, 'outcome_model'),
            calibration_method=self._get_env_or_config('calibration_method', 'isotonic', 'outcome_model'),
        )
    
    def _init_risk_engine_config(self) -> RiskEngineConfig:
        return RiskEngineConfig(
            base_risk_scale=self._get_env_or_config('base_risk_scale', 1.0, 'risk_engine'),
            max_risk_scale=self._get_env_or_config('max_risk_scale', 2.0, 'risk_engine'),
            min_risk_scale=self._get_env_or_config('min_risk_scale', 0.0, 'risk_engine'),
            shock_penalty_factor=self._get_env_or_config('shock_penalty_factor', 2.0, 'risk_engine'),
            uncertainty_penalty_factor=self._get_env_or_config('uncertainty_penalty_factor', 1.5, 'risk_engine'),
        )
    
    def _init_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            walkforward_train_size=self._get_env_or_config('walkforward_train_size', 2000, 'training'),
            walkforward_test_size=self._get_env_or_config('walkforward_test_size', 500, 'training'),
            walkforward_step_size=self._get_env_or_config('walkforward_step_size', 250, 'training'),
            min_train_samples=self._get_env_or_config('min_train_samples', 500, 'training'),
            calibration_window=self._get_env_or_config('calibration_window', 500, 'training'),
        )
    
    def _init_monitoring_config(self) -> MonitoringConfig:
        return MonitoringConfig(
            drift_window_short=self._get_env_or_config('drift_window_short', 100, 'monitoring'),
            drift_window_long=self._get_env_or_config('drift_window_long', 500, 'monitoring'),
            brier_score_window=self._get_env_or_config('brier_score_window', 250, 'monitoring'),
            regime_flip_threshold=self._get_env_or_config('regime_flip_threshold', 5, 'monitoring'),
            feature_drift_threshold=self._get_env_or_config('feature_drift_threshold', 2.0, 'monitoring'),
            alert_cooldown_periods=self._get_env_or_config('alert_cooldown_periods', 10, 'monitoring'),
        )
    
    def _init_persistence_config(self) -> PersistenceConfig:
        return PersistenceConfig(
            model_registry_path=self._get_env_or_config('model_registry_path', 'probabilistic_market_engine/persistence/model_registry', 'persistence'),
            feature_store_path=self._get_env_or_config('feature_store_path', 'probabilistic_market_engine/persistence/feature_store', 'persistence'),
            logs_path=self._get_env_or_config('logs_path', 'probabilistic_market_engine/persistence/logs', 'persistence'),
            max_models_to_keep=self._get_env_or_config('max_models_to_keep', 10, 'persistence'),
        )
    
    def _init_api_config(self) -> APIConfig:
        return APIConfig(
            host=self._get_env_or_config('host', '0.0.0.0', 'api'),
            port=self._get_env_or_config('port', 8000, 'api'),
            debug=self._get_env_or_config('debug', False, 'api'),
            request_timeout=self._get_env_or_config('request_timeout', 30, 'api'),
            max_ohlcv_lookback=self._get_env_or_config('max_ohlcv_lookback', 256, 'api'),
        )
    
    def to_dict(self) -> Dict:
        """Export all settings to dictionary for logging."""
        return {
            'model_version': self.model_version,
            'environment': self.environment,
            'physics': self.physics.__dict__,
            'liquidity': self.liquidity.__dict__,
            'feature_pipeline': self.feature_pipeline.__dict__,
            'regime_model': self.regime_model.__dict__,
            'outcome_model': self.outcome_model.__dict__,
            'risk_engine': self.risk_engine.__dict__,
            'training': self.training.__dict__,
            'monitoring': self.monitoring.__dict__,
            'persistence': self.persistence.__dict__,
            'api': self.api.__dict__,
        }


# Global settings instance
settings = Settings()
