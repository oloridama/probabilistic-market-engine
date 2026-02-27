"""
Layer 1: Deterministic Physics Engine

Computes market physics features from OHLCV data using only deterministic
transformations. No ML, no probability, no labels.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema

from probabilistic_market_engine.config.settings import PhysicsConfig


@dataclass
class PhysicsState:
    """Container for computed physics features at time t."""
    timestamp: Optional[pd.Timestamp] = None
    
    # Normalized pressure features
    pressure_raw: float = 0.0
    pressure_norm: float = 0.0
    
    # Multi-scale flow features
    flow_short: float = 0.0  # Fast timescale
    flow_medium: float = 0.0  # Medium timescale
    flow_long: float = 0.0  # Slow timescale
    flow_alignment: float = 0.0  # Alignment across scales
    
    # Directional stability
    path_efficiency: float = 0.0
    directional_inertia: float = 0.0
    
    # Acceleration and convexity (path dependency)
    acceleration: float = 0.0
    convexity: float = 0.0
    
    # Compression index
    compression_index: float = 0.0
    range_efficiency: float = 0.0
    
    # Energy accumulation
    energy_kinetic: float = 0.0
    energy_potential: float = 0.0
    energy_total: float = 0.0
    
    # Shock metrics
    volatility_ratio: float = 1.0
    shock_index: float = 0.0
    flow_discontinuity: float = 0.0
    
    # Derived alignment score
    alignment_score: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector."""
        return np.array([
            self.pressure_norm,
            self.flow_short,
            self.flow_medium,
            self.flow_long,
            self.flow_alignment,
            self.path_efficiency,
            self.directional_inertia,
            self.acceleration,
            self.convexity,
            self.compression_index,
            self.range_efficiency,
            self.energy_kinetic,
            self.energy_potential,
            self.energy_total,
            self.volatility_ratio,
            self.shock_index,
            self.flow_discontinuity,
            self.alignment_score,
        ])
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            'timestamp': self.timestamp,
            'pressure_raw': self.pressure_raw,
            'pressure_norm': self.pressure_norm,
            'flow_short': self.flow_short,
            'flow_medium': self.flow_medium,
            'flow_long': self.flow_long,
            'flow_alignment': self.flow_alignment,
            'path_efficiency': self.path_efficiency,
            'directional_inertia': self.directional_inertia,
            'acceleration': self.acceleration,
            'convexity': self.convexity,
            'compression_index': self.compression_index,
            'range_efficiency': self.range_efficiency,
            'energy_kinetic': self.energy_kinetic,
            'energy_potential': self.energy_potential,
            'energy_total': self.energy_total,
            'volatility_ratio': self.volatility_ratio,
            'shock_index': self.shock_index,
            'flow_discontinuity': self.flow_discontinuity,
            'alignment_score': self.alignment_score,
        }


class PhysicsEngine:
    """
    Deterministic physics engine for computing market state features.
    
    This class is stateless except for controlled lookback buffers.
    All computations are deterministic transformations of OHLCV data.
    """
    
    def __init__(self, config: Optional[PhysicsConfig] = None):
        self.config = config or PhysicsConfig()
        
        # Buffers for stateful computations (controlled memory only)
        self._returns_buffer: List[float] = []
        self._pressure_buffer: List[float] = []
        self._flow_buffer: List[float] = []
        self._max_buffer_size = max(
            self.config.pressure_normalization_lookback,
            max(self.config.lookback_windows) * 2
        )
    
    def reset(self):
        """Clear all buffers."""
        self._returns_buffer.clear()
        self._pressure_buffer.clear()
        self._flow_buffer.clear()
    
    def compute(self, ohlcv_window: pd.DataFrame) -> PhysicsState:
        """
        Compute physics features from OHLCV window.
        
        Args:
            ohlcv_window: DataFrame with columns [open, high, low, close, volume]
                         Minimum length determined by lookback_windows
        
        Returns:
            PhysicsState containing all computed features
        """
        if len(ohlcv_window) < max(self.config.lookback_windows) + 1:
            raise ValueError(f"OHLCV window must have at least {max(self.config.lookback_windows) + 1} bars")
        
        state = PhysicsState()
        state.timestamp = ohlcv_window.index[-1] if isinstance(ohlcv_window.index, pd.DatetimeIndex) else None
        
        # Extract series
        opens = ohlcv_window['open'].values
        highs = ohlcv_window['high'].values
        lows = ohlcv_window['low'].values
        closes = ohlcv_window['close'].values
        volumes = ohlcv_window['volume'].values if 'volume' in ohlcv_window.columns else np.ones(len(ohlcv_window))
        
        # Compute returns
        returns = np.diff(closes) / closes[:-1]
        self._returns_buffer.extend(returns[-self._max_buffer_size:])
        self._returns_buffer = self._returns_buffer[-self._max_buffer_size:]
        
        # === 1. Normalized Pressure ===
        state.pressure_raw = self._compute_pressure(opens, highs, lows, closes, volumes)
        self._pressure_buffer.append(state.pressure_raw)
        self._pressure_buffer = self._pressure_buffer[-self.config.pressure_normalization_lookback:]
        state.pressure_norm = self._normalize_pressure(state.pressure_raw)
        
        # === 2. Multi-scale Flow ===
        state.flow_short, state.flow_medium, state.flow_long = self._compute_multi_scale_flow(returns)
        state.flow_alignment = self._compute_flow_alignment(state.flow_short, state.flow_medium, state.flow_long)
        
        # === 3. Directional Stability ===
        state.path_efficiency = self._compute_path_efficiency(returns)
        state.directional_inertia = self._compute_directional_inertia(returns)
        
        # === 4. Path Dependency (Acceleration, Convexity) ===
        state.acceleration, state.convexity = self._compute_path_dependency(returns)
        
        # === 5. Compression Index ===
        state.compression_index = self._compute_compression_index(highs, lows, closes)
        state.range_efficiency = self._compute_range_efficiency(highs, lows, closes)
        
        # === 6. Energy Accumulation ===
        state.energy_kinetic, state.energy_potential, state.energy_total = self._compute_energy(
            closes, returns, volumes
        )
        
        # === 7. Shock Metrics ===
        state.volatility_ratio = self._compute_volatility_ratio(returns)
        state.shock_index = self._compute_shock_index(returns, state.volatility_ratio)
        state.flow_discontinuity = self._compute_flow_discontinuity(returns)
        
        # === 8. Alignment Score ===
        state.alignment_score = self._compute_alignment_score(state)
        
        return state
    
    def _compute_pressure(self, opens: np.ndarray, highs: np.ndarray, 
                          lows: np.ndarray, closes: np.ndarray, 
                          volumes: np.ndarray) -> float:
        """
        Compute buying/selling pressure metric.
        Positive = buying pressure, Negative = selling pressure
        """
        # Position within bar
        bar_position = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1] + 1e-10)
        
        # Volume-weighted pressure
        volume_pressure = volumes[-1] * (2 * bar_position - 1)
        
        # Trend of close vs open
        direction = np.sign(closes[-1] - opens[-1])
        
        # Combine
        pressure = direction * volume_pressure * (abs(closes[-1] - opens[-1]) / (opens[-1] + 1e-10))
        
        # Clip extreme values
        return np.clip(pressure, -10, 10)
    
    def _normalize_pressure(self, pressure: float) -> float:
        """Normalize pressure using historical statistics."""
        if len(self._pressure_buffer) < self.config.min_samples_for_std:
            return 0.0
        
        buf = np.array(self._pressure_buffer)
        mean_p = np.mean(buf)
        std_p = np.std(buf) + 1e-10
        
        return (pressure - mean_p) / std_p
    
    def _compute_multi_scale_flow(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """Compute flow at multiple timescales."""
        windows = self.config.lookback_windows
        
        def compute_flow(ret_window: np.ndarray, window_size: int) -> float:
            if len(ret_window) < window_size:
                return 0.0
            recent = ret_window[-window_size:]
            # Signed momentum with trend persistence
            return np.sum(recent) / (np.std(recent) + 1e-10) * np.sqrt(window_size)
        
        flow_short = compute_flow(returns, windows[0])
        flow_medium = compute_flow(returns, windows[1])
        flow_long = compute_flow(returns, windows[2])
        
        return flow_short, flow_medium, flow_long
    
    def _compute_flow_alignment(self, flow_s: float, flow_m: float, flow_l: float) -> float:
        """Compute alignment of flows across timescales."""
        flows = np.array([flow_s, flow_m, flow_l])
        signs = np.sign(flows)
        
        # Check if all same sign
        if np.all(signs == signs[0]) and signs[0] != 0:
            # Alignment strength based on magnitude consistency
            return np.sign(flows[0]) * np.min(np.abs(flows)) / (np.max(np.abs(flows)) + 1e-10)
        return 0.0
    
    def _compute_path_efficiency(self, returns: np.ndarray) -> float:
        """
        Compute path efficiency: net displacement / total distance.
        1.0 = straight line, 0.0 = all noise
        """
        if len(returns) < self.config.path_memory_window:
            return 0.5
        
        recent = returns[-self.config.path_memory_window:]
        net_displacement = abs(np.sum(recent))
        total_distance = np.sum(np.abs(recent))
        
        if total_distance < 1e-10:
            return 0.5
        
        return net_displacement / total_distance
    
    def _compute_directional_inertia(self, returns: np.ndarray) -> float:
        """Compute tendency to continue in same direction."""
        if len(returns) < 4:
            return 0.0
        
        signs = np.sign(returns[-4:])
        # Count consecutive same-sign returns
        inertia = 0.0
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1] and signs[i] != 0:
                inertia += 1.0
        
        return inertia / 3.0 - 0.5  # Center around 0
    
    def _compute_path_dependency(self, returns: np.ndarray) -> Tuple[float, float]:
        """Compute acceleration and convexity."""
        if len(returns) < 3:
            return 0.0, 0.0
        
        # Acceleration: change in returns
        r = returns[-3:]
        acceleration = (r[-1] - r[-2]) - (r[-2] - r[-1]) if len(r) >= 3 else 0.0
        
        # Convexity: curvature of price path
        if len(returns) >= 4:
            cumulative = np.cumsum(returns[-4:])
            # Second derivative approximation
            convexity = (cumulative[-1] - 2*cumulative[-2] + cumulative[-3])
        else:
            convexity = 0.0
        
        # Normalize by volatility
        std_ret = np.std(returns) + 1e-10
        return acceleration / std_ret, convexity / std_ret
    
    def _compute_compression_index(self, highs: np.ndarray, lows: np.ndarray, 
                                   closes: np.ndarray) -> float:
        """
        Compute compression index: low volatility relative to recent range.
        High compression = potential for explosive move
        """
        if len(highs) < 20:
            return 0.5
        
        recent_range = np.max(highs[-20:]) - np.min(lows[-20:])
        current_range = highs[-1] - lows[-1]
        
        if recent_range < 1e-10:
            return 0.5
        
        compression = 1.0 - (current_range / recent_range)
        return np.clip(compression, 0.0, 1.0)
    
    def _compute_range_efficiency(self, highs: np.ndarray, lows: np.ndarray, 
                                  closes: np.ndarray) -> float:
        """Compute how efficiently price used its range."""
        if len(highs) < 2:
            return 0.5
        
        bar_range = highs[-1] - lows[-1]
        body = abs(closes[-1] - closes[-2])
        
        if bar_range < 1e-10:
            return 0.5
        
        return body / bar_range
    
    def _compute_energy(self, closes: np.ndarray, returns: np.ndarray, 
                        volumes: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute energy metrics:
        - Kinetic: velocity * mass (return * volume)
        - Potential: stored energy (compression)
        - Total: combination
        """
        if len(returns) < self.config.energy_accumulation_window:
            return 0.0, 0.0, 0.0
        
        recent_returns = returns[-self.config.energy_accumulation_window:]
        recent_volumes = volumes[-self.config.energy_accumulation_window:]
        
        # Kinetic energy: velocity squared times mass
        velocity = np.mean(recent_returns) * np.sqrt(self.config.energy_accumulation_window)
        mass = np.mean(recent_volumes) / (np.std(recent_volumes) + 1e-10)
        kinetic = velocity ** 2 * np.sign(velocity) * np.sqrt(mass)
        
        # Potential energy: stored in range compression
        if len(closes) >= self.config.energy_accumulation_window:
            price_range = np.max(closes[-self.config.energy_accumulation_window:]) - np.min(closes[-self.config.energy_accumulation_window:])
            potential = price_range / (np.std(recent_returns) * np.sqrt(self.config.energy_accumulation_window) + 1e-10)
        else:
            potential = 0.0
        
        total = np.sign(kinetic) * (abs(kinetic) + potential)
        
        # Clip
        kinetic = np.clip(kinetic, -100, 100)
        potential = np.clip(potential, 0, 100)
        total = np.clip(total, -100, 100)
        
        return kinetic, potential, total
    
    def _compute_volatility_ratio(self, returns: np.ndarray) -> float:
        """
        Compute ratio of short-term to longer-term volatility.
        High ratio indicates regime change or shock.
        """
        if len(returns) < 64:
            return 1.0
        
        short_vol = np.std(returns[-4:]) + 1e-10
        medium_vol = np.std(returns[-16:]) + 1e-10
        long_vol = np.std(returns[-64:]) + 1e-10
        
        ratio = short_vol / long_vol
        return np.clip(ratio, 0.1, 10.0)
    
    def _compute_shock_index(self, returns: np.ndarray, vol_ratio: float) -> float:
        """
        Compute shock index: measure of extreme move relative to recent history.
        """
        if len(returns) < 20:
            return 0.0
        
        current_return = returns[-1]
        historical_vol = np.std(returns[-20:]) + 1e-10
        
        # Z-score of current return
        z_score = current_return / historical_vol
        
        # Shock index combines magnitude and volatility regime change
        shock = abs(z_score) * vol_ratio
        
        # Apply threshold
        shock = max(0, shock - self.config.shock_detection_threshold)
        
        return np.clip(shock, 0, 10)
    
    def _compute_flow_discontinuity(self, returns: np.ndarray) -> float:
        """
        Detect discontinuities in flow pattern.
        """
        if len(returns) < 8:
            return 0.0
        
        # Compare recent flow to historical flow
        recent_flow = np.mean(returns[-4:])
        historical_flow = np.mean(returns[-8:-4])
        
        recent_vol = np.std(returns[-4:]) + 1e-10
        historical_vol = np.std(returns[-8:-4]) + 1e-10
        
        # Discontinuity in mean
        mean_shift = abs(recent_flow - historical_flow) / historical_vol
        
        # Discontinuity in volatility
        vol_shift = abs(recent_vol - historical_vol) / historical_vol
        
        discontinuity = (mean_shift + vol_shift) / 2
        
        return np.clip(discontinuity, 0, 5)
    
    def _compute_alignment_score(self, state: PhysicsState) -> float:
        """
        Compute overall alignment score combining multiple features.
        """
        components = [
            state.flow_alignment,
            state.path_efficiency * np.sign(state.flow_medium) if abs(state.flow_medium) > 0.5 else 0,
            (1 - state.compression_index) * np.sign(state.flow_short) if abs(state.flow_short) > 1 else 0,
        ]
        
        # Average non-zero components
        non_zero = [c for c in components if abs(c) > 0.01]
        if not non_zero:
            return 0.0
        
        return np.clip(np.mean(non_zero), -1, 1)


# Backward compatibility
PhysicsFeatures = PhysicsState
