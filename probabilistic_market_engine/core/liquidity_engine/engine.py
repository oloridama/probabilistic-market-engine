"""
Layer 2: Liquidity Topology Engine

Maintains rolling structural density map of price interaction zones.
Uses Kernel Density Estimation of historical turning points.
NO labels. NO outcome awareness.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

from probabilistic_market_engine.config.settings import LiquidityConfig


@dataclass
class LiquidityState:
    """Container for computed liquidity topology features."""
    timestamp: Optional[pd.Timestamp] = None
    
    # Distance to nearest liquidity cluster
    distance_to_support: float = 1.0
    distance_to_resistance: float = 1.0
    
    # Spatial resistance scalar (density of liquidity at current price)
    liquidity_density: float = 0.5
    
    # Relative position within liquidity structure
    relative_position: float = 0.5
    
    # Cluster strength metrics
    support_strength: float = 0.0
    resistance_strength: float = 0.0
    
    # Time-decayed turnover at current level
    turnover_intensity: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector."""
        return np.array([
            self.distance_to_support,
            self.distance_to_resistance,
            self.liquidity_density,
            self.relative_position,
            self.support_strength,
            self.resistance_strength,
            self.turnover_intensity,
        ])
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary."""
        return {
            'timestamp': self.timestamp,
            'distance_to_support': self.distance_to_support,
            'distance_to_resistance': self.distance_to_resistance,
            'liquidity_density': self.liquidity_density,
            'relative_position': self.relative_position,
            'support_strength': self.support_strength,
            'resistance_strength': self.resistance_strength,
            'turnover_intensity': self.turnover_intensity,
        }


class LiquidityEngine:
    """
    Liquidity topology engine for mapping price interaction zones.
    
    This class maintains a rolling KDE of historical turning points (pivots)
    to identify support/resistance levels and liquidity clusters.
    """
    
    def __init__(self, config: Optional[LiquidityConfig] = None):
        self.config = config or LiquidityConfig()
        
        # Rolling pivot storage with time decay
        self._pivots: List[Tuple[float, float]] = []  # (price, weight)
        self._pivot_ages: List[int] = []
        
        # KDE cache
        self._kde: Optional[gaussian_kde] = None
        self._kde_invalid: bool = True
        
        # Price range cache
        self._price_min: float = 0.0
        self._price_max: float = 0.0
    
    def reset(self):
        """Clear all stored data."""
        self._pivots.clear()
        self._pivot_ages.clear()
        self._kde = None
        self._kde_invalid = True
        self._price_min = 0.0
        self._price_max = 0.0
    
    def compute(self, ohlcv_window: pd.DataFrame) -> LiquidityState:
        """
        Compute liquidity topology features from OHLCV window.
        
        Args:
            ohlcv_window: DataFrame with columns [open, high, low, close, volume]
        
        Returns:
            LiquidityState containing liquidity topology features
        """
        state = LiquidityState()
        state.timestamp = ohlcv_window.index[-1] if isinstance(ohlcv_window.index, pd.DatetimeIndex) else None
        
        # Extract price data
        highs = ohlcv_window['high'].values
        lows = ohlcv_window['low'].values
        closes = ohlcv_window['close'].values
        current_price = closes[-1]
        
        # Update price range
        self._price_min = min(np.min(lows), self._price_min) if self._price_min > 0 else np.min(lows)
        self._price_max = max(np.max(highs), self._price_max) if self._price_max > 0 else np.max(highs)
        
        # Detect and add new pivots
        self._update_pivots(highs, lows, closes)
        
        # Apply time decay
        self._apply_time_decay()
        
        # Compute features
        if len(self._pivots) >= self.config.min_pivots_required:
            self._rebuild_kde_if_needed()
            
            distances = self._compute_distance_metrics(current_price)
            state.distance_to_support = distances['support']
            state.distance_to_resistance = distances['resistance']
            
            state.liquidity_density = self._compute_liquidity_density(current_price)
            state.relative_position = self._compute_relative_position(current_price)
            
            strengths = self._compute_cluster_strengths(current_price)
            state.support_strength = strengths['support']
            state.resistance_strength = strengths['resistance']
        else:
            # Not enough data - neutral values
            state.distance_to_support = 1.0
            state.distance_to_resistance = 1.0
            state.liquidity_density = 0.5
            state.relative_position = 0.5
            state.support_strength = 0.0
            state.resistance_strength = 0.0
        
        # Compute turnover intensity
        state.turnover_intensity = self._compute_turnover_intensity(ohlcv_window)
        
        return state
    
    def _update_pivots(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray):
        """Detect and add new pivot points."""
        # Need at least 3 bars to detect pivots
        if len(highs) < 3:
            return
        
        order = 1  # Minimum bars on each side
        
        # Detect local maxima in highs (resistance pivots)
        max_indices = argrelextrema(highs, np.greater, order=order)[0]
        for idx in max_indices:
            if idx >= len(highs) - 5:  # Only recent pivots
                price = highs[idx]
                prominence = self._compute_prominence(highs, lows, idx, is_high=True)
                if prominence >= self.config.pivot_prominence:
                    self._add_pivot(price, prominence)
        
        # Detect local minima in lows (support pivots)
        min_indices = argrelextrema(lows, np.less, order=order)[0]
        for idx in min_indices:
            if idx >= len(lows) - 5:  # Only recent pivots
                price = lows[idx]
                prominence = self._compute_prominence(highs, lows, idx, is_high=False)
                if prominence >= self.config.pivot_prominence:
                    self._add_pivot(price, prominence)
    
    def _compute_prominence(self, highs: np.ndarray, lows: np.ndarray, 
                           idx: int, is_high: bool) -> float:
        """Compute prominence of a pivot point."""
        current_price = highs[idx] if is_high else lows[idx]
        
        # Look at neighboring bars to determine prominence
        left_start = max(0, idx - 5)
        right_end = min(len(highs), idx + 6)
        
        if is_high:
            neighbors = np.concatenate([highs[left_start:idx], highs[idx+1:right_end]])
            if len(neighbors) > 0:
                return max(0, current_price - np.max(neighbors)) / current_price
        else:
            neighbors = np.concatenate([lows[left_start:idx], lows[idx+1:right_end]])
            if len(neighbors) > 0:
                return max(0, np.min(neighbors) - current_price) / current_price
        
        return 0.0
    
    def _add_pivot(self, price: float, weight: float):
        """Add a new pivot point."""
        # Check if similar pivot exists (within bandwidth)
        bandwidth = self.config.kde_bandwidth * price
        for i, (p, w) in enumerate(self._pivots):
            if abs(p - price) < bandwidth:
                # Merge with existing
                self._pivots[i] = (price, max(w, weight))
                self._pivot_ages[i] = 0
                self._kde_invalid = True
                return
        
        # Add new pivot
        self._pivots.append((price, weight))
        self._pivot_ages.append(0)
        self._kde_invalid = True
        
        # Trim old pivots
        if len(self._pivots) > self.config.historical_lookback:
            # Remove oldest
            oldest_idx = np.argmax(self._pivot_ages)
            self._pivots.pop(oldest_idx)
            self._pivot_ages.pop(oldest_idx)
            self._kde_invalid = True
    
    def _apply_time_decay(self):
        """Apply time decay to pivot weights."""
        decay = self.config.time_decay_factor
        
        for i in range(len(self._pivots)):
            self._pivot_ages[i] += 1
            price, weight = self._pivots[i]
            # Apply decay to weight
            new_weight = weight * decay
            self._pivots[i] = (price, new_weight)
        
        # Remove heavily decayed pivots
        threshold = 0.01
        active_pivots = [(p, w) for p, w in self._pivots if w > threshold]
        active_ages = [a for i, a in enumerate(self._pivot_ages) if self._pivots[i][1] > threshold]
        
        if len(active_pivots) != len(self._pivots):
            self._pivots = active_pivots
            self._pivot_ages = active_ages
            self._kde_invalid = True
    
    def _rebuild_kde_if_needed(self):
        """Rebuild KDE if pivots have changed."""
        if not self._kde_invalid or len(self._pivots) < self.config.min_pivots_required:
            return
        
        prices = np.array([p for p, w in self._pivots])
        weights = np.array([w for p, w in self._pivots])
        
        # Normalize weights
        weights = weights / (weights.sum() + 1e-10)
        
        # Compute adaptive bandwidth based on price range
        price_range = prices.max() - prices.min()
        bandwidth = max(self.config.kde_bandwidth * np.mean(prices), 
                       price_range / len(prices))
        
        try:
            self._kde = gaussian_kde(prices, weights=weights, bw_method=bandwidth / np.std(prices))
            self._kde_invalid = False
        except Exception:
            # Fall back to simpler KDE
            self._kde = None
            self._kde_invalid = True
    
    def _compute_distance_metrics(self, current_price: float) -> Dict[str, float]:
        """Compute distance to nearest support and resistance."""
        if not self._pivots:
            return {'support': 1.0, 'resistance': 1.0}
        
        prices = np.array([p for p, w in self._pivots])
        weights = np.array([w for p, w in self._pivots])
        
        # Normalize prices
        price_range = self._price_max - self._price_min
        if price_range < 1e-10:
            return {'support': 1.0, 'resistance': 1.0}
        
        # Find support (pivots below current price)
        support_mask = prices < current_price
        if np.any(support_mask):
            support_prices = prices[support_mask]
            support_weights = weights[support_mask]
            # Weighted distance to nearest support
            distances = (current_price - support_prices) / price_range
            nearest_idx = np.argmin(distances)
            dist_to_support = distances[nearest_idx]
            support_strength = support_weights[nearest_idx]
        else:
            dist_to_support = 1.0
            support_strength = 0.0
        
        # Find resistance (pivots above current price)
        resistance_mask = prices > current_price
        if np.any(resistance_mask):
            resistance_prices = prices[resistance_mask]
            resistance_weights = weights[resistance_mask]
            distances = (resistance_prices - current_price) / price_range
            nearest_idx = np.argmin(distances)
            dist_to_resistance = distances[nearest_idx]
            resistance_strength = resistance_weights[nearest_idx]
        else:
            dist_to_resistance = 1.0
            resistance_strength = 0.0
        
        # Store strengths for later
        self._temp_support_strength = support_strength
        self._temp_resistance_strength = resistance_strength
        
        return {
            'support': np.clip(dist_to_support, 0, 1),
            'resistance': np.clip(dist_to_resistance, 0, 1)
        }
    
    def _compute_liquidity_density(self, current_price: float) -> float:
        """Compute liquidity density at current price using KDE."""
        if self._kde is None or self._kde_invalid:
            # Fallback: count nearby pivots
            prices = np.array([p for p, w in self._pivots])
            bandwidth = self.config.kde_bandwidth * current_price
            nearby = np.sum(np.abs(prices - current_price) < bandwidth)
            return np.clip(nearby / self.config.min_pivots_required, 0, 1)
        
        try:
            density = self._kde.evaluate(current_price)[0]
            # Normalize density
            prices = np.array([p for p, w in self._pivots])
            max_density = np.max(self._kde.evaluate(prices))
            return np.clip(density / (max_density + 1e-10), 0, 1)
        except Exception:
            return 0.5
    
    def _compute_relative_position(self, current_price: float) -> float:
        """Compute relative position within liquidity structure."""
        if self._price_max <= self._price_min:
            return 0.5
        
        # Position within historical range
        position = (current_price - self._price_min) / (self._price_max - self._price_min)
        return np.clip(position, 0, 1)
    
    def _compute_cluster_strengths(self, current_price: float) -> Dict[str, float]:
        """Compute support and resistance cluster strengths."""
        # Use values computed in distance_metrics
        return {
            'support': getattr(self, '_temp_support_strength', 0.0),
            'resistance': getattr(self, '_temp_resistance_strength', 0.0)
        }
    
    def _compute_turnover_intensity(self, ohlcv_window: pd.DataFrame) -> float:
        """Compute time-decayed turnover at current price level."""
        if 'volume' not in ohlcv_window.columns or len(ohlcv_window) < 2:
            return 0.0
        
        volumes = ohlcv_window['volume'].values
        closes = ohlcv_window['close'].values
        current_price = closes[-1]
        
        # Weight recent volume more heavily
        weights = np.exp(-0.1 * np.arange(len(volumes))[::-1])
        weights = weights / weights.sum()
        
        # Volume intensity relative to recent average
        avg_volume = np.mean(volumes)
        weighted_volume = np.sum(volumes * weights)
        
        intensity = weighted_volume / (avg_volume + 1e-10) - 1.0
        
        return np.clip(intensity, -1, 1)


# Backward compatibility
LiquidityFeatures = LiquidityState
