"""
Regime Tracking Module

Tracks:
- Regime probabilities over time
- Regime flip frequency
- Dominant regime persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque, Counter
import logging

from probabilistic_market_engine.config.settings import MonitoringConfig
from probabilistic_market_engine.core.regime_model.model import RegimeState


class RegimeTracker:
    """
    Tracks regime statistics over time.
    
    Monitors:
    - Regime flip frequency
    - Regime persistence
    - Regime distribution stability
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # Tracking buffers
        self._regime_history: deque = deque(maxlen=self.config.drift_window_long)
        self._flip_count: int = 0
        self._last_regime: Optional[str] = None
        
        # Statistics
        self._regime_durations: Dict[str, List[int]] = {}
        self._current_duration: int = 0
    
    def update(self, regime_state: RegimeState):
        """
        Update tracker with new regime state.
        
        Args:
            regime_state: Current regime state
        """
        current_regime = regime_state.dominant_regime
        
        # Track flip
        if self._last_regime is not None and current_regime != self._last_regime:
            self._flip_count += 1
            
            # Record previous regime duration
            if self._last_regime not in self._regime_durations:
                self._regime_durations[self._last_regime] = []
            self._regime_durations[self._last_regime].append(self._current_duration)
            
            self._current_duration = 0
        
        self._current_duration += 1
        self._last_regime = current_regime
        
        # Store state
        self._regime_history.append({
            'timestamp': regime_state.timestamp,
            'dominant_regime': regime_state.dominant_regime,
            'trend_prob': regime_state.trend_probability,
            'range_prob': regime_state.range_probability,
            'shock_prob': regime_state.shock_probability,
            'entropy': regime_state.regime_entropy,
            'confidence': regime_state.confidence,
        })
    
    def get_flip_frequency(self, window: Optional[int] = None) -> float:
        """
        Get regime flip frequency.
        
        Args:
            window: Lookback window (default: drift_window_short)
        
        Returns:
            Flips per period
        """
        window = window or self.config.drift_window_short
        
        if len(self._regime_history) < 2:
            return 0.0
        
        # Count flips in window
        recent = list(self._regime_history)[-window:]
        flips = 0
        last = recent[0]['dominant_regime']
        
        for r in recent[1:]:
            if r['dominant_regime'] != last:
                flips += 1
            last = r['dominant_regime']
        
        return flips / len(recent)
    
    def is_flip_frequency_anomalous(self) -> bool:
        """Check if flip frequency is above threshold."""
        freq = self.get_flip_frequency()
        return freq > self.config.regime_flip_threshold / self.config.drift_window_short
    
    def get_regime_distribution(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        Get regime distribution over recent window.
        
        Args:
            window: Lookback window (default: all history)
        
        Returns:
            Dict mapping regime to frequency
        """
        if len(self._regime_history) == 0:
            return {}
        
        window = window or len(self._regime_history)
        recent = list(self._regime_history)[-window:]
        
        regimes = [r['dominant_regime'] for r in recent]
        counts = Counter(regimes)
        total = len(regimes)
        
        return {regime: count / total for regime, count in counts.items()}
    
    def get_average_entropy(self, window: Optional[int] = None) -> float:
        """Get average regime entropy."""
        if len(self._regime_history) == 0:
            return 1.0
        
        window = window or len(self._regime_history)
        recent = list(self._regime_history)[-window:]
        
        return np.mean([r['entropy'] for r in recent])
    
    def get_regime_durations(self, regime: str) -> Dict[str, float]:
        """Get duration statistics for a regime."""
        durations = self._regime_durations.get(regime, [])
        
        if not durations:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        return {
            'mean': float(np.mean(durations)),
            'std': float(np.std(durations)),
            'min': int(np.min(durations)),
            'max': int(np.max(durations)),
            'count': len(durations),
        }
    
    def get_current_regime_stats(self) -> Dict:
        """Get statistics for current regime."""
        if len(self._regime_history) == 0:
            return {}
        
        current = self._regime_history[-1]
        
        return {
            'dominant_regime': current['dominant_regime'],
            'duration': self._current_duration,
            'trend_prob': current['trend_prob'],
            'range_prob': current['range_prob'],
            'shock_prob': current['shock_prob'],
            'entropy': current['entropy'],
            'confidence': current['confidence'],
        }
    
    def get_summary(self) -> Dict:
        """Get summary of regime tracking."""
        return {
            'flip_frequency': self.get_flip_frequency(),
            'flip_anomaly': self.is_flip_frequency_anomalous(),
            'regime_distribution': self.get_regime_distribution(),
            'average_entropy': self.get_average_entropy(),
            'current_regime': self.get_current_regime_stats(),
            'regime_durations': {
                regime: self.get_regime_durations(regime)
                for regime in ['trend', 'range', 'shock']
            },
        }
    
    def reset(self):
        """Reset tracker state."""
        self._regime_history.clear()
        self._flip_count = 0
        self._last_regime = None
        self._regime_durations.clear()
        self._current_duration = 0
