"""
Feature Store

Stores and retrieves computed features for training and inference.
"""

import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np


class FeatureStore:
    """
    Store for persisting computed features.
    
    Supports:
    - Feature caching
    - Versioned storage
    - Retrieval by date range
    - Metadata tracking
    """
    
    def __init__(self, store_path: Optional[str] = None):
        """
        Initialize feature store.
        
        Args:
            store_path: Path to store features.
                       Defaults to ~/.probabilistic_market_engine/features
        """
        if store_path is None:
            home = Path.home()
            store_path = home / ".probabilistic_market_engine" / "features"
        
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Index of stored feature sets
        self._index_file = self.store_path / "feature_index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load feature index from disk."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                return json.load(f)
        return {"feature_sets": {}}
    
    def _save_index(self):
        """Save feature index to disk."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f, indent=2, default=str)
    
    def store_features(
        self,
        name: str,
        feature_sets: List[Any],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store computed features.
        
        Args:
            name: Identifier for this feature set
            feature_sets: List of FeatureSet objects
            metadata: Additional metadata
            
        Returns:
            storage_key: Key to retrieve these features
        """
        storage_key = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        feature_path = self.store_path / f"{storage_key}.pkl"
        
        # Extract feature arrays for storage
        X_features = np.array([fs.X_t for fs in feature_sets])
        R_features = np.array([fs.R_t for fs in feature_sets])
        timestamps = [fs.timestamp for fs in feature_sets if hasattr(fs, 'timestamp')]
        
        feature_data = {
            "X_features": X_features,
            "R_features": R_features,
            "timestamps": timestamps,
            "n_samples": len(feature_sets),
            "X_shape": X_features.shape,
            "R_shape": R_features.shape
        }
        
        # Save to disk
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_data, f)
        
        # Update index
        self._index["feature_sets"][storage_key] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "n_samples": len(feature_sets),
            "metadata": metadata or {},
            "file": str(feature_path.relative_to(self.store_path))
        }
        self._save_index()
        
        return storage_key
    
    def load_features(self, storage_key: str) -> Dict[str, Any]:
        """
        Load stored features.
        
        Args:
            storage_key: Key returned by store_features
            
        Returns:
            Dict with keys: X_features, R_features, timestamps
        """
        if storage_key not in self._index["feature_sets"]:
            raise ValueError(f"Feature set {storage_key} not found")
        
        info = self._index["feature_sets"][storage_key]
        feature_path = self.store_path / info["file"]
        
        with open(feature_path, 'rb') as f:
            feature_data = pickle.load(f)
        
        return feature_data
    
    def list_feature_sets(self, name_filter: Optional[str] = None) -> Dict[str, Dict]:
        """
        List stored feature sets.
        
        Args:
            name_filter: Optional string to filter by name
            
        Returns:
            Dict of storage_key -> info
        """
        result = {}
        for key, info in self._index["feature_sets"].items():
            if name_filter is None or name_filter in info["name"]:
                result[key] = info
        return result
    
    def delete_features(self, storage_key: str):
        """
        Delete a feature set.
        
        Args:
            storage_key: Key of feature set to delete
        """
        if storage_key not in self._index["feature_sets"]:
            raise ValueError(f"Feature set {storage_key} not found")
        
        info = self._index["feature_sets"][storage_key]
        feature_path = self.store_path / info["file"]
        
        if feature_path.exists():
            feature_path.unlink()
        
        del self._index["feature_sets"][storage_key]
        self._save_index()
    
    def get_feature_stats(self, storage_key: str) -> Dict[str, Any]:
        """
        Get statistics about stored features without loading full data.
        
        Args:
            storage_key: Key of feature set
            
        Returns:
            Dict with statistics
        """
        if storage_key not in self._index["feature_sets"]:
            raise ValueError(f"Feature set {storage_key} not found")
        
        info = self._index["feature_sets"][storage_key]
        
        return {
            "name": info["name"],
            "created_at": info["created_at"],
            "n_samples": info["n_samples"],
            "metadata": info["metadata"]
        }
