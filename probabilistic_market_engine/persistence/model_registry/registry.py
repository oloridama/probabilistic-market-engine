"""
Model Registry

Handles versioning, saving, and loading of trained models.
"""

import os
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ModelRegistry:
    """
    Registry for managing model versions.
    
    Supports:
    - Version tracking
    - Model persistence (pickle)
    - Active version management
    - Metadata storage
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to store model versions. 
                          Defaults to ~/.probabilistic_market_engine/models
        """
        if registry_path is None:
            home = Path.home()
            registry_path = home / ".probabilistic_market_engine" / "models"
        
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # Index of registered versions
        self._index_file = self.registry_path / "registry_index.json"
        self._index = self._load_index()
        
        # Active version
        self._active_version = self._index.get("active_version", None)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load registry index from disk."""
        if self._index_file.exists():
            with open(self._index_file, 'r') as f:
                return json.load(f)
        return {"versions": {}}
    
    def _save_index(self):
        """Save registry index to disk."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f, indent=2, default=str)
    
    def register_version(
        self,
        version: str,
        regime_model: Any,
        outcome_models: Any,
        feature_pipeline: Any,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Register a new model version.
        
        Args:
            version: Version string (e.g., "v1.0.0")
            regime_model: Trained RegimeInferenceModel
            outcome_models: Trained OutcomeExpertModels
            feature_pipeline: FeaturePipeline instance
            description: Human-readable description
            metadata: Additional metadata dict
            
        Returns:
            version_id: The registered version string
        """
        version_path = self.registry_path / version
        version_path.mkdir(exist_ok=True)
        
        # Save models
        model_data = {
            "regime_model": regime_model,
            "outcome_models": outcome_models,
            "feature_pipeline": feature_pipeline
        }
        
        model_file = version_path / "models.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata
        version_info = {
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "model_file": str(model_file.relative_to(self.registry_path))
        }
        
        info_file = version_path / "info.json"
        with open(info_file, 'w') as f:
            json.dump(version_info, f, indent=2, default=str)
        
        # Update index
        self._index["versions"][version] = {
            "created_at": version_info["created_at"],
            "description": description
        }
        self._save_index()
        
        return version
    
    def load_version(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a specific model version.
        
        Args:
            version: Version to load. If None, loads active version.
            
        Returns:
            Dict with keys: regime_model, outcome_models, feature_pipeline
        """
        if version is None:
            version = self._active_version
            
        if version is None:
            raise ValueError("No version specified and no active version set")
        
        if version not in self._index["versions"]:
            raise ValueError(f"Version {version} not found in registry")
        
        version_path = self.registry_path / version
        model_file = version_path / "models.pkl"
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data
    
    def set_active_version(self, version: str):
        """
        Set the active version for inference.
        
        Args:
            version: Version string to activate
        """
        if version not in self._index["versions"]:
            raise ValueError(f"Version {version} not found in registry")
        
        self._active_version = version
        self._index["active_version"] = version
        self._save_index()
    
    def get_active_version(self) -> Optional[str]:
        """Get currently active version."""
        return self._active_version
    
    def list_versions(self) -> Dict[str, Dict]:
        """List all registered versions."""
        return self._index["versions"].copy()
    
    def delete_version(self, version: str):
        """
        Delete a version from registry.
        
        Args:
            version: Version to delete
        """
        if version not in self._index["versions"]:
            raise ValueError(f"Version {version} not found")
        
        # Remove files
        version_path = self.registry_path / version
        if version_path.exists():
            import shutil
            shutil.rmtree(version_path)
        
        # Update index
        del self._index["versions"][version]
        
        # Reset active if needed
        if self._active_version == version:
            self._active_version = None
            self._index["active_version"] = None
        
        self._save_index()
