"""
FastAPI Application

Exposes POST /predict endpoint for real-time inference.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from probabilistic_market_engine.config.settings import settings
from probabilistic_market_engine.core.physics_engine.engine import PhysicsEngine
from probabilistic_market_engine.core.liquidity_engine.engine import LiquidityEngine
from probabilistic_market_engine.core.feature_pipeline.pipeline import FeaturePipeline
from probabilistic_market_engine.core.regime_model.model import RegimeInferenceModel
from probabilistic_market_engine.core.outcome_models.experts import OutcomeExpertModels
from probabilistic_market_engine.core.inference_engine.engine import InferenceEngine
from probabilistic_market_engine.core.risk_engine.engine import RiskEngine
from probabilistic_market_engine.monitoring.drift_detection.detector import DriftDetector
from probabilistic_market_engine.monitoring.regime_tracking.tracker import RegimeTracker
from probabilistic_market_engine.monitoring.calibration_tracking.tracker import CalibrationTracker
from probabilistic_market_engine.persistence.model_registry.registry import ModelRegistry
from probabilistic_market_engine.persistence.feature_store.store import FeatureStore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class OHLCVBar(BaseModel):
    """Single OHLCV bar."""
    timestamp: Optional[str] = None
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(default=0, ge=0)


class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    ohlcv_window: List[OHLCVBar] = Field(..., min_length=64, max_length=256)
    symbol: str = Field(default="UNKNOWN")


class PredictResponse(BaseModel):
    """Response from /predict endpoint."""
    prob_continuation: float = Field(..., ge=0, le=1)
    regime_probabilities: Dict[str, float]
    shock_probability: float = Field(..., ge=0, le=1)
    risk_scaling_factor: float
    model_version: str
    timestamp: str
    
    # Optional debug info
    prediction_confidence: Optional[float] = None
    regime_entropy: Optional[float] = None
    alerts: Optional[List[Dict]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_version: str
    model_loaded: bool


# Global state (controlled memory)
class AppState:
    """Application state container."""
    
    def __init__(self):
        self.physics_engine: Optional[PhysicsEngine] = None
        self.liquidity_engine: Optional[LiquidityEngine] = None
        self.feature_pipeline: Optional[FeaturePipeline] = None
        self.regime_model: Optional[RegimeInferenceModel] = None
        self.outcome_models: Optional[OutcomeExpertModels] = None
        self.inference_engine: Optional[InferenceEngine] = None
        self.risk_engine: Optional[RiskEngine] = None
        
        # Monitoring
        self.drift_detector: Optional[DriftDetector] = None
        self.regime_tracker: Optional[RegimeTracker] = None
        self.calibration_tracker: Optional[CalibrationTracker] = None
        
        # Persistence
        self.model_registry: Optional[ModelRegistry] = None
        self.feature_store: Optional[FeatureStore] = None
        
        # State
        self.is_loaded: bool = False
        self.current_version: str = "unknown"


app_state = AppState()


def load_models():
    """Load models from registry."""
    global app_state
    
    logger.info("Loading models...")
    
    # Initialize persistence
    app_state.model_registry = ModelRegistry()
    app_state.feature_store = FeatureStore()
    
    # Get active version
    active_version = app_state.model_registry.get_active_version()
    
    if active_version is None:
        logger.warning("No active model version found. Using placeholder models.")
        # Initialize with default models for demo
        _init_placeholder_models()
        return
    
    # Load models
    regime_model, outcome_models, feature_stats = app_state.model_registry.load_version(
        active_version.version
    )
    
    # Initialize engines
    app_state.physics_engine = PhysicsEngine()
    app_state.liquidity_engine = LiquidityEngine()
    app_state.feature_pipeline = FeaturePipeline(
        physics_engine=app_state.physics_engine,
        liquidity_engine=app_state.liquidity_engine
    )
    
    # Load feature stats
    app_state.feature_pipeline.load_stats(active_version.feature_stats_path)
    
    # Set models
    app_state.regime_model = regime_model
    app_state.outcome_models = outcome_models
    app_state.inference_engine = InferenceEngine(outcome_models)
    app_state.inference_engine.set_model_version(active_version.version)
    app_state.risk_engine = RiskEngine()
    
    # Initialize monitoring
    app_state.drift_detector = DriftDetector()
    app_state.regime_tracker = RegimeTracker()
    app_state.calibration_tracker = CalibrationTracker()
    
    # Set reference for drift detection
    # (Would be loaded from training data in production)
    
    app_state.is_loaded = True
    app_state.current_version = active_version.version
    
    logger.info(f"Models loaded successfully. Version: {active_version.version}")


def _init_placeholder_models():
    """Initialize placeholder models for demo/testing."""
    global app_state
    
    app_state.physics_engine = PhysicsEngine()
    app_state.liquidity_engine = LiquidityEngine()
    app_state.feature_pipeline = FeaturePipeline(
        physics_engine=app_state.physics_engine,
        liquidity_engine=app_state.liquidity_engine
    )
    app_state.regime_model = RegimeInferenceModel()
    app_state.outcome_models = OutcomeExpertModels()
    app_state.inference_engine = InferenceEngine(app_state.outcome_models)
    app_state.risk_engine = RiskEngine()
    
    app_state.drift_detector = DriftDetector()
    app_state.regime_tracker = RegimeTracker()
    app_state.calibration_tracker = CalibrationTracker()
    
    app_state.is_loaded = True
    app_state.current_version = "placeholder"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup
    logger.info("Starting up...")
    load_models()
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Probabilistic State-Space Trading Engine",
    description="Production-grade nonlinear probabilistic state-space trading engine",
    version=settings.model_version,
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if app_state.is_loaded else "degraded",
        model_version=app_state.current_version,
        model_loaded=app_state.is_loaded
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict continuation probability from OHLCV window.
    
    Args:
        request: PredictRequest with OHLCV window
    
    Returns:
        PredictResponse with probabilities and risk scaling
    """
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        data = {
            'open': [b.open for b in request.ohlcv_window],
            'high': [b.high for b in request.ohlcv_window],
            'low': [b.low for b in request.ohlcv_window],
            'close': [b.close for b in request.ohlcv_window],
            'volume': [b.volume for b in request.ohlcv_window],
        }
        
        # Create index from timestamps if provided
        if request.ohlcv_window[0].timestamp:
            index = pd.to_datetime([b.timestamp for b in request.ohlcv_window])
        else:
            index = pd.date_range(end=datetime.utcnow(), periods=len(request.ohlcv_window), freq='15min')
        
        df = pd.DataFrame(data, index=index)
        
        # Compute features
        feature_set = app_state.feature_pipeline.transform(df)
        
        if not feature_set.is_valid:
            raise HTTPException(status_code=400, detail=f"Feature computation failed: {feature_set.missing_features}")
        
        # Regime inference
        regime_state = app_state.regime_model.predict(feature_set.R_t, feature_set.timestamp)
        
        # Outcome prediction
        inference_result = app_state.inference_engine.predict(
            feature_set.X_t, regime_state, feature_set.timestamp
        )
        
        # Risk adjustment
        risk_adjustment = app_state.risk_engine.compute(inference_result)
        
        # Update monitoring
        alerts = app_state.drift_detector.update(
            features=feature_set.X_t,
            regime_probs=regime_state.smoothed_probs,
            prediction=inference_result.prob_continuation,
            timestamp=feature_set.timestamp
        )
        app_state.regime_tracker.update(regime_state)
        
        # Background: save features
        background_tasks.add_task(
            app_state.feature_store.save_features,
            timestamp=feature_set.timestamp or datetime.utcnow(),
            features=feature_set.X_t,
            feature_names=feature_set.feature_names,
            metadata={
                'symbol': request.symbol,
                'regime': regime_state.dominant_regime,
                'prediction': inference_result.prob_continuation,
            }
        )
        
        # Build response
        response = PredictResponse(
            prob_continuation=inference_result.prob_continuation,
            regime_probabilities={
                'trend': regime_state.trend_probability,
                'range': regime_state.range_probability,
                'shock': regime_state.shock_probability,
            },
            shock_probability=regime_state.shock_probability,
            risk_scaling_factor=risk_adjustment.risk_scaling_factor,
            model_version=app_state.current_version,
            timestamp=feature_set.timestamp.isoformat() if feature_set.timestamp else datetime.utcnow().isoformat(),
            prediction_confidence=inference_result.prediction_confidence,
            regime_entropy=regime_state.regime_entropy,
            alerts=[a.to_dict() for a in alerts] if alerts else None,
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/status")
async def monitoring_status():
    """Get current monitoring status."""
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        'drift_detection': {
            'feature_stats': app_state.drift_detector.get_feature_statistics(),
            'regime_stats': app_state.drift_detector.get_regime_statistics(),
        },
        'regime_tracking': app_state.regime_tracker.get_summary(),
        'calibration': app_state.calibration_tracker.get_calibration_report(),
    }


@app.post("/feedback")
async def feedback(outcome: int, timestamp: str):
    """
    Submit outcome feedback for calibration tracking.
    
    Args:
        outcome: Actual outcome (0 or 1)
        timestamp: Timestamp of prediction
    """
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        ts = datetime.fromisoformat(timestamp)
        
        # Load the prediction from feature store
        features = app_state.feature_store.load_features(ts)
        
        if features and 'metadata' in features:
            prediction = features['metadata'].get('prediction', 0.5)
            app_state.calibration_tracker.update(prediction, outcome, ts)
        
        return {'status': 'ok'}
        
    except Exception as e:
        logger.exception("Feedback error")
        raise HTTPException(status_code=500, detail=str(e))
