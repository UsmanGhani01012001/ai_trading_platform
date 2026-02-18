"""
AWM AI Prediction Engine
========================
Orchestrates the HMM → LGBM prediction pipeline.

Flow:
1. Receive bar data
2. Build HMM features → Scale → HMM predict state + probabilities
3. Build LGBM features (includes HMM probs) → Scale → LGBM predict TAKE/SKIP
4. Return decision with confidence

Models required (place in /models folder):
- hmm_model_4states.pkl
- hmm_scaler.pkl
- lgbm_model_take_skip.pkl
- lgbm_scaler.pkl
- lgbm_config.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from app.features import FeatureBuffer, HMMFeatureBuilder, LGBMFeatureBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of the TAKE/SKIP prediction"""
    decision: str  # "TAKE" or "SKIP"
    confidence: float  # 0.0 to 1.0
    hmm_state: int  # 0-3
    hmm_probs: list  # [p0, p1, p2, p3]
    lgbm_raw_prob: float  # raw probability from LGBM
    timestamp: str
    bar_index: int
    features_ready: bool
    

class AWMPredictionEngine:
    """
    Main prediction engine for AWM AI Decision Layer.
    
    Handles:
    - Model loading
    - Feature computation via buffer
    - HMM regime detection
    - LGBM TAKE/SKIP decision
    - Latency tracking (<200ms target)
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
        # Feature builders
        self.hmm_feature_builder = HMMFeatureBuilder()
        self.lgbm_feature_builder = LGBMFeatureBuilder()
        
        # Feature buffer (per instrument/series)
        self.buffers: Dict[str, FeatureBuffer] = {}
        
        # Models and scalers (loaded on first use)
        self.hmm_model = None
        self.hmm_scaler = None
        self.lgbm_model = None
        self.lgbm_scaler = None
        self.lgbm_threshold = 0.092   # default from training
        self.lgbm_features = None
        
        # State
        self.models_loaded = False
        
    def load_models(self) -> bool:
        """Load all models and scalers from disk"""
        try:
            # HMM model
            hmm_path = self.models_dir / "hmm_model_4states.pkl"
            if hmm_path.exists():
                with open(hmm_path, 'rb') as f:
                    self.hmm_model = pickle.load(f)
                logger.info(f"✓ HMM model loaded from {hmm_path}")
            else:
                logger.warning(f"⚠ HMM model not found at {hmm_path}")
                return False
            
            # HMM scaler
            hmm_scaler_path = self.models_dir / "hmm_scaler.pkl"
            if hmm_scaler_path.exists():
                with open(hmm_scaler_path, 'rb') as f:
                    self.hmm_scaler = pickle.load(f)
                logger.info(f"✓ HMM scaler loaded from {hmm_scaler_path}")
            else:
                logger.warning(f"⚠ HMM scaler not found at {hmm_scaler_path}")
                return False
            
            # LGBM model
            lgbm_path = self.models_dir / "lgbm_model_take_skip.pkl"
            if lgbm_path.exists():
                with open(lgbm_path, 'rb') as f:
                    self.lgbm_model = pickle.load(f)
                logger.info(f"✓ LGBM model loaded from {lgbm_path}")
            else:
                logger.warning(f"⚠ LGBM model not found at {lgbm_path}")
                return False
            
            # LGBM scaler
            lgbm_scaler_path = self.models_dir / "lgbm_scaler.pkl"
            if lgbm_scaler_path.exists():
                with open(lgbm_scaler_path, 'rb') as f:
                    self.lgbm_scaler = pickle.load(f)
                logger.info(f"✓ LGBM scaler loaded from {lgbm_scaler_path}")
            else:
                logger.warning(f"⚠ LGBM scaler not found at {lgbm_scaler_path}")
                return False
            
            # LGBM config (threshold + features)
            config_path = self.models_dir / "lgbm_config.pkl"
            if config_path.exists():
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)
                self.lgbm_threshold = config.get('optimal_threshold', 0.1253)
                self.lgbm_features = config.get('features', None)
                logger.info(f"✓ LGBM config loaded: threshold={self.lgbm_threshold}")
            else:
                logger.warning(f"⚠ LGBM config not found, using default threshold=0.1253")
            
            self.models_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"✗ Error loading models: {e}")
            return False
    
    def get_buffer(self, instrument: str, series: str) -> FeatureBuffer:
        """Get or create buffer for instrument/series combination"""
        key = f"{instrument}_{series}"
        if key not in self.buffers:
            self.buffers[key] = FeatureBuffer(maxlen=300)
        return self.buffers[key]
    
    def predict(self, bar: dict) -> PredictionResult:
        """
        Main prediction method. Called on each bar close.
        
        Args:
            bar: Dict with keys: instrument, series, time, open, high, low, close, volume, bar_index
        
        Returns:
            PredictionResult with decision, confidence, and diagnostics
        """
        start_time = datetime.now()
        
        # Default result for warmup/error cases
        default_result = PredictionResult(
            decision="SKIP",
            confidence=0.0,
            hmm_state=-1,
            hmm_probs=[0.25, 0.25, 0.25, 0.25],
            lgbm_raw_prob=0.0,
            timestamp=bar.get('time', ''),
            bar_index=bar.get('bar_index', -1),
            features_ready=False
        )
        
        # Load models if not already loaded
        if not self.models_loaded:
            if not self.load_models():
                logger.warning("Models not loaded, returning SKIP")
                return default_result
        
        # Get buffer for this instrument/series
        instrument = bar.get('instrument', 'NQ')
        series = bar.get('series', '50-50-75')
        buffer = self.get_buffer(instrument, series)
        
        # Add bar to buffer
        has_enough_data = buffer.add_bar(bar)
        
        if not has_enough_data:
            logger.info(f"Warmup: {len(buffer)}/30 bars collected")
            return default_result
        
        # Get DataFrame
        df = buffer.get_df()
        
        try:
            # ============================================
            # STEP 1: Build HMM features
            # ============================================
            hmm_features = self.hmm_feature_builder.build_features(df)
            if hmm_features is None:
                logger.warning("HMM features returned None")
                return default_result
            
            # Scale HMM features
            hmm_features_scaled = self.hmm_scaler.transform(hmm_features.reshape(1, -1))
            
            # ============================================
            # STEP 2: HMM prediction
            # ============================================
            hmm_state = self.hmm_model.predict(hmm_features_scaled)[0]
            hmm_probs = self.hmm_model.predict_proba(hmm_features_scaled)[0]
            
            # ============================================
            # STEP 3: Build LGBM features (with HMM probs)
            # ============================================
            lgbm_features = self.lgbm_feature_builder.build_features(df, hmm_probs)
            if lgbm_features is None:
                logger.warning("LGBM features returned None")
                return default_result
            
            # Scale LGBM features
            lgbm_features_scaled = self.lgbm_scaler.transform(lgbm_features.reshape(1, -1))
            
            # ============================================
            # STEP 4: LGBM prediction
            # ============================================
            lgbm_raw_prob = self.lgbm_model.predict(lgbm_features_scaled)[0]
            
            # Apply threshold
            decision = "TAKE" if lgbm_raw_prob >= self.lgbm_threshold else "SKIP"
            
            # Confidence: distance from threshold, normalized
            if decision == "TAKE":
                confidence = min(1.0, (lgbm_raw_prob - self.lgbm_threshold) / (1.0 - self.lgbm_threshold))
            else:
                confidence = min(1.0, (self.lgbm_threshold - lgbm_raw_prob) / self.lgbm_threshold)
            
            # Timing check
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            if elapsed_ms > 200:
                logger.warning(f"⚠ Prediction took {elapsed_ms:.1f}ms (target <200ms)")
            
            return PredictionResult(
                decision=decision,
                confidence=float(confidence),
                hmm_state=int(hmm_state),
                hmm_probs=[float(p) for p in hmm_probs],
                lgbm_raw_prob=float(lgbm_raw_prob),
                timestamp=str(bar.get('time', '')),
                bar_index=bar.get('bar_index', -1),
                features_ready=True
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return default_result
    
    def update_trade_outcome(self, is_win: bool, pnl: float = 0.0):
        """
        Call this after a trade completes to update rolling metrics.
        This affects rolling_drawdown and win_loss_ratio features.
        """
        self.hmm_feature_builder.update_trade_result(is_win, pnl)
    
    def get_status(self) -> dict:
        """Return engine status for diagnostics"""
        return {
            "models_loaded": self.models_loaded,
            "buffers": {k: len(v) for k, v in self.buffers.items()},
            "lgbm_threshold": self.lgbm_threshold,
            "lgbm_features": self.lgbm_features,
        }


# Global engine instance (singleton pattern for FastAPI)
_engine: Optional[AWMPredictionEngine] = None


def get_engine(models_dir: str = "models") -> AWMPredictionEngine:
    """Get or create global prediction engine"""
    global _engine
    if _engine is None:
        _engine = AWMPredictionEngine(models_dir=models_dir)
    return _engine