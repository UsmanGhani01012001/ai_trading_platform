"""
AWM AI Ingest Module
====================
FastAPI endpoints for receiving bars and returning predictions.

Endpoints:
- POST /awm/bar      → Receive bar, return TAKE/SKIP decision
- POST /awm/outcome  → Report trade outcome (for rolling metrics)
- GET  /awm/status   → Get engine status
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
import logging

from app.queue import push_bar, get_bar_count
from app.engine import get_engine, PredictionResult

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================
# REQUEST MODELS
# ============================================

class AWMBar(BaseModel):
    """Bar data from NinjaTrader"""
    instrument: str
    bars_type: str
    series: str
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_index: int
    is_closed: bool


class TradeOutcome(BaseModel):
    """Trade outcome for updating rolling metrics"""
    is_win: bool
    pnl: float = 0.0
    instrument: str = "NQ"
    series: str = "50-50-75"


# ============================================
# RESPONSE MODELS
# ============================================

class DecisionResponse(BaseModel):
    """TAKE/SKIP decision response"""
    status: str
    decision: str
    confidence: float
    hmm_state: int
    hmm_probs: List[float]
    lgbm_prob: float
    bar_index: int
    warmup: bool


class StatusResponse(BaseModel):
    """Engine status response"""
    models_loaded: bool
    buffers: dict
    threshold: float
    queue_size: int


# ============================================
# ENDPOINTS
# ============================================

@router.post("/awm/bar", response_model=DecisionResponse)
async def receive_bar(bar: AWMBar):
    """
    Receive bar data and return TAKE/SKIP decision.
    
    This is the main endpoint called by NinjaTrader on each bar close.
    
    Flow:
    1. Convert bar to dict
    2. Store in queue (for backup/logging)
    3. Run prediction pipeline (HMM → LGBM)
    4. Return decision
    
    Target latency: <200ms
    """
    # Convert to dict
    bar_dict = bar.dict()
    bar_dict['time'] = str(bar.time)  # Ensure string for consistency
    
    # Store in queue (for background writer / logging)
    push_bar(bar_dict)
    
    # Get prediction engine
    engine = get_engine(models_dir="models")
    
    # Run prediction
    result: PredictionResult = engine.predict(bar_dict)
    
    # Log decision
    logger.info(
        f"[{bar.bar_index}] {result.decision} "
        f"(conf={result.confidence:.2f}, hmm={result.hmm_state}, "
        f"prob={result.lgbm_raw_prob:.3f})"
    )
    
    return DecisionResponse(
        status="ok",
        decision=result.decision,
        confidence=result.confidence,
        hmm_state=result.hmm_state,
        hmm_probs=result.hmm_probs,
        lgbm_prob=result.lgbm_raw_prob,
        bar_index=result.bar_index,
        warmup=not result.features_ready
    )


@router.post("/awm/outcome")
async def report_outcome(outcome: TradeOutcome):
    """
    Report trade outcome for updating rolling metrics.
    
    Call this after each trade completes to update:
    - rolling_drawdown
    - win_loss_ratio
    
    These feed into HMM features for regime detection.
    """
    engine = get_engine()
    engine.update_trade_outcome(is_win=outcome.is_win, pnl=outcome.pnl)
    
    return {
        "status": "ok",
        "message": f"Recorded {'WIN' if outcome.is_win else 'LOSS'} with PnL={outcome.pnl}"
    }


@router.get("/awm/status", response_model=StatusResponse)
async def get_status():
    """
    Get engine status for diagnostics.
    
    Returns:
    - models_loaded: Whether all models are loaded
    - buffers: Bar count per instrument/series
    - threshold: Current LGBM threshold
    - queue_size: Bars in queue
    """
    engine = get_engine()
    status = engine.get_status()
    
    return StatusResponse(
        models_loaded=status["models_loaded"],
        buffers=status["buffers"],
        threshold=status["lgbm_threshold"],
        queue_size=get_bar_count()
    )


@router.get("/awm/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}