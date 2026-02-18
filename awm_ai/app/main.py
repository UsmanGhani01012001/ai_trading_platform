"""
AWM AI Decision Layer - Main Application
========================================

FastAPI application for real-time TAKE/SKIP decisions.

Usage:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Flow:
    NinjaTrader → POST /awm/bar → HMM features → HMM predict 
    → LGBM features (with HMM probs) → LGBM predict → TAKE/SKIP → NinjaTrader

Requirements:
    Place these files in /models folder:
    - hmm_model_4states.pkl
    - hmm_scaler.pkl
    - lgbm_model_take_skip.pkl
    - lgbm_scaler.pkl
    - lgbm_config.pkl
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from threading import Thread
from contextlib import asynccontextmanager
import logging

from app.ingest import router
from app.background import parquet_writer_loop
from app.engine import get_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # STARTUP
    logger.info("=" * 60)
    logger.info("AWM AI Decision Layer Starting...")
    logger.info("=" * 60)
    
    # Pre-load models
    engine = get_engine(models_dir="models")
    if engine.load_models():
        logger.info("✓ All models loaded successfully")
    else:
        logger.warning("⚠ Some models failed to load - check /models folder")
    
    # Start background writer thread
    writer_thread = Thread(target=parquet_writer_loop, daemon=True)
    writer_thread.start()
    logger.info("✓ Background writer started")
    
    logger.info("=" * 60)
    logger.info("Ready to receive bars at POST /awm/bar")
    logger.info("=" * 60)
    
    yield  # Application runs here
    
    # SHUTDOWN
    logger.info("AWM AI Decision Layer shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AWM AI Decision Layer",
    description="Real-time TAKE/SKIP decisions for AWM trading strategy",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow NinjaTrader and local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "AWM AI Decision Layer",
        "version": "1.0.0",
        "endpoints": {
            "POST /awm/bar": "Receive bar, return TAKE/SKIP decision",
            "POST /awm/outcome": "Report trade outcome",
            "GET /awm/status": "Get engine status",
            "GET /awm/health": "Health check"
        }
    }


# For running directly with Python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)