#!/usr/bin/env python3
"""Real-time prediction API for Tennis Predictor 101.

FastAPI-based server providing real-time tennis match predictions
with comprehensive analytics and market analysis.
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
import json
from contextlib import asynccontextmanager

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import get_config
from models.ensemble import StackingEnsemble
from features import ELORatingSystem, MomentumAnalyzer
from data.collectors import OddsAPICollector, JeffSackmannCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction_api")

# Global variables for models and data
ensemble_model = None
elo_system = None
momentum_analyzer = None
redis_client = None
config = get_config()


# Pydantic models for API
class PlayerInfo(BaseModel):
    """Player information."""
    player_id: str
    name: str
    ranking: Optional[int] = None
    country: Optional[str] = None


class MatchRequest(BaseModel):
    """Match prediction request."""
    player1: PlayerInfo
    player2: PlayerInfo
    surface: str = Field(..., regex="^(Clay|Hard|Grass)$")
    tournament: Optional[str] = None
    round_info: Optional[str] = None
    date: Optional[datetime] = None
    best_of: int = Field(default=3, ge=3, le=5)
    indoor: bool = False
    altitude: float = Field(default=0.0, ge=0.0, le=5000.0)
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    

class PredictionResponse(BaseModel):
    """Match prediction response."""
    match_id: str
    player1_win_probability: float = Field(..., ge=0.0, le=1.0)
    player2_win_probability: float = Field(..., ge=0.0, le=1.0)
    prediction_confidence: float = Field(..., ge=0.0, le=1.0)
    elo_difference: float
    momentum_analysis: Dict[str, Any]
    surface_advantage: Optional[str] = None
    key_factors: List[str]
    market_analysis: Optional[Dict[str, Any]] = None
    prediction_timestamp: datetime
    model_version: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    model_loaded: bool
    data_freshness: Optional[str] = None
    version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()


# Create FastAPI app
app = FastAPI(
    title="Tennis Predictor 101 API",
    description="Advanced tennis match outcome prediction with real-time analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def startup_event():
    """Initialize models and data on startup."""
    global ensemble_model, elo_system, momentum_analyzer, redis_client
    
    logger.info("Starting Tennis Predictor 101 API")
    
    try:
        # Initialize Redis cache
        redis_url = config.get('api.caching.redis_url', 'redis://localhost:6379')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()  # Test connection
        logger.info("Redis cache connected")
    except Exception as e:
        logger.warning(f"Redis cache not available: {e}")
        redis_client = None
    
    # Load models
    try:
        model_dir = "models/saved"
        if os.path.exists(model_dir):
            # Find latest ensemble model
            ensemble_files = [f for f in os.listdir(model_dir) if f.startswith('ensemble_')]
            if ensemble_files:
                latest_ensemble = sorted(ensemble_files)[-1]
                ensemble_path = os.path.join(model_dir, latest_ensemble)
                
                ensemble_model = StackingEnsemble(base_models=[])
                ensemble_model.load_model(ensemble_path)
                logger.info(f"Ensemble model loaded from {ensemble_path}")
            
            # Load ELO ratings
            elo_files = [f for f in os.listdir(model_dir) if f.startswith('elo_ratings_')]
            if elo_files:
                latest_elo = sorted(elo_files)[-1]
                elo_path = os.path.join(model_dir, latest_elo)
                
                elo_system = ELORatingSystem()
                elo_ratings = pd.read_csv(elo_path)
                elo_system.load_ratings(elo_ratings)
                logger.info(f"ELO ratings loaded from {elo_path}")
        
        # Initialize momentum analyzer
        momentum_analyzer = MomentumAnalyzer()
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    
    logger.info("API startup completed")


async def shutdown_event():
    """Cleanup on shutdown."""
    global redis_client
    
    logger.info("Shutting down Tennis Predictor 101 API")
    
    if redis_client:
        redis_client.close()
    
    logger.info("API shutdown completed")


def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key."""
    key_parts = [prefix]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    return ":".join(key_parts)


def cache_get(key: str) -> Optional[Dict]:
    """Get data from cache."""
    if not redis_client:
        return None
    
    try:
        cached_data = redis_client.get(key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    
    return None


def cache_set(key: str, data: Dict, ttl: int = 300):
    """Set data in cache."""
    if not redis_client:
        return
    
    try:
        redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.warning(f"Cache set error: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_loaded = ensemble_model is not None and elo_system is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.now(),
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_match(request: MatchRequest, background_tasks: BackgroundTasks):
    """Predict tennis match outcome."""
    # Check if models are loaded
    if not ensemble_model or not elo_system:
        raise HTTPException(
            status_code=503, 
            detail="Prediction models not available"
        )
    
    # Generate match ID
    match_id = f"{request.player1.player_id}_vs_{request.player2.player_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Check cache
    cache_key = get_cache_key(
        "prediction",
        p1=request.player1.player_id,
        p2=request.player2.player_id,
        surface=request.surface
    )
    
    cached_result = cache_get(cache_key)
    if cached_result:
        logger.info(f"Returning cached prediction for {match_id}")
        return PredictionResponse(**cached_result)
    
    try:
        # Calculate prediction
        prediction_result = await calculate_prediction(request, match_id)
        
        # Cache result
        background_tasks.add_task(
            cache_set, cache_key, prediction_result.dict(), 300
        )
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Prediction error for {match_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction calculation failed: {str(e)}"
        )


async def calculate_prediction(request: MatchRequest, match_id: str) -> PredictionResponse:
    """Calculate match prediction."""
    logger.info(f"Calculating prediction for {match_id}")
    
    # Get ELO ratings
    player1_elo = elo_system.get_rating(request.player1.player_id, request.surface)
    player2_elo = elo_system.get_rating(request.player2.player_id, request.surface)
    
    elo_diff = player1_elo - player2_elo
    
    # Calculate base probability from ELO
    elo_probability = elo_system.calculate_match_probability(
        request.player1.player_id, 
        request.player2.player_id, 
        request.surface
    )
    
    # Create feature vector
    features = create_feature_vector(request, player1_elo, player2_elo)
    
    # Get ensemble prediction
    if ensemble_model.is_trained:
        prediction_proba = ensemble_model.predict_proba(features)
        model_probability = prediction_proba[0, 1]  # Probability of player1 winning
        confidence = max(model_probability, 1 - model_probability)
    else:
        model_probability = elo_probability
        confidence = 0.7  # Default confidence
    
    # Analyze momentum (simplified)
    momentum_analysis = {
        "player1_momentum": 0.5,  # Placeholder
        "player2_momentum": 0.5,  # Placeholder
        "momentum_edge": "neutral"
    }
    
    # Determine surface advantage
    surface_advantage = determine_surface_advantage(request)
    
    # Key factors
    key_factors = identify_key_factors(
        elo_diff, request.surface, momentum_analysis
    )
    
    # Market analysis (if odds available)
    market_analysis = await get_market_analysis(request)
    
    result = PredictionResponse(
        match_id=match_id,
        player1_win_probability=model_probability,
        player2_win_probability=1 - model_probability,
        prediction_confidence=confidence,
        elo_difference=elo_diff,
        momentum_analysis=momentum_analysis,
        surface_advantage=surface_advantage,
        key_factors=key_factors,
        market_analysis=market_analysis,
        prediction_timestamp=datetime.now(),
        model_version="1.0.0"
    )
    
    logger.info(f"Prediction completed for {match_id}: P1={model_probability:.3f}")
    return result


def create_feature_vector(request: MatchRequest, player1_elo: float, player2_elo: float) -> pd.DataFrame:
    """Create feature vector for prediction."""
    features = {
        'winner_elo': player1_elo,
        'loser_elo': player2_elo,
        'elo_diff': player1_elo - player2_elo,
        'elo_ratio': player1_elo / (player2_elo + 1),
        'elo_sum': player1_elo + player2_elo,
        'surface_clay': 1.0 if request.surface == 'Clay' else 0.0,
        'surface_grass': 1.0 if request.surface == 'Grass' else 0.0,
        'surface_hard': 1.0 if request.surface == 'Hard' else 0.0,
        'best_of': request.best_of,
        'indoor_outdoor': 1.0 if request.indoor else 0.0,
        'altitude_factor': request.altitude / 1000.0,
        'temperature_factor': (request.temperature or 22.0) / 35.0,
        'is_final': 1.0 if request.round_info == 'F' else 0.0,
        'is_semifinal': 1.0 if request.round_info == 'SF' else 0.0,
        'momentum_score': 0.5,  # Placeholder
        'recent_form': 0.5,     # Placeholder
        'confidence_index': 0.5  # Placeholder
    }
    
    return pd.DataFrame([features])


def determine_surface_advantage(request: MatchRequest) -> Optional[str]:
    """Determine surface advantage between players."""
    # This would require historical surface performance data
    # For now, return None (no clear advantage)
    return None


def identify_key_factors(elo_diff: float, surface: str, momentum: Dict) -> List[str]:
    """Identify key factors affecting the match."""
    factors = []
    
    if abs(elo_diff) > 100:
        factors.append(f"Significant rating difference ({elo_diff:+.0f} points)")
    
    if surface == 'Clay':
        factors.append("Clay court specialists may have advantage")
    elif surface == 'Grass':
        factors.append("Grass court adaptation crucial")
    
    factors.append("Recent form and momentum important")
    
    return factors


async def get_market_analysis(request: MatchRequest) -> Optional[Dict[str, Any]]:
    """Get betting market analysis if available."""
    try:
        # This would integrate with odds API
        # For now, return placeholder
        return {
            "market_available": False,
            "implied_probability": None,
            "value_bet_detected": False
        }
    except Exception as e:
        logger.warning(f"Market analysis error: {e}")
        return None


@app.get("/player/{player_id}/stats")
async def get_player_stats(player_id: str):
    """Get player statistics."""
    if not elo_system:
        raise HTTPException(status_code=503, detail="ELO system not available")
    
    cache_key = get_cache_key("player_stats", player_id=player_id)
    cached_stats = cache_get(cache_key)
    
    if cached_stats:
        return cached_stats
    
    try:
        stats = elo_system.get_player_statistics(player_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Player not found")
        
        # Cache for 1 hour
        cache_set(cache_key, stats, 3600)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting player stats for {player_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/rankings/{surface}")
async def get_surface_rankings(surface: str, limit: int = 50):
    """Get surface-specific rankings."""
    if surface not in ['clay', 'hard', 'grass', 'overall']:
        raise HTTPException(status_code=400, detail="Invalid surface")
    
    if not elo_system:
        raise HTTPException(status_code=503, detail="ELO system not available")
    
    try:
        # Export all ratings and filter/sort
        all_ratings = elo_system.export_ratings()
        
        if surface == 'overall':
            rankings = all_ratings.sort_values('overall_rating', ascending=False)
            rating_col = 'overall_rating'
        else:
            rating_col = f'{surface}_rating'
            if rating_col in all_ratings.columns:
                rankings = all_ratings.dropna(subset=[rating_col])
                rankings = rankings.sort_values(rating_col, ascending=False)
            else:
                rankings = all_ratings.sort_values('overall_rating', ascending=False)
                rating_col = 'overall_rating'
        
        # Limit results
        rankings = rankings.head(limit)
        
        # Format response
        result = []
        for i, (_, row) in enumerate(rankings.iterrows(), 1):
            result.append({
                'rank': i,
                'player_id': row['player_id'],
                'rating': row[rating_col],
                'matches_played': row.get('match_count', 0)
            })
        
        return {
            'surface': surface,
            'last_updated': datetime.now().isoformat(),
            'rankings': result
        }
        
    except Exception as e:
        logger.error(f"Error getting {surface} rankings: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tennis Predictor 101 API",
        "version": "1.0.0",
        "description": "Advanced tennis match outcome prediction",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "player_stats": "/player/{player_id}/stats (GET)",
            "rankings": "/rankings/{surface} (GET)",
            "docs": "/docs (GET)"
        },
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Configuration
    host = config.get('api.prediction_service.host', '0.0.0.0')
    port = config.get('api.prediction_service.port', 8000)
    workers = config.get('api.prediction_service.workers', 1)
    
    # Run server
    uvicorn.run(
        "prediction_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )