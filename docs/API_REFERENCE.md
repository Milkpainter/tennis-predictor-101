# Tennis Predictor 101 API Reference

## Overview

The Tennis Predictor 101 API provides real-time tennis match predictions with comprehensive analytics. The API is built with FastAPI and offers automatic documentation, request validation, and high performance.

**Base URL**: `http://localhost:8000`

**Interactive Documentation**: `http://localhost:8000/docs`

## Authentication

Currently, the API is open access. For production deployment, implement authentication as needed.

## Endpoints

### Health Check

#### GET `/health`

Check API health and model status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-09-20T21:15:32Z",
  "model_loaded": true,
  "data_freshness": "2 hours ago",
  "version": "1.0.0"
}
```

**Status Codes**:
- `200`: API is healthy
- `503`: API is degraded (models not loaded)

---

### Match Prediction

#### POST `/predict`

Predict the outcome of a tennis match.

**Request Body**:
```json
{
  "player1": {
    "player_id": "novak_djokovic",
    "name": "Novak Djokovic",
    "ranking": 1,
    "country": "SRB"
  },
  "player2": {
    "player_id": "carlos_alcaraz",
    "name": "Carlos Alcaraz", 
    "ranking": 2,
    "country": "ESP"
  },
  "surface": "Hard",
  "tournament": "US Open",
  "round_info": "F",
  "date": "2025-09-20T18:00:00Z",
  "best_of": 5,
  "indoor": false,
  "altitude": 10.0,
  "temperature": 25.0,
  "humidity": 65.0
}
```

**Required Fields**:
- `player1.player_id`: String
- `player1.name`: String  
- `player2.player_id`: String
- `player2.name`: String
- `surface`: One of "Clay", "Hard", "Grass"

**Optional Fields**:
- `player1.ranking`: Integer
- `player1.country`: String (3-letter code)
- `tournament`: String
- `round_info`: String ("F", "SF", "QF", "R16", etc.)
- `date`: ISO datetime
- `best_of`: Integer (3 or 5, default: 3)
- `indoor`: Boolean (default: false)
- `altitude`: Float (meters above sea level, 0-5000)
- `temperature`: Float (Celsius)
- `humidity`: Float (percentage)

**Response**:
```json
{
  "match_id": "novak_djokovic_vs_carlos_alcaraz_20250920_181532",
  "player1_win_probability": 0.652,
  "player2_win_probability": 0.348,
  "prediction_confidence": 0.824,
  "elo_difference": 127.5,
  "momentum_analysis": {
    "player1_momentum": 0.68,
    "player2_momentum": 0.71,
    "momentum_edge": "player2"
  },
  "surface_advantage": "player1",
  "key_factors": [
    "Significant rating difference (+128 points)",
    "Player1 strong on hard courts",
    "Recent form favors Player2"
  ],
  "market_analysis": {
    "market_available": true,
    "implied_probability": 0.61,
    "value_bet_detected": true,
    "expected_value": 0.042
  },
  "prediction_timestamp": "2025-09-20T21:15:32Z",
  "model_version": "1.0.0"
}
```

**Status Codes**:
- `200`: Successful prediction
- `400`: Invalid request data
- `503`: Models not available
- `500`: Internal server error

---

### Player Statistics

#### GET `/player/{player_id}/stats`

Get comprehensive statistics for a player.

**Parameters**:
- `player_id`: String - Player identifier

**Example**: `GET /player/novak_djokovic/stats`

**Response**:
```json
{
  "player_id": "novak_djokovic",
  "current_rating": 2087.5,
  "peak_rating": 2156.3,
  "rating_volatility": 45.2,
  "rating_trend": 12.3,
  "match_count": 847,
  "last_update": "2025-09-15T14:30:00Z",
  "surface_ratings": {
    "clay": 2001.2,
    "hard": 2087.5,
    "grass": 2134.8
  },
  "recent_form": {
    "last_10_matches": "8-2",
    "win_percentage": 0.80,
    "momentum_score": 0.73
  },
  "surface_performance": {
    "clay": {"matches": 245, "win_rate": 0.798},
    "hard": {"matches": 421, "win_rate": 0.823},
    "grass": {"matches": 181, "win_rate": 0.856}
  }
}
```

**Status Codes**:
- `200`: Successful response
- `404`: Player not found
- `503`: ELO system not available

---

### Surface Rankings

#### GET `/rankings/{surface}`

Get player rankings for a specific surface.

**Parameters**:
- `surface`: String - One of "clay", "hard", "grass", "overall"
- `limit`: Integer - Number of players to return (default: 50, max: 100)

**Example**: `GET /rankings/clay?limit=20`

**Response**:
```json
{
  "surface": "clay",
  "last_updated": "2025-09-20T21:15:32Z",
  "rankings": [
    {
      "rank": 1,
      "player_id": "rafael_nadal",
      "rating": 2234.7,
      "matches_played": 523
    },
    {
      "rank": 2, 
      "player_id": "novak_djokovic",
      "rating": 2001.2,
      "matches_played": 245
    }
  ]
}
```

**Status Codes**:
- `200`: Successful response
- `400`: Invalid surface parameter
- `503`: ELO system not available

---

### Root Information

#### GET `/`

Get API information and available endpoints.

**Response**:
```json
{
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
  "timestamp": "2025-09-20T21:15:32Z"
}
```

## Response Models

### PredictionResponse

| Field | Type | Description |
|-------|------|-------------|
| `match_id` | string | Unique match identifier |
| `player1_win_probability` | float | Probability of player1 winning (0-1) |
| `player2_win_probability` | float | Probability of player2 winning (0-1) |
| `prediction_confidence` | float | Model confidence in prediction (0-1) |
| `elo_difference` | float | ELO rating difference (player1 - player2) |
| `momentum_analysis` | object | Momentum scores and analysis |
| `surface_advantage` | string | Player with surface advantage |
| `key_factors` | array | Important factors affecting prediction |
| `market_analysis` | object | Betting market analysis (if available) |
| `prediction_timestamp` | datetime | When prediction was made |
| `model_version` | string | Model version used |

### PlayerStats

| Field | Type | Description |
|-------|------|-------------|
| `player_id` | string | Player identifier |
| `current_rating` | float | Current ELO rating |
| `peak_rating` | float | Highest ELO rating achieved |
| `rating_volatility` | float | Rating standard deviation |
| `rating_trend` | float | Recent rating trend |
| `match_count` | integer | Total matches played |
| `last_update` | datetime | Last rating update |
| `surface_ratings` | object | Surface-specific ratings |

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `400` | Bad Request - Invalid input data |
| `404` | Not Found - Resource doesn't exist |
| `422` | Unprocessable Entity - Validation error |
| `429` | Too Many Requests - Rate limit exceeded |
| `500` | Internal Server Error - Server issue |
| `503` | Service Unavailable - Models not loaded |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit**: 100 requests per minute
- **Burst Limit**: 20 requests in 10 seconds
- **Headers**: Rate limit info in response headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1632165600
```

## Caching

The API uses Redis caching for improved performance:

- **Prediction Cache**: 5 minutes TTL
- **Player Stats Cache**: 1 hour TTL
- **Rankings Cache**: 6 hours TTL

## Examples

### Python Client

```python
import requests
import json

class TennisPredictor:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict_match(self, player1_id, player1_name, 
                     player2_id, player2_name, surface):
        """Predict match outcome."""
        data = {
            "player1": {"player_id": player1_id, "name": player1_name},
            "player2": {"player_id": player2_id, "name": player2_name},
            "surface": surface
        }
        
        response = requests.post(f"{self.base_url}/predict", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_player_stats(self, player_id):
        """Get player statistics."""
        response = requests.get(f"{self.base_url}/player/{player_id}/stats")
        response.raise_for_status()
        return response.json()
    
    def get_rankings(self, surface="overall", limit=20):
        """Get surface rankings."""
        response = requests.get(
            f"{self.base_url}/rankings/{surface}",
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

# Usage
predictor = TennisPredictor()

# Make prediction
result = predictor.predict_match(
    "novak_djokovic", "Novak Djokovic",
    "carlos_alcaraz", "Carlos Alcaraz", 
    "Hard"
)

print(f"Djokovic win probability: {result['player1_win_probability']:.1%}")

# Get player stats
stats = predictor.get_player_stats("novak_djokovic")
print(f"Current rating: {stats['current_rating']:.0f}")

# Get clay court rankings
rankings = predictor.get_rankings("clay", limit=10)
for player in rankings['rankings']:
    print(f"{player['rank']}. {player['player_id']}: {player['rating']:.0f}")
```

### JavaScript Client

```javascript
class TennisPredictor {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async predictMatch(player1, player2, surface) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({player1, player2, surface})
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    async getPlayerStats(playerId) {
        const response = await fetch(`${this.baseUrl}/player/${playerId}/stats`);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
}

// Usage
const predictor = new TennisPredictor();

// Make prediction
const prediction = await predictor.predictMatch(
    {player_id: 'novak_djokovic', name: 'Novak Djokovic'},
    {player_id: 'carlos_alcaraz', name: 'Carlos Alcaraz'},
    'Hard'
);

console.log(`Win probability: ${(prediction.player1_win_probability * 100).toFixed(1)}%`);
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Match prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "player1": {"player_id": "novak_djokovic", "name": "Novak Djokovic"},
    "player2": {"player_id": "carlos_alcaraz", "name": "Carlos Alcaraz"},
    "surface": "Hard",
    "tournament": "US Open",
    "round_info": "F"
  }'

# Player statistics
curl -X GET "http://localhost:8000/player/novak_djokovic/stats"

# Clay court rankings (top 10)
curl -X GET "http://localhost:8000/rankings/clay?limit=10"
```

## Production Deployment

### Environment Variables

```bash
# API Configuration
TENNIS_ENV=production
SECRET_KEY=your_secret_key_here

# Data Sources
ODDS_API_KEY=your_odds_api_key
WEATHER_API_KEY=your_weather_api_key

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/tennis
REDIS_URL=redis://localhost:6379

# Monitoring
WANDB_API_KEY=your_wandb_key
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "api/prediction_server.py"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - TENNIS_ENV=production
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## Monitoring

The API provides built-in monitoring capabilities:

- **Health checks**: `/health` endpoint
- **Metrics**: Request counts, response times, error rates
- **Logging**: Structured JSON logs
- **Alerts**: Configurable thresholds for accuracy, latency

Monitor key metrics:
- Prediction accuracy over time
- API response times
- Model freshness
- Cache hit rates
- Error rates by endpoint