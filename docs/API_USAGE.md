# API Usage Guide

Quick guide for running and using the RTS Curator Recommendation API.

## Setup

### 1. Install API Dependencies

```bash
# Install API dependencies
uv sync --group api

# Or install everything (api + dev tools)
uv sync --all-groups
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
# At minimum, set your TMDB API key:
# APP_TMDB__API_KEY=your_key_here
```

### 3. Ensure Models and Data Are Ready

Make sure you've run the data pipelines:

```bash
make prepare-programming    # Process programming data
make extract-whatson       # Extract WhatsOn catalog
make historical-programming # Match broadcasts to catalog
make train-audience-ratings # Train audience model
make extract-IL-training-data # Extract IL training samples
# Train CTS model (via interactive testing or pipeline)
```

## Running the API

### Development (Local)

```bash
# Run with auto-reload (default .env settings)
python -m api.main

# Or with uvicorn directly
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:
- **API**: http://127.0.0.1:8000
- **Interactive Docs**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/api/v1/health/ready

### Production

```bash
# Use production env file
cp .env.prod.example .env

# Edit .env with production settings
# Then run:
python -m api.main
```

## Environment Configuration

### Development (.env)
```bash
APP_ENV=dev
APP_API_HOST=127.0.0.1  # localhost only
APP_API_PORT=8000
APP_API_RELOAD=true
APP_CORS_ORIGINS=["*"]
```

### Production (.env)
```bash
APP_ENV=prod
APP_API_HOST=0.0.0.0  # all interfaces
APP_API_PORT=8000
APP_API_RELOAD=false
APP_API_WORKERS=4
APP_CORS_ORIGINS=["https://yourdomain.com"]
```

### Docker
```bash
APP_ENV=prod
APP_API_HOST=0.0.0.0
APP_DATA_ROOT=/app/data
APP_MODELS_DIR=/app/data/models
```

## API Endpoints

### Health Check

**GET /api/v1/health/ready**
```bash
curl http://127.0.0.1:8000/api/v1/health/ready
```

**GET /api/v1/health/model-info**
```bash
curl http://127.0.0.1:8000/api/v1/health/model-info
```

### Generate Recommendations

**POST /api/v1/recommendations**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-08-15",
    "hour": 20,
    "channel": "RTS 1",
    "top_n": 10
  }'
```

**Response:**
```json
{
  "recommendations": [
    {
      "catalog_id": "550",
      "title": "Fight Club",
      "tmdb_id": 550,
      "genres": ["Drama"],
      "total_score": 0.85,
      "signal_contributions": {...},
      "signal_weights": {...},
      "predicted_audience": 250000
    }
  ],
  "context": {...},
  "model_version": "v1.0.0",
  "candidates_considered": 150
}
```

### Browse Catalog

**GET /api/v1/catalog**
```bash
# Browse all movies
curl http://127.0.0.1:8000/api/v1/catalog?limit=10

# Filter by genre
curl http://127.0.0.1:8000/api/v1/catalog?genre=Drama&limit=20

# Search titles
curl http://127.0.0.1:8000/api/v1/catalog?search=inception
```

### Submit Feedback (Future)

**POST /api/v1/feedback**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-08-15",
    "hour": 20,
    "channel": "RTS 1",
    "selected_movie_id": "550",
    "unselected_movie_ids": ["551", "552"]
  }'
```

### Scrape Competition Schedules

**POST /api/v1/competition/sync**
```bash
curl -X POST http://127.0.0.1:8000/api/v1/competition/sync \
  -H "Content-Type: application/json" \
  -d '{
    "days_ahead": 21
  }'
```

## Interactive Documentation

Visit http://127.0.0.1:8000/docs for:
- Interactive API explorer
- Request/response schemas
- Try out endpoints directly in browser

## Architecture

```
┌─────────────┐
│   Client    │  (Frontend, curl, Postman)
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────────────────────┐
│   FastAPI (api/main.py)     │
│   - CORS middleware         │
│   - Request validation      │
└──────┬──────────────────────┘
       │
       ↓
┌─────────────────────────────┐
│   Routers (api/routers/)    │
│   - recommendations.py      │
│   - feedback.py             │
│   - competition.py          │
│   - catalog.py              │
│   - health.py               │
└──────┬──────────────────────┘
       │
       ↓
┌─────────────────────────────┐
│   Services (api/services/)  │
│   - recommendation_service  │
│   - competition_service     │
└──────┬──────────────────────┘
       │
       ↓
┌─────────────────────────────┐
│   Dependencies              │
│   - AppState (singleton)    │
│   - CTS Model               │
│   - TVProgrammingEnvironment│
│   - Catalog DataFrame       │
└──────┬──────────────────────┘
       │
       ↓
┌─────────────────────────────┐
│   Repository (data access)  │
│   - LocalFileRepository     │
│   - (Future: DatabaseRepo)  │
└──────┬──────────────────────┘
       │
       ↓
┌─────────────────────────────┐
│   Data Storage              │
│   - Parquet files           │
│   - Joblib models           │
│   - (Future: PostgreSQL)    │
└─────────────────────────────┘
```

## Troubleshooting

### Models Not Loading

```bash
# Check if models exist
ls data/models/

# Should see:
# - cts_model.npz
# - cts_model.rng.pkl
# - audience_ratings_model.joblib
# - curator_logistic_model.joblib
```

### CORS Issues

Update `.env`:
```bash
APP_CORS_ORIGINS=["http://localhost:3000", "http://127.0.0.1:3000"]
```

### Port Already in Use

Change port in `.env`:
```bash
APP_API_PORT=8001
```
