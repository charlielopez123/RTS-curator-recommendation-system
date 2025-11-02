# Film Recommender System - FastAPI Architecture

## Overview

This is a FastAPI-based film recommendation system that uses a Contextual Thompson Sampling algorithm to provide personalized movie recommendations based on temporal context (date/hour). The system integrates with Supabase for storage and database management, and implements continuous learning through user feedback.

## Application Structure

```
app/
├── main.py                  # FastAPI application factory and startup
├── routers/                 # API endpoints (controllers)
│   ├── infer.py            # Recommendation generation
│   ├── feedback.py         # User feedback and parameter updates
│   ├── model_management.py # Model versioning and management
│   ├── competition.py      # Competitor data scraping and sync
│   └── health.py           # Health checks and readiness probes
├── services/                # Business logic layer
│   ├── recommender.py      # Core recommendation logic
│   ├── model_version_manager.py  # Model versioning system
│   ├── data_loader.py      # Supabase data loading utilities
│   ├── model_initializer.py
│   ├── model_registrar.py
│   └── plot_utils.py       # Visualization utilities
└── schemas/                 # Pydantic models for request/response
    ├── infer.py            # Recommendation schemas
    └── feedback.py         # Feedback schemas
```

## Core Components

### 1. Application Entry Point (`main.py`)

**Purpose:** Application factory and configuration

**Key Features:**
- Creates FastAPI app with CORS middleware
- Mounts API routers with prefixes
- Implements background preloading of heavy environment on startup
- Lazy initialization to allow port binding before full model load

```python
def create_app() -> FastAPI:
    app = FastAPI(title="Film Recommender API")
    # CORS configuration
    # Router mounting
    # Startup event handler for background preload
    return app
```

**Mounted Routers:**
- `/infer` - Recommendation generation
- `/` - Feedback endpoints (update-parameters, regenerate)
- `/models` - Model management endpoints
- `/competition` - Competition data endpoints
- `/health` - Health check endpoints

### 2. Services Layer

#### 2.1 Recommender Service (`services/recommender.py`)

**Purpose:** Core business logic for recommendations and model updates

**Key Components:**

**Environment Management:**
- `env`: Global singleton instance of `TVProgrammingEnvironment`
- Lazy initialization pattern with state tracking (`_env_ready`, `_env_init_started`)
- `preload_env()`: Async background initialization
- `get_env()`: Blocking initialization getter

**Core Functions:**

1. **`infer_logic(req: InferRequest)`**
   - Generates top-N movie recommendations for a given date/hour context
   - Excludes previously selected movies (from Supabase `selection_events` table)
   - Uses Contextual Thompson Sampling to score candidates
   - Returns recommendations with:
     - Movie metadata
     - Signal contributions (curator, audience, competition, diversity, novelty, rights)
     - Signal weights
     - Predicted audience
     - Model version info

2. **`update_signals_logic(req: UpdateParametersRequest)`**
   - Applies user feedback (selected movie) to update CTS parameters
   - Updates CTS with positive reward (r=1) for selected movie
   - Updates CTS with negative reward (r=0) for unselected movies
   - Triggers auto-save of model version after N selections (via `ModelVersionManager`)
   - Returns updated parameter statistics

3. **`regenerate_logic(req: RegenerateRequest)`**
   - Applies negative feedback for unselected movies
   - Excludes rejected movies from pool (per-context exclusion tracking)
   - Generates fresh recommendations from remaining candidates
   - Maintains exclusion state across regenerations within same context
   - Supports "abandon context" to clear exclusions

**Key Design Pattern:**
- Singleton environment instance shared across requests
- Context-based exclusion tracking: `excluded_ids_by_context[context_key]`
- String-based ID matching to handle mixed ID types

#### 2.2 Model Version Manager (`services/model_version_manager.py`)

**Purpose:** Automatic model versioning and persistence

**Key Features:**
- Semantic versioning (v1.0.0, v1.0.1, etc.)
- Automatic checkpoint creation every N selections
- Supabase Storage integration for model files
- Database metadata tracking in `model_parameter_logs` table
- Local backup storage for fallback

**Core Methods:**
1. `record_selection()` - Tracks selection count, triggers save
2. `save_model_version()` - Serializes CTS to NPZ, uploads to Supabase
3. `load_model_version()` - Downloads and deserializes CTS from version
4. `get_version_history()` - Lists all versions with metadata
5. `get_current_version_info()` - Returns active version details

**Persistence:**
- Model files: Supabase Storage (`models/versions/{version}/contextual_thompson_sampler.npz`)
- Metadata: Supabase table (`model_parameter_logs`)
- Backup: Local filesystem (`models/versions/{version}/`)

#### 2.3 Data Loader (`services/data_loader.py`)

**Purpose:** Abstraction layer for loading data from Supabase Storage

**Features:**
- Unified interface for loading parquet, pickle, NPZ files
- Automatic fallback to local files if Supabase fails
- Tracks data source (Supabase vs local) for debugging
- Supports strict mode (fails if Supabase unavailable)

**Methods:**
- `load_parquet_from_storage()` - Loads pandas DataFrames
- `load_pickle_from_storage()` - Loads pickled objects
- `load_model_from_storage()` - Loads ML models
- `load_npz_from_storage()` - Loads numpy arrays

### 3. API Routers (Controllers)

#### 3.1 Infer Router (`routers/infer.py`)

**Endpoint:** `POST /infer/`

**Purpose:** Generate movie recommendations

**Flow:**
1. Updates competition historical data (non-blocking)
2. Calls `infer_logic()` with date/hour context
3. Returns list of `Recommendation` objects with metadata

#### 3.2 Feedback Router (`routers/feedback.py`)

**Endpoints:**

1. **`POST /update-parameters`**
   - Applies user selection feedback
   - Updates CTS parameters based on selected movie
   - Triggers model versioning if threshold reached

2. **`POST /regenerate`**
   - Marks movies as rejected (r=0)
   - Excludes rejected movies from pool
   - Returns new slate of recommendations

#### 3.3 Model Management Router (`routers/model_management.py`)

**Endpoints:**

1. **`GET /models/versions`** - List all model versions
2. **`POST /models/load-version`** - Load specific version
3. **`GET /models/current-version`** - Get active version info
4. **`POST /models/create-checkpoint`** - Manual checkpoint creation
5. **`DELETE /models/versions/{version_name}`** - Delete version
6. **`POST /models/admin/reset-test-data`** - Reset system state
7. **`GET /models/weight-marginals-url`** - Get visualization URL

#### 3.4 Competition Router (`routers/competition.py`)

**Endpoints:**

1. **`POST /competition/sync`** - Scrape competitor schedules, upload to Supabase
2. **`POST /competition/upload`** - Manual XML upload
3. **`POST /competition/schedules`** - Download XMLs, parse, return flattened schedules

**Purpose:** Manage competitor TV schedule data for competition signal calculation

#### 3.5 Health Router (`routers/health.py`)

**Endpoints:**

1. **`GET /health/model-sources`** - Reports data source locations (Supabase vs local)
2. **`GET /health/ready`** - Readiness probe for environment initialization

### 4. Data Models (Schemas)

#### 4.1 Inference Schemas (`schemas/infer.py`)

```python
class InferRequest(BaseModel):
    date: str
    hour: int

class Recommendation(BaseModel):
    movie_id: Union[int, str]
    title: str
    poster_path: Optional[str]
    director: Optional[str]
    actors: Optional[str]
    start_rights: Optional[str]
    end_rights: Optional[str]
    total_score: float
    signal_contributions: Dict[str, float]  # Raw signal values
    signal_weights: Dict[str, float]        # CTS weights
    weighted_signals: Dict[str, float]      # Contributions × weights
    context_features: List[float]           # Context vector
    predicted_audience: Optional[float]     # Audience model prediction
```

#### 4.2 Feedback Schemas (`schemas/feedback.py`)

```python
class UpdateParametersRequest(BaseModel):
    selected_movie: MovieSelection
    unselected_movies: List[MovieSelection]
    timestamp: datetime

class RegenerateRequest(BaseModel):
    date: str
    hour: int
    unselected_movies: List[UnselectedMovie]
    generate_new_slate: bool = True
    abandon_context: bool = False
```

## Key Design Patterns

### 1. Singleton Pattern
- **Environment Instance:** Single `env` instance shared across all requests
- **Data Loader:** Single `data_loader` instance for storage access

### 2. Lazy Initialization
- Environment loads on first request, not at import time
- Allows fast app startup and port binding
- Background preload task for production

### 3. Strategy Pattern
- Feedback logic separated into `update_signals_logic` and `regenerate_logic`
- Different update strategies for selection vs rejection

### 4. Repository Pattern
- `DataLoader` abstracts storage backend (Supabase vs local)
- `ModelVersionManager` abstracts model persistence

### 5. Versioned State Management
- Automatic model checkpointing every N selections
- Semantic versioning for rollback capability
- Metadata tracking for model performance

## Data Flow

### Recommendation Flow

```
Client Request
    ↓
POST /infer/ (date, hour)
    ↓
infer_logic()
    ↓
1. Get context features from date/hour
2. Get available movies (rights + context constraints)
3. Exclude globally selected movies (Supabase query)
4. CTS samples weights: w_tilde = sample(U, context)
5. Score all candidates: score = w_tilde · signals
6. Select top-N movies
7. Build Recommendation objects
    ↓
Return JSON response with:
    - recommendations[]
    - model_version
    - model_stats
    - candidate_count
```

### Feedback Flow

```
Client submits selection
    ↓
POST /update-parameters
    ↓
update_signals_logic()
    ↓
1. Extract context features from selected movie
2. Extract signal values
3. Update CTS with r=1 for selected movie
4. Update CTS with r=0 for unselected movies
5. Compute parameter statistics
6. Check if auto-save threshold reached
7. If yes: save_model_version() → Supabase
    ↓
Return success + updated stats
```

### Model Versioning Flow

```
Selection recorded
    ↓
record_selection() → count++
    ↓
count % threshold == 0?
    ↓ Yes
save_model_version()
    ↓
1. Generate version name (v1.0.N)
2. Serialize CTS to NPZ (U, b, h_U, h_b, hyperparams)
3. Upload to Supabase Storage (models/versions/{version}/)
4. Insert metadata to model_parameter_logs table
5. Save local backup
6. Generate weight marginal plots (non-blocking)
    ↓
New version active
```

## Integration Points

### Supabase Integration

**Storage Buckets:**
- `models` - Model NPZ files, plot images
- `movie-data` - Catalog, historical data
- `ml-models` - Audience prediction models
- `competition-xml` - Competitor schedule XMLs

**Database Tables:**
- `selection_events` - User selection history (feedback)
- `model_parameter_logs` - Model version metadata

### External Dependencies

**ML Environment:**
- `TVProgrammingEnvironment` (from `envs.env`) - Core RL environment
- `ContextualThompsonSampler` (from `contextual_thompson`) - Bandit algorithm
- Audience prediction model (Random Forest)

**Data Processing:**
- `CompetitorDataScraper` - XML parsing for competitor schedules
- Preprocessing utilities for feature engineering

## Key Configuration

**Environment Variables:**
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_ANON_KEY` - Public API key
- `SUPABASE_SERVICE_ROLE_KEY` - Admin key for write access
- `REQUIRE_SUPABASE_STORAGE` - Strict mode flag

**Model Parameters:**
- `selections_threshold: int = 3` - Auto-save frequency
- `num_signals: int = 6` - Number of recommendation signals
- Signal names: ["curator", "audience", "competition", "diversity", "novelty", "rights"]

## Recommendations for Refactoring

When creating a clean version, consider:

1. **Separation of Concerns:**
   - Extract environment initialization into dedicated service
   - Separate CTS logic from API logic
   - Create dedicated repository classes for each data source

2. **Dependency Injection:**
   - Pass `env` and `model_version_manager` as dependencies instead of globals
   - Use FastAPI dependency injection system

3. **Error Handling:**
   - Standardize error responses
   - Add custom exception classes
   - Implement proper logging

4. **Testing:**
   - Separate business logic from FastAPI for easier unit testing
   - Mock Supabase client for integration tests

5. **Configuration Management:**
   - Centralize configuration in settings object
   - Use environment-specific configs (dev/staging/prod)

6. **API Versioning:**
   - Add `/v1/` prefix to all routes for future versioning

7. **Documentation:**
   - Add OpenAPI schema descriptions
   - Include example requests/responses
   - Document error codes

8. **Monitoring:**
   - Add structured logging
   - Implement metrics collection (request latency, model performance)
   - Add distributed tracing
