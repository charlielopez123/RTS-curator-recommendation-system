# Pipeline Architecture

The system has 5 independent pipelines. Each pipeline:
- Lives in `src/cts_recommender/pipelines/`
- Has a corresponding CLI script in `src/cts_recommender/cli/`
- Returns `(dataframe, output_path)` tuple
- Uses atomic writes (temp file → rename)
- Has default paths from `settings.py`

## Pipeline Structure Pattern

```python
def run_pipeline(
    input_file: Optional[Path] = None,
    out_file: Optional[Path] = None
) -> tuple[pd.DataFrame, Path]:
    """
    Pipeline docstring explaining what it does.

    Parameters: input files, output files (all optional with defaults)
    Returns: (processed_df, output_path)
    """
    # 1. Set defaults from settings
    if input_file is None:
        input_file = cfg.some_default_path
    if out_file is None:
        out_file = cfg.some_output_path

    # 2. Log start
    logger.info("=== Pipeline Name ===")

    # 3. Load data
    df = readers.read_format(input_file)

    # 4. Process data (call preprocessing functions)
    processed_df = preprocessing_module.do_work(df)

    # 5. Save atomically
    atomic_write_parquet(processed_df, out_file)

    # 6. Log completion
    logger.info(f"Output: {out_file}")

    return processed_df, out_file
```

## 1. Programming Processing Pipeline

**File**: `programming_processing_pipeline.py`

**Purpose**: Process RTS programming data in 3 stages

**Stages**:
1. `run_original_Rdata_programming_pipeline()` - Load RData, filter, extract movies
2. `run_enrich_programming_with_movie_metadata_pipeline()` - Add TMDB metadata
3. `run_ML_feature_processing_pipeline()` - Build ML feature matrix

**Key preprocessing**:
- `preprocessing/programming.py` - Load/preprocess RData (uses `pyreadr`)
- `preprocessing/movie_features.py` - TMDB enrichment
- `preprocessing/ML_feature_processing.py` - Feature engineering

**Outputs**:
- `programming.parquet` - Preprocessed
- `programming_enriched.parquet` - With TMDB
- `ML_features.parquet` - ML-ready

**Feature order matters**: See `features/schemas.py` for sklearn compatibility

## 2. WhatsOn Extraction Pipeline

**File**: `whatson_extraction_pipeline.py`

**Purpose**: Build movie catalog from WhatsOn CSV

**Process**:
1. Load CSV with movie showings
2. Extract unique movies
3. Enrich with TMDB
4. Build catalog with features

**Key preprocessing**:
- `preprocessing/whatson_extraction.py` - CSV parsing
- `preprocessing/catalog_features.py` - Feature building

**Output**: `whatson_catalog.parquet` (indexed by `catalog_id`)

**Schema**: See `features/whatson_schema.py` and `features/catalog_schema.py`

## 3. Historical Programming Pipeline

**File**: `historical_programming_pipeline.py`

**Purpose**: Match TV broadcasts to catalog (fact table)

**Process**:
1. Load enriched programming (has TMDB IDs)
2. Load catalog (indexed by catalog_id)
3. Match via TMDB ID
4. Create fact table: broadcast metadata + catalog_id link

**Key preprocessing**:
- `preprocessing/historical_programming.py` - Matching logic

**Output**: `historical_programming.parquet`

**Schema**: Broadcast facts (date, channel, time, catalog_id, audience metrics)
- **No TMDB features** - those live in catalog
- Join with catalog to get movie features

## 4. Audience Ratings Pipeline

**File**: `audience_ratings_regression_pipeline.py`

**Purpose**: Train regression model for audience ratings

**Process**:
1. Load ML features
2. Train/test split
3. Train RandomForest regressor
4. Evaluate metrics
5. Save model

**Key modules**:
- `models/training.py` - Training logic
- `models/audience_ratings_regressor.py` - Model wrapper

**Output**: `audience_ratings_model.joblib`

**Target**: `rt_m` (audience ratings)

## 5. IL Training Data Pipeline

**File**: `IL_training_data_pipeline.py`

**Purpose**: Extract training samples for imitation learning

**Process**:
1. Load historical programming + catalog + audience model
2. Create TV environment
3. Extract positive samples (curator decisions)
4. Generate negative samples (alternatives)
5. Compute pseudo-rewards
6. Save raw samples + numpy arrays

**Key modules**:
- `imitation_learning/IL_training.py` - HistoricalDataProcessor
- `environments/TV_environment.py` - Context/feature extraction
- `environments/reward.py` - Reward computation

**Outputs**:
- `training_samples.joblib` - Raw dicts (for viz)
- `training_data.joblib` - Numpy arrays (for training)

**Parameters**:
- `gamma`: Weight curator decisions vs pseudo-rewards
- `negative_sampling_ratio`: Negatives per positive
- `time_split_date`: Optional train/val split

## Common Patterns

### Data Loading
- Use `io/readers.py` - handles R, JSON, Parquet
- Use `io/writers.py` - atomic writes

### Settings
```python
from cts_recommender.settings import get_settings
cfg = get_settings()  # Singleton, reads .env once
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Message")
```

### TMDB Integration
- `adapters/tmdb/tmdb.py` - High-level API
- `adapters/tmdb/client.py` - Low-level HTTP
- Includes retry logic, rate limiting, fuzzy matching

### Feature Schemas
- `features/schemas.py` - ML feature definitions
- Order matters for sklearn models
- Base → Categorical → TMDB → Boolean → Computed → Genres

## Pipeline Dependencies

```
prepare-programming → ML_features.parquet → train-audience-ratings → model.joblib
                                                                         ↓
extract-whatson → catalog.parquet ─────────────────────────────────────→
         ↓                                                               ↓
         └─→ historical-programming.parquet → extract-IL-training-data
```

Run in this order:
1. `prepare-programming` (generates ML features)
2. `extract-whatson` (builds catalog)
3. `historical-programming` (matches broadcasts to catalog)
4. `train-audience-ratings` (trains model on ML features)
5. `extract-IL-training-data` (needs historical + catalog + model)

## Error Handling

Each pipeline includes:
- File existence validation
- Column presence checks
- TMDB API error handling (retry logic)
- Logging for debugging

## Performance Notes

- **Parquet**: Efficient binary format
- **Atomic writes**: Temp file → rename (prevents corruption)
- **TMDB rate limits**: Built-in respect for API limits
- **Memory**: Use chunking for very large datasets if needed
- **Parallelization**: Could parallelize TMDB enrichment (not yet implemented)
