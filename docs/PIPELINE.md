# Pipeline Architecture

This document explains the RTS Curator Recommendation System's data processing pipeline architecture, stages, and how they work together.

## Overview

The pipeline consists of three main stages that transform raw RTS programming data into ML-ready features:

```
Raw R Data → Preprocessed → TMDB Enriched → ML Features
```

Each stage is implemented as a separate pipeline function that can be run independently or as part of the complete workflow.

## Pipeline Functions

### Core Pipeline Functions

All pipeline functions are located in `src/cts_recommender/pipelines/programming_pipeline.py`:

```python
from cts_recommender.pipelines import programming_pipeline

# Stage 1: Raw data preprocessing
run_original_Rdata_programming_pipeline()

# Stage 2: TMDB metadata enrichment
run_enrich_programming_with_movie_metadata_pipeline()

# Stage 3: ML feature preparation
run_ML_feature_processing_pipeline()
```

## Stage 1: Raw Data Preprocessing

**Function**: `run_original_Rdata_programming_pipeline()`

**Purpose**: Convert raw RTS programming data into clean, feature-engineered format

### Input
- **File**: `.RData` file containing RTS programming schedule
- **Format**: R data frame with Swiss broadcasting data
- **Columns**: Date, Title, Description, Channel, Times, Classifications, Metrics

### Process

1. **Data Loading** (`programming.load_original_programming()`)
   - Loads R data file using `rpy2`
   - Validates expected column structure
   - Handles R-specific data types

2. **Data Filtering** (`programming.preprocess_programming()`)
   - **Audience Filter**: Keep only "Personnes 3+" (Adults 3+)
   - **Activity Filter**: Keep only "Overnight+7" (Live + 7-day replay)
   - **Data Quality**: Remove records with missing critical fields

3. **Column Standardization**
   - Rename columns from French/R format to Python conventions
   - Convert data types (dates, durations, numeric metrics)
   - Standardize text fields

4. **Feature Engineering**
   - **Temporal Features**:
     - `hour`: Hour of broadcast (0-23)
     - `weekday`: Day of week (0-6)
     - `is_weekend`: Boolean weekend indicator
     - `season`: Meteorological season (Spring, Summer, Fall, Winter)
   - **Duration Features**:
     - `duration_min`: Program duration in minutes
   - **Holiday Detection**:
     - `public_holiday`: Boolean using Swiss holiday calendar
     - Uses reference data from `data/reference/holidays.json`

### Output
- **File**: `data/processed/programming.parquet`
- **Format**: Clean pandas DataFrame with standardized features
- **Size**: Filtered to relevant content (~50-70% of original data)

### Code Example

```python
from pathlib import Path
from cts_recommender.pipelines import programming_pipeline

# Run preprocessing
processed_df, output_file = programming_pipeline.run_original_Rdata_programming_pipeline(
    raw_file=Path("data/raw/original_R_dataset.RData"),
    out_file=Path("data/processed/programming.parquet")
)

print(f"Processed {len(processed_df)} records")
print(f"Output saved to: {output_file}")
```

## Stage 2: TMDB Metadata Enrichment

**Function**: `run_enrich_programming_with_movie_metadata_pipeline()`

**Purpose**: Enrich movie records with comprehensive metadata from The Movie Database

### Input
- **File**: Preprocessed programming data (`programming.parquet`)
- **Dependencies**: TMDB API credentials in environment

### Process

1. **Movie Identification** (`movie_features.enrich_programming_with_movie_metadata()`)
   - **Classification-Based**: Use RTS `class_key` to identify movies
   - **Special Cases**: Handle movies with generic titles using description text
   - **Movie Types**: Feature films, TV movies, documentaries

2. **TMDB Matching** (`adapters.tmdb.tmdb.find_best_match()`)
   - **Title Search**: Primary matching by movie title
   - **Runtime Validation**: Secondary matching using program duration
   - **Fuzzy Matching**: Handle title variations and translations
   - **Decomposition**: Try title parts if full title fails

3. **Metadata Extraction** (`adapters.tmdb.tmdb.get_movie_features()`)
   - **Basic Info**: Adult rating, original language, popularity
   - **Financial Data**: Revenue, vote averages, vote counts
   - **Content Data**: Genres (as structured list), release dates
   - **Missing Indicators**: Track which movies have no TMDB match

4. **Data Integration**
   - Merge TMDB data with original programming records
   - Preserve non-movie content unchanged
   - Handle missing data with appropriate defaults

### TMDB API Integration

The system uses a robust TMDB client with:
- **Rate Limiting**: Respects API limits
- **Retry Logic**: Handles temporary failures
- **Error Handling**: Graceful degradation for missing data
- **Caching**: Potential for response caching (not yet implemented)

### Output
- **File**: `data/processed/programming_enriched.parquet`
- **Format**: Programming data + TMDB metadata for movies
- **New Columns**:
  - `adult`, `original_language`, `popularity`
  - `revenue`, `vote_average`, `genres`
  - `release_date`, `missing_tmdb`

## Stage 3: ML Feature Preparation

**Function**: `run_ML_feature_processing_pipeline()`

**Purpose**: Create ML-ready feature matrix with consistent schema for all content types

### Input
- **Files**: Both preprocessed and enriched datasets
- **Schema**: Feature definitions from `features.schemas.ML_FEATURES`

### Process

1. **Data Loading** (`ML_feature_processing.load_processed_and_enriched_programming()`)
   - Load both processed and enriched datasets
   - Separate movie vs non-movie content

2. **Feature Harmonization** (`ML_feature_processing.build_X_features()`)
   - **Non-Movie Defaults**: Add TMDB-style features with sensible defaults
   - **Missing Value Handling**: Fill nulls with appropriate values
   - **Type Consistency**: Ensure matching data types across content

3. **Feature Engineering**
   - **Genre Encoding**: One-hot encode movie genres
   - **Date Features**: Extract movie age from release dates
   - **Binary Indicators**: Create flags for missing data
   - **Content Type**: Add `is_movie` flag

4. **Schema Enforcement**
   - **Column Selection**: Keep only features defined in `ML_FEATURES`
   - **Order Consistency**: Ensure same column order for sklearn
   - **Type Conversion**: Convert booleans to integers (0/1)

5. **Data Quality Checks**
   - **NaN Detection**: Assert no missing values remain
   - **Type Validation**: Verify expected data types
   - **Feature Completeness**: Confirm all required features present

### Output
- **File**: `data/processed/ML_features.parquet`
- **Format**: ML-ready feature matrix



### Feature Categories

The final feature set includes:

1. **Programming Features** (8 features)
   - `rt_m`, `pdm`: Audience metrics
   - `hour`, `weekday`, `is_weekend`: Temporal
   - `duration_min`: Content length
   - `season`, `public_holiday`: Calendar

2. **Categorical Features** (2 features)
   - `channel`: Broadcasting channel
   - `original_language`: Content language

3. **TMDB Numerical** (3 features)
   - `popularity`, `revenue`, `vote_average`

4. **Boolean Indicators** (4 features)
   - `adult`, `missing_release_date`, `missing_tmdb`, `is_movie`

5. **Computed Features** (1 feature)
   - `movie_age`: Years since release

6. **Dynamic Genre Features** (~19 features)
   - One-hot encoded movie genres (Action, Comedy, etc.)

### Code Example

```python
# Run ML feature preparation
ml_df, output_file = programming_pipeline.run_ML_feature_processing_pipeline(
    processed_file=Path("data/processed/programming.parquet"),
    enriched_file=Path("data/processed/programming_enriched.parquet"),
    out_file=Path("data/processed/ML_features.parquet")
)

print(f"Feature matrix shape: {ml_df.shape}")
print(f"Features: {list(ml_df.columns)}")
```

## Complete Pipeline Execution

### Sequential Execution

```python
from pathlib import Path
from cts_recommender.pipelines import programming_pipeline

# Define paths
raw_file = Path("data/raw/original_R_dataset.RData")
processed_file = Path("data/processed/programming.parquet")
enriched_file = Path("data/processed/programming_enriched.parquet")
ml_file = Path("data/processed/ML_features.parquet")

# Stage 1: Preprocess raw data
processed_df, _ = programming_pipeline.run_original_Rdata_programming_pipeline(
    raw_file=raw_file,
    out_file=processed_file
)

# Stage 2: Enrich with TMDB
enriched_df, _ = programming_pipeline.run_enrich_programming_with_movie_metadata_pipeline(
    processed_file=processed_file,
    out_file=enriched_file
)

# Stage 3: Prepare ML features
ml_df, _ = programming_pipeline.run_ML_feature_processing_pipeline(
    processed_file=processed_file,
    enriched_file=enriched_file,
    out_file=ml_file
)

print("Pipeline completed successfully!")
```

### Error Handling

Each pipeline stage includes error handling for common issues:

- **File Access**: Validates input files exist and are readable
- **Data Quality**: Checks for expected columns and data types
- **API Issues**: Handles TMDB API failures gracefully
- **Memory**: Uses efficient pandas operations for large datasets

### Configuration

Pipeline behavior can be configured via settings:

```python
from cts_recommender.settings import get_settings

cfg = get_settings()
print(f"Data root: {cfg.data_root}")
print(f"TMDB API: {cfg.tmdb.api_base_url}")
```

## Performance Considerations

### Memory Usage
- **Chunking**: Large datasets may require processing in chunks
- **Parquet**: Efficient storage format reduces I/O overhead
- **Selective Loading**: Only load required columns when possible

### API Efficiency
- **Batch Processing**: TMDB enrichment processes movies sequentially
- **Rate Limiting**: Built-in respect for API rate limits
- **Caching**: Future enhancement for repeated pipeline runs

### Parallelization
- **Movie Enrichment**: Could be parallelized for better performance
- **Feature Engineering**: Most operations are vectorized pandas operations
- **I/O**: Atomic writes prevent data corruption

## Monitoring and Debugging

### Logging
Each pipeline stage provides detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Run pipeline with logging
processed_df, _ = programming_pipeline.run_original_Rdata_programming_pipeline(...)

```

### Common Issues
1. **TMDB API Key**: Verify credentials in `.env`
2. **Memory Limits**: Consider chunking for very large datasets
3. **Missing Holidays**: Ensure `holidays.json` exists
4. **R Dependencies**: Verify `rpy2` and R installation