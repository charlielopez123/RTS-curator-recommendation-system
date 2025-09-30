# API Reference

## Pipeline Functions

### `programming_pipeline.py`

**`run_original_Rdata_programming_pipeline(raw_file, out_file)`**
- Loads R data file and preprocesses programming data
- Returns: `(DataFrame, Path)`

**`run_enrich_programming_with_movie_metadata_pipeline(processed_file, out_file)`**
- Enriches movies with TMDB metadata
- Returns: `(DataFrame, Path)`

**`run_ML_feature_processing_pipeline(processed_file, enriched_file, out_file)`**
- Creates ML-ready feature matrix
- Returns: `(DataFrame, Path)`

## Core Processing Functions

### `ML_feature_processing.py`

**`load_processed_and_enriched_programming(data_path_processed, data_path_enriched)`**
- Loads both parquet files
- Returns: `(processed_df, enriched_df)`

**`build_X_features(processed_df, df_enriched)`**
- Creates feature matrix from both datasets
- Returns: `DataFrame` (ML-ready features)

### `programming.py`

**`load_original_programming(file_path)`**
- Loads .RData file using rpy2
- Returns: `DataFrame`

**`preprocess_programming(df_raw)`**
- Filters and engineers features
- Returns: `DataFrame`

### `movie_features.py`

**`enrich_programming_with_movie_metadata(df_processed)`**
- Adds TMDB data to movie records
- Returns: `DataFrame`

## TMDB API

### `tmdb.TMDB_API`

**`find_best_match(title, known_runtime)`**
- Finds best TMDB match for movie
- Returns: `int` (movie_id) or `None`

**`get_movie_features(movie_id)`**
- Extracts features from TMDB
- Returns: `Dict[str, Any]`