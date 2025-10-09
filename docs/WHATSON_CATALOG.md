# WhatsOn Catalog Creation

## Overview

This document describes the complete pipeline for creating a deduplicated WhatsOn movie catalog with unique catalog IDs, similar to the reference notebook implementation.

## Pipeline Stages

The catalog creation consists of 3 main stages:

```
Raw CSV → Filter Movies → Enrich TMDB → Create Catalog
```

### Stage 1: Movie Extraction
**Function**: `run_whatson_extraction_pipeline()`

Filters movies from the raw WhatsOn CSV using 11 filtering rules (see [WHATSON_EXTRACTION.md](WHATSON_EXTRACTION.md)).

**Output**: `whatson_catalogue.parquet` - ~11,000 filtered movies

### Stage 2: TMDB Enrichment
**Function**: `run_enrich_catalog_with_movie_metadata_pipeline()`

Enriches movies with TMDB metadata:
- Searches for best TMDB match by title + duration
- Extracts features: genres, release_date, vote_average, popularity, revenue, etc.
- Marks movies without TMDB match as `missing_tmdb_id = True`

**Output**: `whatson_catalogue_enriched_tmdb.parquet`

### Stage 3: Catalog Creation
**Function**: `run_whatson_catalog_creation_pipeline()`

Creates the final deduplicated catalog:

1. **Deduplication**
   - Scores each row by number of non-null fields (`info_score`)
   - Groups by `processed_title` (from TMDB)
   - Keeps row with highest info score (most complete data)

2. **Catalog ID Assignment**
   - Uses TMDB ID when available (e.g., `"220"` for East of Eden)
   - Assigns custom IDs for missing TMDB: `WO_000000`, `WO_000001`, etc.
   - Prefix: `WHATSON_CATALOG_ID_PREFIX = "WO_"`

3. **Date Conversion**
   - Converts date columns to proper datetime format:
     - `start_rights`, `end_rights` (broadcasting rights)
     - `date_diff_1`, `date_last_diff` (last diffusion dates)
     - `date_rediff_1` through `date_rediff_4` (rediffusion dates)

4. **Computed Features**
   - `times_shown`: Initialize to 0 (will be updated from programming data)
   - `movie_age`: Calculate from `release_date` (years since release)

5. **Indexing**
   - Sets `catalog_id` as DataFrame index
   - Enables fast lookups by catalog ID

**Output**: `whatson_catalog.parquet` - ~9,300 unique movies indexed by catalog_id

## Usage

### CLI Command

```bash
cts-reco-extract-whatson \
  --csv data/raw/original_raw_whatson.csv \
  --out_catalog data/processed/whatson_catalog.parquet
```

This runs all 3 stages automatically.

### Python API

```python
from pathlib import Path
from cts_recommender.pipelines import whatson_extraction_pipeline

# Run complete pipeline
catalog_df, catalog_file = whatson_extraction_pipeline.run_whatson_catalog_creation_pipeline(
    enriched_file=Path("data/processed/whatson_catalogue_enriched_tmdb.parquet"),
    out_file=Path("data/processed/whatson_catalog.parquet")
)

# Catalog is indexed by catalog_id
print(catalog_df.head())
```

### Stage-by-Stage

```python
# Stage 1: Extract movies
movies_df, _ = whatson_extraction_pipeline.run_whatson_extraction_pipeline(
    raw_file=Path("data/raw/original_raw_whatson.csv")
)

# Stage 2: Enrich with TMDB
enriched_df, _ = whatson_extraction_pipeline.run_enrich_catalog_with_movie_metadata_pipeline()

# Stage 3: Create catalog
catalog_df, _ = whatson_extraction_pipeline.run_whatson_catalog_creation_pipeline()
```

## Catalog Schema

### Catalog ID Format
- **TMDB movies**: Use TMDB ID directly (`"220"`, `"11806"`, etc.)
- **Non-TMDB movies**: Custom format `WO_XXXXXX` (e.g., `WO_000042`)

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `catalog_id` (index) | str | Unique catalog identifier |
| `title` | str | Display title |
| `processed_title` | str | Standardized title from TMDB |
| `tmdb_id` | float | TMDB movie ID (NaN for custom IDs) |
| `missing_tmdb_id` | bool | True if no TMDB match found |
| `genres` | list[dict] | Movie genres from TMDB |
| `release_date` | str | Release date (YYYY-MM-DD) |
| `vote_average` | float | TMDB user rating |
| `popularity` | float | TMDB popularity score |
| `start_rights` | datetime | Broadcasting rights start |
| `end_rights` | datetime | Broadcasting rights end |
| `movie_age` | int | Years since release |
| `times_shown` | int | Number of times broadcast (default: 0) |

## Implementation Details

### Deduplication Logic

```python
# Score rows by completeness
df['info_score'] = df.notna().sum(axis=1)

# Keep most complete record per title
catalog_df = df.sort_values('info_score', ascending=False) \
               .drop_duplicates(subset=['processed_title'], keep='first')
```

### Catalog ID Assignment

```python
# Use TMDB ID as string
catalog_df['catalog_id'] = catalog_df['tmdb_id'].astype('Int64').astype(str)

# Assign custom IDs for missing TMDB
missing_mask = catalog_df['missing_tmdb_id'] == True
catalog_df.loc[missing_mask, 'catalog_id'] = [
    f"{RTS_constants.WHATSON_CATALOG_ID_PREFIX}{i:06d}"
    for i in range(missing_mask.sum())
]
```

### Date Conversion

```python
date_cols = ['start_rights', 'end_rights', 'date_diff_1', 'date_last_diff',
             'date_rediff_1', 'date_rediff_2', 'date_rediff_3', 'date_rediff_4']

for col in date_cols:
    if col in catalog_df.columns:
        catalog_df[col] = pd.to_datetime(catalog_df[col], errors='coerce')
```

### Movie Age Calculation

```python
release_dates = pd.to_datetime(catalog_df['release_date'], errors='coerce')
catalog_df['movie_age'] = (pd.Timestamp.now() - release_dates).dt.days // 365
```

## Statistics

From a typical run:

- **Input records**: 17,336 (raw WhatsOn CSV)
- **After filtering**: ~11,000 movies
- **After enrichment**: ~11,000 with TMDB data
- **Final catalog**: ~9,300 unique movies
  - TMDB IDs: ~8,800 (94%)
  - Custom IDs: ~500 (6%)

## Files

- **Preprocessing**: [whatson_extraction.py](../src/cts_recommender/preprocessing/whatson_extraction.py)
  - `create_whatson_catalog()` - Main catalog creation function
- **Pipeline**: [whatson_extraction_pipeline.py](../src/cts_recommender/pipelines/whatson_extraction_pipeline.py)
  - `run_whatson_catalog_creation_pipeline()` - Stage 3 orchestration
- **CLI**: [whatson_extraction.py](../src/cts_recommender/cli/whatson_extraction.py)
  - Command-line interface
- **Constants**: [RTS_constants.py](../src/cts_recommender/RTS_constants.py)
  - `WHATSON_CATALOG_ID_PREFIX = "WO_"`

## Next Steps

1. **Update `times_shown`**: Count actual broadcasts from programming data
2. **Validation**: Cross-check catalog IDs with programming schedule
3. **Rights filtering**: Add queries for movies with valid broadcasting rights
4. **Export formats**: Add JSON/CSV exports for downstream systems

---

**Related Documentation**:
- [WhatsOn Extraction Rules](WHATSON_EXTRACTION.md)
- [Pipeline Architecture](PIPELINE.md)
