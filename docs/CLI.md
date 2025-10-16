# CLI Reference

All commands are available via `make` or directly via `uv run`.

## 1. cts-reco-prepare-programming

Process RTS programming data: RData → preprocessed → TMDB enriched → ML features.

```bash
make prepare-programming

# Or with custom paths
uv run cts-reco-prepare-programming \
  --rdata data/raw/original_R_dataset.RData \
  --out_processed data/processed/programming/programming.parquet \
  --out_enriched data/processed/programming/programming_enriched.parquet \
  --out_ml data/processed/programming/ML_features.parquet
```

**Outputs:**
- `programming.parquet`: Preprocessed (filtered, temporal features)
- `programming_enriched.parquet`: With TMDB metadata
- `ML_features.parquet`: ML-ready feature matrix

## 2. cts-reco-extract-whatson

Extract WhatsOn catalog from CSV and enrich with TMDB.

```bash
make extract-whatson WHATSON_CSV=data/raw/original_raw_whatson.csv

# Or directly
uv run cts-reco-extract-whatson \
  --csv data/raw/original_raw_whatson.csv \
  --out_catalog data/processed/whatson/whatson_catalog.parquet
```

**Output:**
- `whatson_catalog.parquet`: Movie catalog with TMDB features

## 3. cts-reco-historical-programming

Match TV broadcasts to WhatsOn catalog.

```bash
make historical-programming

# Or directly
uv run cts-reco-historical-programming \
  --programming data/processed/programming/programming_enriched.parquet \
  --catalog data/processed/whatson/whatson_catalog.parquet \
  --out data/processed/programming/historical_programming.parquet
```

**Output:**
- `historical_programming.parquet`: Fact table (date, channel, catalog_id, audience)

## 4. cts-reco-train-audience-ratings

Train audience ratings regression model.

```bash
make train-audience-ratings

# Or directly
uv run cts-reco-train-audience-ratings \
  --ml_features data/processed/programming/ML_features.parquet \
  --model_output data/models/audience_ratings_model.joblib
```

**Output:**
- `audience_ratings_model.joblib`: Trained regression model

## 5. cts-reco-extract-IL-training-data

Extract imitation learning training samples.

```bash
make extract-IL-training-data

# Or with custom parameters
uv run cts-reco-extract-IL-training-data \
  --historical data/processed/programming/historical_programming.parquet \
  --catalog data/processed/whatson/whatson_catalog.parquet \
  --audience-model data/models/audience_ratings_model.joblib \
  --out data/processed/IL/training_data.joblib \
  --gamma 0.6 \
  --negative-ratio 5 \
  --split-date 2024-01-01
```

**Parameters:**
- `--gamma`: Weight for curator decisions vs pseudo-rewards (default: 0.6)
- `--negative-ratio`: Negative samples per positive (default: 5)
- `--split-date`: Optional train/val split date (YYYY-MM-DD)

**Outputs:**
- `training_samples.joblib`: Raw samples (for visualization)
- `training_data.joblib`: Numpy arrays (for model training)

## Pipeline Order

Run pipelines in this order:

```bash
make prepare-programming        # Step 1
make extract-whatson           # Step 2
make historical-programming    # Step 3 (needs 1 & 2)
make train-audience-ratings    # Step 4 (needs 1)
make extract-IL-training-data  # Step 5 (needs 3 & 4)
```

## Common Options

All commands support:
- `--help`: Show help message
- Custom output paths via arguments

Use `make help` to see all available targets.
