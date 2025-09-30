# CLI Reference

## Available Commands

### cts-reco-prepare-programming

Runs the complete data processing pipeline: raw RTS data → preprocessed → TMDB enriched → ML features.

**Usage:**
```bash
cts-reco-prepare-programming \
  --rdata data/raw/original_R_dataset.RData \
  --out_processed data/processed/programming.parquet \
  --out_enriched data/processed/programming_enriched.parquet
```

**Arguments:**
- `--rdata` (optional): Path to input .RData file
- `--out_processed` (required): Output path for preprocessed data
- `--out_enriched` (required): Output path for TMDB-enriched data

**What it does:**
1. Loads and preprocesses raw RTS programming data
2. Enriches movies with TMDB metadata
3. Creates ML-ready feature matrix (saved as `data/processed/ML_features.parquet`)

**Example:**
```bash
cts-reco-prepare-programming \
  --out_processed data/processed/programming.parquet \
  --out_enriched data/processed/programming_enriched.parquet
```

**Requirements:**
- TMDB API key in environment: `APP_TMDB__API_KEY=your_key`
- Holidays file: `data/reference/holidays.json` (can be empty: `echo '[]' > data/reference/holidays.json`)