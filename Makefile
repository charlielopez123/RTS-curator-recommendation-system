# ---- Config ---------------------------------------------------------------
RDATA ?= data/raw/original_R_dataset.RData
WHATSON_CSV ?= data/raw/original_raw_whatson.csv

OUT_PROCESSED ?= data/processed/programming/processed_programming.parquet
OUT_ENRICHED ?= data/processed/programming/programming_enriched.parquet
ML_FEATURES_PATH ?= data/processed/programming/ML_features.parquet

MODEL_OUTPUT ?= data/models/audience_ratings_model.joblib

OUT_WHATSON_INIT_EXTRACTION ?= data/processed/whatson/whatson_extracted_movies.parquet
OUT_WHATSON_ENRICHED ?= data/processed/whatson/whatson_catalogue_enriched_tmdb.parquet
OUT_WHATSON_CATALOG ?= data/processed/whatson/whatson_catalog.parquet

OUT_HISTORICAL_PROGRAMMING ?= data/processed/whatson/historical_programming.parquet
OUT_BROADCAST_STATS ?= data/processed/whatson/broadcast_statistics.parquet

LOG_LEVEL ?= INFO

# SSL Configuration - Use certifi's CA bundle for HTTPS requests (macOS fix)
export REQUESTS_CA_BUNDLE := $(shell .venv/bin/python -c "import certifi; print(certifi.where())" 2>/dev/null)

# ---- Defaults -------------------------------------------------------------
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.PHONY: help setup bootstrap lint fmt fix test prepare-programming train-audience-ratings extract-whatson historical-programming clean env

help: ## List available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## / - /'

setup: ## Install / sync deps
	uv sync

bootstrap: ## One-time dev setup
	uv sync
	uv run pre-commit install

lint: ## Lint
	LOG_LEVEL=$(LOG_LEVEL)uv run ruff check .

fmt: ## Format
	LOG_LEVEL=$(LOG_LEVEL) uv run ruff format .

fix: ## Lint with fixes
	LOG_LEVEL=$(LOG_LEVEL) uv run ruff check --fix .

test: ## Run tests
	LOG_LEVEL=$(LOG_LEVEL) uv run pytest -q

prepare-programming: ## Build programming parquet from RData
	mkdir -p $(dir $(OUT_PROCESSED))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-prepare-programming --rdata $(RDATA) --out_processed $(OUT_PROCESSED) --out_enriched $(OUT_ENRICHED) --out_ml $(ML_FEATURES_PATH)
# If no console script, use:
#	uv run python -m cts_recommender.cli.preprocessing_programming --rdata $(RDATA) --out $(OUT)

train-audience-ratings: ## Train audience ratings model
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-audience-ratings --ml_features $(ML_FEATURES_PATH) --model_output $(MODEL_OUTPUT)
# If no console script, use:
#	uv run python -m cts_recommender.cli.train_audience_ratings --ml_features $(ML_FEATURES_PATH) --model_output $(MODEL_OUTPUT)

extract-whatson: ## Process WhatsOn catalog (extract movies, enrich with TMDB, build additional features)
	mkdir -p $(dir $(OUT_WHATSON_CATALOG))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-extract-whatson \
		--csv $(WHATSON_CSV) \
		--out_init_extraction $(OUT_WHATSON_INIT_EXTRACTION) \
		--out_enriched $(OUT_WHATSON_ENRICHED) \
		--out_catalog $(OUT_WHATSON_CATALOG)

historical-programming: ## Build historical programming by matching broadcasts to catalog
	mkdir -p $(dir $(OUT_HISTORICAL_PROGRAMMING))
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-historical-programming \
		--programming $(OUT_ENRICHED) \
		--catalog $(OUT_WHATSON_CATALOG) \
		--out $(OUT_HISTORICAL_PROGRAMMING) \
		--out-stats $(OUT_BROADCAST_STATS)

clean: ## Clean caches
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info

env: ## Show env info
	uv --version
	uv run python -V
	uv run python -c "import sys,site;print('site-packages:', site.getsitepackages()); print('python:', sys.executable)"
