# ---- Config ---------------------------------------------------------------
RDATA ?= data/raw/original_R_dataset.RData
WHATSON_DATA ?= data/raw/original_raw_whatson.csv
OUT_PROCESSED ?= data/processed/processed_programming.parquet
OUT_ENRICHED ?= data/processed/programming_enriched.parquet
ML_FEATURES_PATH ?= data/processed/ML_features.parquet
MODEL_OUTPUT ?= data/models/audience_ratings_model.joblib
OUT_WHATSON ?= data/processed/whatson.parquet
APP   ?= cts-reco-prepare-programming
LOG_LEVEL ?= INFO

# ---- Defaults -------------------------------------------------------------
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.PHONY: help setup bootstrap lint fmt fix test preprocess-programming clean env

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
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-prepare-programming --rdata $(RDATA) --out_processed $(OUT_PROCESSED) --out_enriched $(OUT_ENRICHED)
# If no console script, use:
#	uv run python -m cts_recommender.cli.preprocessing_programming --rdata $(RDATA) --out $(OUT)

train-audience-ratings: ## Train audience ratings model
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-train-audience-ratings --ml_features $(ML_FEATURES_PATH) --model_output $(MODEL_OUTPUT)
# If no console script, use:
#	uv run python -m cts_recommender.cli.train_audience_ratings --ml_features $(ML_FEATURES_PATH) --model_output $(MODEL_OUTPUT)

extract-whatson: ## Extract Whatson data into parquet file
	LOG_LEVEL=$(LOG_LEVEL) uv run cts-reco-extract-whatson --file_path $(WHATSON_DATA) --out_file $(OUT_WHATSON)

clean: ## Clean caches
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info

env: ## Show env info
	uv --version
	uv run python -V
	uv run python -c "import sys,site;print('site-packages:', site.getsitepackages()); print('python:', sys.executable)"
