# ---- Config ---------------------------------------------------------------
RDATA ?= data/raw/original_R_dataset.RData
OUT   ?= data/processed/programming.parquet
APP   ?= cts-reco-prepare-programming

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
	uv run ruff check .

fmt: ## Format
	uv run ruff format .

fix: ## Lint with fixes
	uv run ruff check --fix .

test: ## Run tests
	uv run pytest -q

prepare-programming: ## Build programming parquet from RData
	mkdir -p $(dir $(OUT))
	uv run $(APP) --rdata $(RDATA) --out $(OUT)
# If no console script, use:
#	uv run python -m cts_recommender.cli.preprocessing_programming --rdata $(RDATA) --out $(OUT)

clean: ## Clean caches
	rm -rf .pytest_cache .ruff_cache dist build *.egg-info

env: ## Show env info
	uv --version
	uv run python -V
	uv run python -c "import sys,site;print('site-packages:', site.getsitepackages()); print('python:', sys.executable)"
