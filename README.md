# RTS Curator Recommendation System

A contextual movie recommendation system designed for RTS (Radio Télévision Suisse) that processes TV programming data and enriches it with movie metadata to enable intelligent content curation and recommendation capabilities.

## 🎯 Project Overview

Data processing pipeline for RTS (Radio Télévision Suisse) that:

1. Processes TV programming data from R datasets
2. Enriches with TMDB movie metadata
3. Builds ML features for audience ratings regression
4. Extracts imitation learning training data for content curation

**Tech Stack**: Python 3.9-3.12, pandas, scikit-learn, pydantic-settings, TMDB API

## 📁 Project Structure

```
src/cts_recommender/
├── adapters/tmdb/              # TMDB API client
├── cli/                        # CLI commands (5 scripts)
├── competition/                # Competitor content scraping
├── environments/               # RL environment for TV programming
├── features/                   # Feature schemas
├── imitation_learning/         # IL training data extraction
├── io/                         # Data readers/writers
├── models/                     # ML models (audience ratings regressor)
├── pipelines/                  # Data processing pipelines (5 pipelines)
├── preprocessing/              # Data preprocessing modules
├── utils/                      # Utilities
├── RTS_constants.py           # RTS-specific constants
└── settings.py                # Configuration

data/
├── raw/                       # Original datasets
├── processed/                 # Processed parquet files
│   ├── programming/          # Programming pipelines output
│   ├── whatson/              # WhatsOn catalog
│   └── IL/                   # Imitation learning training data
├── reference/                # Static data (holidays)
└── models/                   # Trained models (.joblib)
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.9 - 3.12 (3.13+ not yet supported)
- **uv**: Modern Python package manager
- **TMDB API Key**: For movie metadata enrichment

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RTS-curator-recommendation-system
   ```

2. **Install uv** (if not already installed)
   ```bash
   # macOS with Homebrew
   brew install uv

   # Or with pip
   pip install uv
   ```

3. **Set up virtual environment and install dependencies**
   ```bash
   # Create virtual environment with compatible Python version
   uv venv --python python3.12

   # Install all dependencies
   uv sync
   ```

4. **Configure environment variables**
   ```bash
   # Copy and edit the .env file
   cp .env.example .env  # if available, or create new

   # Add your TMDB API credentials to .env:
   APP_TMDB__API_BASE_URL=https://api.themoviedb.org/3
   APP_TMDB__API_KEY=your_tmdb_api_key_here
   ```

5. **Set up data directories**
   ```bash
   mkdir -p data/{raw,processed,reference}

   # Add your holidays reference file
   echo '{}' > data/reference/holidays.json  # or add actual holidays data
   ```

### Running Pipelines

Available commands (see `make help`):

```bash
# 1. Process programming data (RData → preprocessed → enriched → ML features)
make prepare-programming

# 2. Extract WhatsOn catalog from CSV
make extract-whatson WHATSON_CSV=data/raw/original_raw_whatson.csv

# 3. Build historical programming (match broadcasts to catalog)
make historical-programming

# 4. Train audience ratings model
make train-audience-ratings

# 5. Extract imitation learning training data
make extract-IL-training-data
```

Or use CLI directly:
```bash
uv run cts-reco-prepare-programming --help
uv run cts-reco-extract-whatson --help
uv run cts-reco-historical-programming --help
uv run cts-reco-train-audience-ratings --help
uv run cts-reco-extract-IL-training-data --help
```

## 🔧 Configuration

### Environment Variables

The system uses environment-based configuration through `.env` files:

```bash
# Application Settings
APP_ENV=dev                              # Environment: dev, staging, prod
APP_LOG_LEVEL=INFO                       # Logging level

# TMDB API Configuration
APP_TMDB__API_BASE_URL=https://api.themoviedb.org/3
APP_TMDB__API_KEY=your_api_key_here     # Your TMDB API key

# Data Paths (optional overrides)
APP_DATA_ROOT=data                       # Root data directory
APP_RAW_DIR=data/raw                    # Raw data location
APP_PROCESSED_DIR=data/processed        # Processed data location
```

## 📚 Documentation

- **[CLI Reference](docs/CLI.md)**: All CLI commands
- **[Pipeline Architecture](docs/PIPELINE.md)**: Data processing pipelines
- **[WhatsOn Catalog](docs/WHATSON_CATALOG.md)**: Catalog creation and deduplication
- **[Imitation Learning](docs/IMITATION_LEARNING.md)**: IL training data extraction
- **[Competition Scraping](docs/COMPETITION_SCRAPING.md)**: Competitor TV data
- **[TMDB Enrichment](docs/TMDB_ENRICHMENT.md)**: TMDB matching consistency
- **[Testing Guide](docs/TESTING.md)**: Test suite overview
- **[CTS Hyperparameters](docs/CTS_HYPERPARAMETERS.md)**: Model configuration guide
- **[Interactive Testing](docs/INTERACTIVE_TESTING.md)**: Interactive CTS testing tool

## 🧪 Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=cts_recommender

# Run specific test file
uv run pytest tests/unit/test_dates.py
```

### Code Quality

```bash
# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/

# Type checking (if mypy is installed)
uv run mypy src/
```

### Jupyter Development

```bash
# Install notebook dependencies
uv sync --group notebooks

# Start Jupyter Lab
uv run jupyter lab
```

## 🔍 Common Issues

**Python version**: Use Python 3.12
```bash
uv venv --python python3.12
```

**Missing TMDB key**: Add to `.env`
```bash
APP_TMDB__API_KEY=your_key_here
```

**Missing holidays**: Create empty file
```bash
echo '{}' > data/reference/holidays.json
```