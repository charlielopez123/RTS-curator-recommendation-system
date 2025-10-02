# RTS Curator Recommendation System

A contextual movie recommendation system designed for RTS (Radio Télévision Suisse) that processes TV programming data and enriches it with movie metadata to enable intelligent content curation and recommendation capabilities.

## 🎯 Project Overview

The RTS Curator Recommendation System is a data processing pipeline that:

- **Processes TV Programming Data**: Cleans and preprocesses RTS programming schedules from R data files
- **Enriches with Movie Metadata**: Integrates with The Movie Database (TMDB) API to add comprehensive movie information
- **Feature Engineering**: Creates temporal, categorical, and contextual features for recommendation algorithms
- **Swiss Broadcasting Focus**: Specifically tailored for RTS programming patterns and Swiss audience preferences

### Key Capabilities

- ✅ **RTS-Specific Logic**: Handles special movie categorizations and competitor content codes
- ✅ **TMDB Integration**: Robust API client with retry logic, rate limiting, and error handling
- ✅ **Feature Engineering**: Temporal features (holidays, weekdays, seasons), content features (genres, ratings), and viewing metrics
- ✅ **Production Ready**: Comprehensive logging, configuration management, and testing framework
- ✅ **CLI Interface**: Command-line tools for data processing pipelines

## 📁 Project Structure

```
RTS-curator-recommendation-system/
├── src/cts_recommender/           # Main package source code
│   ├── adapters/                  # External API integrations
│   │   └── tmdb/                  # TMDB API client and utilities
│   ├── cli/                       # Command-line interfaces
│   │   └── prepare_programming.py # Data processing CLI tool
│   ├── features/                  # Feature extraction specifications
│   │   ├── schemas.py            # Data schemas and column mappings
│   │   └── tmdb_extract.py       # TMDB feature extraction specs
│   ├── io/                        # Input/output utilities
│   │   ├── readers.py            # Data reading utilities (R, JSON, Parquet)
│   │   └── writers.py            # Data writing utilities
│   ├── pipelines/                 # Data processing pipelines
│   │   └── programming_pipeline.py # Main data processing workflows
│   ├── preprocessing/             # Data preprocessing modules
│   │   ├── dates.py              # Date and holiday processing
│   │   ├── movie_features.py     # TMDB metadata enrichment
│   │   ├── programming.py        # Programming data preprocessing
│   │   └── X_ML_feature_processing.py # ML feature preparation
│   ├── utils/                     # Utility functions
│   │   ├── health.py             # Health check utilities
│   │   └── text_cleaning.py      # Text processing utilities
│   ├── RTS_constants.py          # RTS-specific constants and mappings
│   ├── settings.py               # Configuration management
│   └── __init__.py               # Package initialization
├── data/                          # Data directories
│   ├── raw/                      # Original R data files
│   ├── interim/                  # Intermediate processing results
│   ├── processed/                # Final processed datasets
│   └── reference/                # Reference data (holidays, etc.)
├── experiments/                   # Jupyter notebooks and experiments
│   └── notebooks/                # Analysis and development notebooks
├── tests/                         # Test suite
│   └── unit/                     # Unit tests
├── pyproject.toml                # Project configuration and dependencies
├── .env                          # Environment variables (TMDB API keys)
└── README.md                     # This file
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

### Running the Data Pipeline

1. **Prepare your data**
   - Place your original R dataset in `data/raw/original_R_dataset.RData`

2. **Run the complete processing pipeline**
   ```bash
   # Activate environment (optional with uv)
   source .venv/bin/activate

   # Process data through both stages
   cts-reco-prepare-programming \
     --rdata data/raw/original_R_dataset.RData \
     --out_processed data/processed/programming.parquet \
     --out_enriched data/processed/programming_enriched.parquet
   ```

   This will:
   - Load and preprocess the original R programming data
   - Filter for relevant content (Overnight+7, Personnes 3+)
   - Extract movies using RTS classification keys
   - Enrich movie data with TMDB metadata
   - Generate feature-ready datasets

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

### RTS-Specific Configuration

The system includes Swiss broadcasting-specific constants in `RTS_constants.py`:

- **Movie Classification Keys**: Codes used to identify movies in programming data
- **Competitor Codes**: Classification codes for competitor content
- **Special Movie Names**: Titles requiring description-based extraction

## 📚 Documentation

- **[Pipeline Architecture](docs/PIPELINE.md)**: Detailed explanation of the 3-stage data processing pipeline
- **[CLI Reference](docs/CLI.md)**: Command-line interface usage and examples
- **[API Reference](docs/API.md)**: Function signatures and module documentation
- **[Development Guide](docs/DEVELOPMENT.md)**: Setup, testing, and contribution guidelines

## 📊 Usage Examples

### Basic Data Processing

```python
from cts_recommender.pipelines import programming_pipeline
from cts_recommender.settings import get_settings

cfg = get_settings()

# Run preprocessing pipeline
processed_df, processed_file = programming_pipeline.run_original_Rdata_programming_pipeline(
    raw_file=cfg.raw_dir / "original_R_dataset.RData",
    out_file=cfg.processed_dir / "programming.parquet"
)

# Run enrichment pipeline
enriched_df, enriched_file = programming_pipeline.run_enrich_programming_with_movie_metadata_pipeline(
    processed_file=processed_file,
    out_file=cfg.processed_dir / "programming_enriched.parquet"
)
```

### Feature Engineering

```python
from cts_recommender.preprocessing import X_ML_feature_processing

# Load processed datasets
processed_df, enriched_df = X_ML_feature_processing.load_processed_and_enriched_programming(
    data_path_processed=cfg.processed_dir / "programming.parquet",
    data_path_enriched=cfg.processed_dir / "programming_enriched.parquet"
)

# Create ML-ready feature matrix
feature_df = X_ML_feature_processing.build_X_features(processed_df, enriched_df)
```

### TMDB API Usage

```python
from cts_recommender.adapters.tmdb import tmdb

# Initialize API client
tmdb_api = tmdb.TMDB_API()

# Search for movie
movie_id = tmdb_api.find_best_match("Inception", runtime_minutes=148)

# Get movie features
if movie_id:
    features = tmdb_api.get_movie_features(movie_id)
    title = tmdb_api.get_movie_title(movie_id)
```

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

## 🔍 Troubleshooting

### Common Issues

**1. Python Version Compatibility**
```bash
# Error: requires-python = ">=3.9,<3.13"
# Solution: Use Python 3.12
uv venv --python python3.12
```

**2. Missing TMDB Credentials**
```bash
# Error: TMDB API key not found
# Solution: Check your .env file has correct format
APP_TMDB__API_KEY=your_actual_api_key_here
```

**3. Missing Reference Data**
```bash
# Error: file not found: data/reference/holidays.json
# Solution: Create the file or add actual holidays data
echo '[]' > data/reference/holidays.json
```

**4. Data Pipeline Failures**
- **TMDB Rate Limits**: The API client includes retry logic, but very large datasets may hit rate limits
- **Missing Movie Matches**: Some titles may not match TMDB records - check logs for details
- **Memory Issues**: Large datasets may require processing in chunks

### Performance Optimization

- **Batch Processing**: For very large datasets, consider processing movies in batches
- **Caching**: TMDB responses could be cached for repeated pipeline runs
- **Parallel Processing**: Movie enrichment could be parallelized for better performance

## 📈 Next Steps

### Planned Enhancements

- **Recommendation Engine**: Implement actual recommendation algorithms using the feature matrix
- **Real-time Processing**: Add streaming data processing capabilities
- **Web Interface**: Build a web dashboard for content curators
- **Advanced Features**: Add text analysis of descriptions, image processing for posters
- **Swiss Context**: Integrate Swiss-specific viewing patterns and cultural preferences

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`uv run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **RTS (Radio Télévision Suisse)** for the broadcasting domain expertise
- **The Movie Database (TMDB)** for providing comprehensive movie metadata
- **Swiss Broadcasting Community** for context and requirements

---

For more detailed information about specific components, see the inline documentation in the source code or contact the development team.