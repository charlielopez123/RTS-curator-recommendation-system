# Development Guide

## Setup

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Environment variables:**
   ```bash
   # Required
   APP_TMDB__API_KEY=your_tmdb_key

   # Optional
   APP_LOG_LEVEL=DEBUG
   ```

3. **Reference data:**
   ```bash
   echo '[]' > data/reference/holidays.json
   ```

## Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=cts_recommender

# Specific test
uv run pytest tests/unit/test_dates.py
```

## Code Quality

```bash
# Linting
uv run ruff check src/

# Formatting
uv run ruff format src/
```

## Project Structure

```
src/cts_recommender/
├── adapters/          # External APIs (TMDB)
├── cli/               # Command-line tools
├── features/          # Data schemas and extraction specs
├── io/                # File I/O utilities
├── pipelines/         # Data processing workflows
├── preprocessing/     # Data transformation modules
├── utils/             # Utility functions
├── RTS_constants.py   # RTS-specific constants
└── settings.py        # Configuration management
```

## Adding New Features

1. **New preprocessing step:** Add to `preprocessing/` module
2. **New pipeline stage:** Extend `pipelines/programming_pipeline.py`
3. **New CLI command:** Add to `cli/` with entry point in `pyproject.toml`
4. **New feature schema:** Update `features/schemas.py`