# Testing Guide

This document provides an overview of the testing strategy and test suite for the RTS Curator Recommendation System.

## Overview

The project uses **pytest** for unit testing, with a focus on testing data processing pipelines, feature engineering functions, and data quality validation logic.

### Test Structure

```
tests/
├── unit/
│   ├── test_dates.py                    # Holiday and season detection
│   ├── test_programming.py              # R data loading and schema validation
│   ├── test_ml_feature_processing.py    # ML feature processing functions
│   └── test_whatson_extraction.py       # WhatsOn catalog movie extraction
└── (future: integration/, e2e/)
```

## Running Tests

### Run All Tests
```bash
# Using pytest directly
uv run pytest

# Using Makefile
make test

# With verbose output
uv run pytest -v

# With coverage
uv run pytest --cov=cts_recommender
```

### Run Specific Test Files
```bash
# Test ML feature processing
uv run pytest tests/unit/test_ml_feature_processing.py -v

# Test WhatsOn extraction
uv run pytest tests/unit/test_whatson_extraction.py -v

# Test specific class
uv run pytest tests/unit/test_ml_feature_processing.py::TestProcessMovieTmdbFeatures -v

# Test specific function
uv run pytest tests/unit/test_whatson_extraction.py::TestIsEpisodeTitle::test_episode_pattern_ep_dot_number -v
```

## Test Coverage by Module

### 1. Date Processing (`test_dates.py`)
**Coverage**: 2 tests

Tests for [preprocessing/dates.py](../src/cts_recommender/preprocessing/dates.py):
- `get_season()` - Season calculation from dates
- `build_holiday_index()` and `is_holiday_indexed()` - Holiday detection

**Example**:
```python
def test_get_season():
    assert get_season(date(2025, 6, 1)) == "summer"
```

### 2. Programming Data Loading (`test_programming.py`)
**Coverage**: 10 tests

Tests for [preprocessing/programming.py](../src/cts_recommender/preprocessing/programming.py):
- Schema validation for R data files
- Column renaming and ordering
- Missing column detection
- Schema integrity checks

**Test Classes**:
- `TestLoadOriginalProgramming` - Data loading with schema validation
- `TestSchemaIntegrity` - Ensures schema changes are tracked

**Example**:
```python
def test_sanity_check_fails_with_missing_column():
    """Verify KeyError when expected columns are missing."""
    columns = [c for c in ORIGINAL_PROGRAMMING_COLUMNS if c != "Title"]
    mock_df = pd.DataFrame(columns=columns)

    with pytest.raises(KeyError, match="Missing expected RData columns.*Title"):
        load_original_programming(Path("dummy.RData"))
```

### 3. ML Feature Processing (`test_ml_feature_processing.py`)
**Coverage**: 20 tests

Tests for [preprocessing/ML_feature_processing.py](../src/cts_recommender/preprocessing/ML_feature_processing.py):

#### Test Classes:

**`TestProcessMovieTmdbFeatures`** (6 tests)
- Complete TMDB data handling
- Missing TMDB data with defaults
- Non-movie defaults (TV shows)
- Empty release date handling
- Genre data type conversions
- Multiple row processing

**Example**:
```python
def test_movie_with_missing_tmdb_data():
    """Test that missing TMDB fields get appropriate defaults."""
    df = pd.DataFrame({
        'adult': [np.nan],
        'original_language': [np.nan],
        'genres': [[]],
        'release_date': [np.nan],
        'revenue': [np.nan],
        'tmdb_id': [np.nan],
        'vote_average': [np.nan],
        'popularity': [np.nan],
    })

    result = process_movie_tmdb_features(df, is_movie=True)

    assert result['adult'].iloc[0] == False
    assert result['original_language'].iloc[0] == 'unknown'
    assert result['revenue'].iloc[0] == 0
    assert result['missing_tmdb'].iloc[0] == True
```

**`TestAddGenreFeatures`** (4 tests)
- Single genre one-hot encoding
- Multiple genres per movie
- Multiple movies with different genres
- Empty genre handling

**`TestAddMovieAgeFeature`** (4 tests)
- Recent movie age calculation
- Old movie age calculation
- Multiple movies with different ages
- Default date (1900-01-01) handling

**`TestFinalizeMLFeatures`** (3 tests)
- Boolean to integer conversion
- NaN validation (passes/fails)

**`TestBuildXFeaturesMoviesOnly`** (3 tests)
- Complete pipeline with single movie
- Complete pipeline with multiple movies
- Pipeline with missing TMDB data

### 4. WhatsOn Catalog Extraction (`test_whatson_extraction.py`)
**Coverage**: 47 tests

Tests for [preprocessing/whatson_extraction.py](../src/cts_recommender/preprocessing/whatson_extraction.py):

#### Test Classes:

**`TestIsEpisodeTitle`** (9 tests)
- Episode pattern detection: `ép.1`, `épisode 1`, `S01E01`, `E1`, `1`, `1 - Title`, `partie 1`, `part 1`
- Non-episode titles
- NaN handling

**`TestHasIgnorePattern`** (4 tests)
- Doublon detection
- Serie/série/series pattern detection
- Clean titles that shouldn't match
- NaN handling

**`TestHasDatePattern`** (6 tests)
- YYYY-MM-DD format
- YY-MM-DD format
- Date with dots (YY.MM.DD)
- Date with slashes (YY/MM/DD)
- Non-date titles
- NaN handling

**`TestParseDurationMinutes`** (5 tests)
- HH:MM:SS format parsing
- MM:SS format parsing
- Zero duration (00:00:00)
- Invalid formats
- NaN handling

**`TestIsFilmCollection`** (5 tests)
- Exact collection name matches
- Keyword matching (case-insensitive)
- Non-film collections
- NaN handling

**`TestSelectBestTitle`** (6 tests)
- Priority 1: original_title (when valid)
- Priority 2: title (when original is episode)
- Priority 3: collection (fallback)
- Episode title filtering
- Empty string handling

**`TestShouldKeepAsMovie`** (12 tests)
- Rule 1: Episode patterns → reject
- Rule 2: Date patterns → reject
- Rule 3: Title frequency > 3 → reject
- Rule 4: Collection frequency > 2 (no film keyword) → reject
- Rule 5: Duration < 35 min → reject
- Rule 8: title == original_title → accept
- Rule 10: Film collection → accept
- Valid movie passes all checks
- Missing title → reject
- Zero duration allowed (00:00:00)
- Long duration passes

**Example**:
```python
def test_rule_8_title_equals_original_title(self):
    """Rule 8: When title == original_title, it's a strong positive signal."""
    row = pd.Series({
        'title': 'Inception',
        'original_title': 'Inception',
        'collection': 'Some Collection',
        'duration': '02:30:00',
    })
    assert should_keep_as_movie(row, title_counts, collection_counts) == True
```

## Testing Patterns and Best Practices

### 1. Use Descriptive Test Names
```python
# Good
def test_movie_with_missing_tmdb_data():
    """Test processing movie with missing TMDB fields."""

# Bad
def test_movie():
```

### 2. Test Edge Cases
```python
def test_empty_release_date_handling():
    """Test that empty string release dates are handled correctly."""
    df = pd.DataFrame({'release_date': ['']})
    result = process_movie_tmdb_features(df, is_movie=True)
    assert result['release_date'].iloc[0] == '1900-01-01'
```

### 3. Test Multiple Scenarios per Function
For each function, test:
- ✅ Happy path (valid data)
- ✅ Missing/NaN data
- ✅ Empty data
- ✅ Edge cases (empty strings, zero values)
- ✅ Multiple rows

### 4. Use Test Classes to Group Related Tests
```python
class TestProcessMovieTmdbFeatures:
    """Tests for process_movie_tmdb_features function."""

    def test_movie_with_complete_tmdb_data(self):
        # Test case 1

    def test_movie_with_missing_tmdb_data(self):
        # Test case 2
```

### 5. Use Setup Methods for Test Fixtures
```python
class TestShouldKeepAsMovie:
    def setup_method(self):
        """Set up test fixtures."""
        self.title_counts = pd.Series({'Episode 1': 40, 'Inception': 1})
        self.collection_counts = pd.Series({'Film': 8003})
```

### 6. Mock External Dependencies
```python
from unittest.mock import patch

def test_load_with_mock():
    with patch('cts_recommender.preprocessing.programming.read_rdata',
               return_value=mock_df):
        result = load_original_programming(Path("dummy.RData"))
```

## Key Testing Concepts

### Data Quality Tests
Tests that validate data processing produces clean, consistent output:
- No NaN values after processing (unless expected)
- Correct data types (booleans → int)
- Valid ranges (duration > 0, year > 1900)
- Schema compliance

### Business Logic Tests
Tests that validate domain-specific rules:
- WhatsOn filtering rules (11 rules for identifying movies)
- Title selection priority
- Genre encoding logic
- Movie age calculation

### Schema Validation Tests
Tests that ensure schema changes are tracked:
- Column name mappings are complete
- No orphaned entries in rename maps
- Expected column counts

## Adding New Tests

When adding new functionality, follow this checklist:

1. **Create test file** in `tests/unit/test_<module_name>.py`
2. **Import functions** to test
3. **Create test classes** for grouping related tests
4. **Write test cases**:
   - Happy path
   - Edge cases
   - Error cases
5. **Run tests** to verify they pass
6. **Update this documentation** with new test coverage

### Example: Adding Tests for a New Function

```python
# In src/cts_recommender/preprocessing/new_module.py
def process_text(text: str) -> str:
    """Process text by removing special characters."""
    return text.strip().lower()

# In tests/unit/test_new_module.py
import pytest
from cts_recommender.preprocessing.new_module import process_text

class TestProcessText:
    """Tests for process_text function."""

    def test_normal_text(self):
        """Test processing normal text."""
        assert process_text("  Hello World  ") == "hello world"

    def test_empty_string(self):
        """Test processing empty string."""
        assert process_text("") == ""

    def test_special_characters(self):
        """Test processing text with special characters."""
        assert process_text("  Hello! World?  ") == "hello! world?"
```

## Test Statistics

**Current Test Count**: 79 tests
- `test_dates.py`: 2 tests
- `test_programming.py`: 10 tests
- `test_ml_feature_processing.py`: 20 tests
- `test_whatson_extraction.py`: 47 tests

**Test Execution Time**: ~1.2 seconds (all tests)

## Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example CI configuration
- name: Run tests
  run: uv run pytest --cov=cts_recommender --cov-report=xml
```

## Debugging Failed Tests

### View Full Error Output
```bash
uv run pytest -v --tb=long
```

### Run Single Failing Test
```bash
uv run pytest tests/unit/test_ml_feature_processing.py::TestProcessMovieTmdbFeatures::test_movie_with_missing_tmdb_data -v
```

### Add Print Statements
```python
def test_debug():
    result = some_function()
    print(f"Result columns: {result.columns.tolist()}")
    print(f"Result shape: {result.shape}")
    print(f"NaN counts: {result.isna().sum()}")
    assert ...
```

### Use pytest's `-s` flag to see print output
```bash
uv run pytest tests/unit/test_ml_feature_processing.py -s
```

## Future Testing Plans

### Integration Tests
- Test complete pipeline flows (raw → processed → enriched → ML features)
- Test interaction between modules
- Test file I/O operations

### End-to-End Tests
- Test CLI commands
- Test complete workflows from raw data to final output
- Test with real data samples

### Performance Tests
- Benchmark data processing speed
- Memory usage profiling
- Large dataset handling

### Data Quality Tests
- Validate output data ranges
- Check for data leakage
- Verify feature distributions

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pandas Testing Guide](https://pandas.pydata.org/docs/reference/testing.html)

---

**Last Updated**: 2025-10-07
**Test Coverage**: 79 tests covering data loading, feature processing, and catalog extraction
