# Schema definitions for ML features

# Original RData programming table column order (as delivered)
ORIGINAL_PROGRAMMING_COLUMNS: tuple[str, ...] = (
    "Date",
    "Title",
    "Description",
    "Channel",
    "Start time",
    "End time",
    "Duration",
    "Net Duration",
    "BrdCstClassKey",
    "Cibles",
    "Activité",
    "Rt-M",
    "PDM-%",
)

# Mapping from original RData column names to desired Pythonic column names
PROGRAMMING_RENAME_MAP: dict[str, str] = {
    "Date": "date",
    "Title": "title",
    "Description": "description",
    "Channel": "channel",
    "Start time": "start_time",
    "End time": "end_time",
    "Duration": "duration",
    "Net Duration": "net_duration",
    "BrdCstClassKey": "class_key",
    "Cibles": "target_audience",
    "Activité": "activity",
    "Rt-M": "rt_m",
    "PDM-%": "pdm",
}

# Feature categories for ML model training (audience ratings regression)
# Order is important for sklearn models - features must always be in the same order

# Base programming features (always first)
BASE_PROGRAMMING_FEATURES: tuple[str, ...] = (
    "rt_m", "hour", "weekday", "is_weekend", "duration_min",
    "season", "public_holiday"
)

# Categorical features that will be encoded
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "channel", "original_language"
)

# Numerical TMDB features
TMDB_NUMERICAL_FEATURES: tuple[str, ...] = (
    "popularity", "revenue", "vote_average"
)

# Boolean derived features
BOOLEAN_FEATURES: tuple[str, ...] = (
    "adult", "missing_release_date", "missing_tmdb", "is_movie"
)

# Computed numerical features
COMPUTED_FEATURES: tuple[str, ...] = (
    "movie_age",
)

# Final ordered feature set for ML models (excluding dynamic genre columns)
# Note: Genre columns (genre_*) are added dynamically during feature processing
ML_FEATURES: tuple[str, ...] = (
    BASE_PROGRAMMING_FEATURES +
    CATEGORICAL_FEATURES +
    TMDB_NUMERICAL_FEATURES +
    BOOLEAN_FEATURES +
    COMPUTED_FEATURES
)

TARGET_FEATURE: str = "rt_m"

