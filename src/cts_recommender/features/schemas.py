from dataclasses import dataclass
from typing import Mapping, Sequence

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
    "rt_m", "pdm", "hour", "weekday", "is_weekend", "duration_min",
    "season", "public_holiday"
)

# Categorical features that will be encoded
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "channel", "class_key", "original_language"
)

# Numerical TMDB features
TMDB_NUMERICAL_FEATURES: tuple[str, ...] = (
    "popularity", "revenue", "vote_average"
)

# Boolean derived features
BOOLEAN_FEATURES: tuple[str, ...] = (
    "adult", "missing_release_date", "missing_tmdb", "is_movie"
)

# Final ordered feature set for ML models
ML_FEATURES: tuple[str, ...] = (
    BASE_PROGRAMMING_FEATURES +
    CATEGORICAL_FEATURES +
    TMDB_NUMERICAL_FEATURES +
    BOOLEAN_FEATURES
)

