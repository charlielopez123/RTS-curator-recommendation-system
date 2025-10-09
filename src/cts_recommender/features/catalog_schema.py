"""
Schema definitions for WhatsOn catalog and historical programming datasets.

Defines expected data types for all columns to ensure consistency across
the pipeline and prevent dtype drift in parquet files.
"""

# WhatsOn Catalog Schema
# Catalog is indexed by catalog_id (string)
CATALOG_DTYPES = {
    # Core identifiers
    'tmdb_id': 'Int64',  # Nullable integer (NaN for movies without TMDB)
    'missing_tmdb_id': 'boolean',
    'best_title': 'string',
    'title': 'string',
    'original_title': 'string',
    'processed_title': 'string',

    # Movie metadata
    'collection': 'string',
    'duration': 'string',  # HH:MM:SS format from CSV
    'short_description': 'string',
    'class_key': 'string',
    'production_regions': 'string',
    'production_year': 'float64',  # Can be NaN
    'parental_control': 'string',
    'department': 'string',
    'director': 'string',
    'actors': 'string',

    # Broadcasting rights and dates
    'tv_rights_start': 'datetime64[ns]',
    'tv_rights_end': 'datetime64[ns]',
    'first_broadcast_date': 'datetime64[ns]',
    'last_broadcast_date': 'datetime64[ns]',
    'rebroadcast_date_1': 'datetime64[ns]',
    'rebroadcast_date_2': 'datetime64[ns]',
    'rebroadcast_date_3': 'datetime64[ns]',
    'rebroadcast_date_4': 'datetime64[ns]',

    # Broadcast counts
    'total_broadcasts': 'Int64',
    'consumed_broadcasts': 'Int64',
    'available_broadcasts': 'Int64',
    'rts1_rts2_broadcasts': 'Int64',
    'tv_rights_count': 'Int64',
    'valid_tv_rights_count': 'Int64',
    'times_shown': 'Int64',  # Computed from historical programming

    # Ratings
    'last_broadcast_rating': 'float64',
    'last_broadcast_rating_plus_7': 'float64',

    # TMDB features
    'adult': 'boolean',
    'original_language': 'string',
    'popularity': 'float64',
    'revenue': 'float64',
    'vote_average': 'float64',
    'release_date': 'string',  # YYYY-MM-DD format from TMDB
    'missing_release_date': 'boolean',

    # Computed features
    'movie_age': 'Int64',  # Years since release

    # Other
    'external_reference': 'string',

    # Note: 'genres' is a list type, handled separately
}

# Historical Programming Schema
HISTORICAL_PROGRAMMING_DTYPES = {
    # Core identifiers
    'catalog_id': 'string',  # Links to catalog (nullable - not all broadcasts matched)
    'tmdb_id': 'Int64',  # Nullable integer

    # Programming metadata
    'date': 'datetime64[ns]',
    'start_time': 'string',  # HH:MM:SS format
    'channel': 'string',
    'title': 'string',
    'processed_title': 'string',

    # Metrics
    'duration_min': 'float64',
    'rt_m': 'float64',  # Audience rating (thousands)
    'pdm': 'float64',  # Market share (%)

    # Temporal features
    'hour': 'Int64',
    'weekday': 'Int64',
    'is_weekend': 'boolean',
    'season': 'string',
    'public_holiday': 'boolean',

    # TMDB features (from enrichment)
    'adult': 'boolean',
    'original_language': 'string',
    'popularity': 'float64',
    'revenue': 'float64',
    'vote_average': 'float64',
    'release_date': 'string',
    'missing_tmdb_id': 'boolean',

    # Note: 'genres' is a list type, handled separately
}

# Broadcast Statistics Schema
BROADCAST_STATISTICS_DTYPES = {
    # Indexed by catalog_id (string)
    'times_shown': 'Int64',
    'first_broadcast': 'datetime64[ns]',
    'last_broadcast': 'datetime64[ns]',
    'avg_rating': 'float64',
    'total_audience': 'float64',
    # Note: 'channels' is a list type, handled separately
}


def enforce_dtypes(df, dtype_dict: dict, skip_missing: bool = True):
    """
    Enforce data types on a DataFrame according to a dtype dictionary.

    Parameters:
    df (pd.DataFrame): DataFrame to enforce types on
    dtype_dict (dict): Dictionary mapping column names to dtype strings
    skip_missing (bool): If True, skip columns not in DataFrame (default: True)

    Returns:
    pd.DataFrame: DataFrame with enforced types
    """
    import pandas as pd

    for col, dtype in dtype_dict.items():
        if col not in df.columns:
            if skip_missing:
                continue
            else:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Skip list columns (genres, channels)
        if isinstance(df[col].iloc[0] if len(df) > 0 else None, list):
            continue

        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            # Log warning but don't fail - some conversions may need special handling
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not convert column '{col}' to {dtype}: {e}")

    return df
