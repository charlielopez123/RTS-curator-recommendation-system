"""
Historical Programming Extraction

This module builds a historical programming dataset by matching TV broadcast records
from the enriched programming data with the WhatsOn movie catalog using TMDB IDs.

The resulting dataset provides a time-series view of which movies from the catalog
were broadcast when, on which channels, and with what audience metrics.
"""

from pathlib import Path
import pandas as pd
import logging


from cts_recommender.RTS_constants import HISTORICAL_PROGRAMMING_COLUMNS
from cts_recommender.io import readers

logger = logging.getLogger(__name__)


def load_enriched_programming(data_path: Path) -> pd.DataFrame:
    """
    Load the TMDB-enriched programming data.

    Parameters:
    data_path (Path): Path to programming_enriched.parquet

    Returns:
    pd.DataFrame: Enriched programming data with tmdb_id column
    """
    df = readers.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} programming records")
    return df


def load_whatson_catalog(catalog_path: Path) -> pd.DataFrame:
    """
    Load the WhatsOn catalog indexed by catalog_id.

    Parameters:
    catalog_path (Path): Path to whatson_catalog.parquet

    Returns:
    pd.DataFrame: Catalog indexed by catalog_id
    """
    df = readers.read_parquet(catalog_path)
    logger.info(f"Loaded {len(df)} catalog entries")
    return df


def match_programming_to_catalog(
    programming_df: pd.DataFrame,
    catalog_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match programming broadcast records to catalog entries using TMDB IDs.

    This function:
    1. Converts TMDB IDs to strings for matching with catalog_id
    2. Performs LEFT join to preserve all programming records
    3. Records with TMDB match get catalog_id populated
    4. Records without TMDB or no catalog match have catalog_id = None

    Parameters:
    programming_df (pd.DataFrame): Enriched programming data with tmdb_id
    catalog_df (pd.DataFrame): WhatsOn catalog indexed by catalog_id

    Returns:
    pd.DataFrame: ALL programming records with catalog_id (when available)
    """
    logger.info("Matching programming records to catalog...")

    prog_df = programming_df.copy()

    # Create catalog lookup ID from TMDB (only for records with valid TMDB)
    has_tmdb = prog_df['tmdb_id'].notna()
    prog_df.loc[has_tmdb, 'catalog_lookup_id'] = prog_df.loc[has_tmdb, 'tmdb_id'].astype('Int64').astype(str)

    logger.info(f"Total programming records: {len(prog_df)}")
    logger.info(f"Programming records with TMDB: {has_tmdb.sum()}")
    logger.info(f"Programming records without TMDB: {(~has_tmdb).sum()}")

    if has_tmdb.sum() > 0:
        unique_tmdb_in_prog = prog_df.loc[has_tmdb, 'catalog_lookup_id'].nunique()
        logger.info(f"Unique TMDB IDs in programming: {unique_tmdb_in_prog}")

    # Check catalog structure
    catalog_tmdb_ids = catalog_df.index[~catalog_df.index.str.startswith('WO_')]
    logger.info(f"Catalog entries with TMDB IDs: {len(catalog_tmdb_ids)}")
    logger.info(f"Catalog entries with custom IDs (WO_*): {catalog_df.index.str.startswith('WO_').sum()}")

    # Perform LEFT join to preserve all programming records
    # Use merge to join on catalog index
    matched_df = prog_df.merge(
        catalog_df[[]],  # Empty columns, we just need the index (catalog_id)
        left_on='catalog_lookup_id',
        right_index=True,
        how='left'  # Keep ALL programming records
    )

    # Add catalog_id as a column
    # For matched records: use the catalog_lookup_id
    # For unmatched records: remains None/NaN
    matched_df['catalog_id'] = matched_df['catalog_lookup_id']

    # Drop the temporary lookup column
    matched_df = matched_df.drop(columns=['catalog_lookup_id'], errors='ignore')

    # Count matches
    has_catalog_id = matched_df['catalog_id'].notna()
    logger.info(f"Successfully matched to catalog: {has_catalog_id.sum()} records")
    logger.info(f"Not in catalog (no catalog_id): {(~has_catalog_id).sum()} records")
    logger.info(f"Unique movies in matched programming: {matched_df.loc[has_catalog_id, 'catalog_id'].nunique()}")

    return matched_df


def build_historical_programming(
    programming_df: pd.DataFrame,
    catalog_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build historical programming dataset with catalog IDs and broadcast metadata.

    The output contains:
    - catalog_id: Link to WhatsOn catalog
    - date, start_time: When the movie was broadcast
    - channel: Which channel broadcast it
    - duration_min: Actual broadcast duration
    - Audience metrics: rt_m, pdm
    - Temporal features: hour, weekday, is_weekend, season, public_holiday
    - TMDB features: genres, release_date, vote_average, etc.

    Parameters:
    programming_df (pd.DataFrame): Enriched programming data
    catalog_df (pd.DataFrame): WhatsOn catalog

    Returns:
    pd.DataFrame: Historical programming with catalog_id
    """
    logger.info("Building historical programming dataset...")

    # Match programming to catalog
    historical_df = match_programming_to_catalog(programming_df, catalog_df)

    # Select and order columns for historical programming
    # Keep core programming columns + catalog_id
    
    # Keep only available columns
    available_cols = [col for col in HISTORICAL_PROGRAMMING_COLUMNS if col in historical_df.columns]
    historical_df = historical_df[available_cols]

    # Sort by date and time
    historical_df = historical_df.sort_values(['date', 'start_time'])

    # Reset index
    historical_df = historical_df.reset_index(drop=True)

    logger.info(f"Historical programming built: {len(historical_df)} broadcasts")
    logger.info(f"Date range: {historical_df['date'].min()} to {historical_df['date'].max()}")
    logger.info(f"Channels: {historical_df['channel'].unique().tolist()}")
    logger.info(f"Unique movies: {historical_df['catalog_id'].nunique()}")

    return historical_df


def compute_broadcast_statistics(historical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute broadcast statistics for each movie in the catalog.

    Returns a summary DataFrame with:
    - catalog_id (index)
    - times_shown: Number of broadcasts
    - first_broadcast: Date of first broadcast
    - last_broadcast: Date of last broadcast
    - channels: List of channels that broadcast the movie
    - avg_rating: Average rt_m rating across broadcasts

    Parameters:
    historical_df (pd.DataFrame): Historical programming data

    Returns:
    pd.DataFrame: Broadcast statistics by catalog_id
    """
    logger.info("Computing broadcast statistics...")

    stats = historical_df.groupby('catalog_id').agg(
        times_shown=('date', 'count'),
        first_broadcast=('date', 'min'),
        last_broadcast=('date', 'max'),
        channels=('channel', lambda x: sorted(x.unique().tolist())),
        avg_rating=('rt_m', 'mean'),
        total_audience=('rt_m', 'sum')
    )

    logger.info(f"Statistics computed for {len(stats)} movies")
    logger.info(f"Average broadcasts per movie: {stats['times_shown'].mean():.1f}")
    logger.info(f"Most broadcast movie: {stats['times_shown'].max()} times")

    return stats
