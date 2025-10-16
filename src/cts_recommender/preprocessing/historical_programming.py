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
    Match programming broadcast records to catalog entries using TMDB IDs and title fallback.

    Matching strategy:
    1. Primary: TMDB ID matching
       - Converts TMDB IDs to strings for matching with catalog_id
       - Performs LEFT join to preserve all programming records
    2. Fallback: Title-based matching for ALL unmatched records
       - Uses 'title' field for exact case-insensitive match
       - Handles two scenarios:
         a) Records without TMDB ID (failed enrichment)
         b) Records with TMDB ID but different TMDB match between catalog/programming
       - Prefers TMDB-based catalog IDs over WO_* IDs for duplicate titles
    3. Result: Records with match get catalog_id, others remain None

    Why title fallback is needed:
    - TMDB fuzzy matching can give different results when run at different times
    - Same movie may get enriched with different TMDB IDs in catalog vs programming
    - Example: "Seul sur Mars" → TMDB 286217 in programming, 837912 in catalog

    Parameters:
    programming_df (pd.DataFrame): Enriched programming data with tmdb_id and processed_title
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
    # Use merge to join on catalog index with indicator to track matches
    matched_df = prog_df.merge(
        catalog_df[[]],  # Empty columns, we just need the index (catalog_id)
        left_on='catalog_lookup_id',
        right_index=True,
        how='left',  # Keep ALL programming records
        indicator=True  # Add _merge column to track match status
    )

    # Add catalog_id as a column ONLY for records that matched the catalog
    # For matched records: use the catalog_lookup_id
    # For unmatched records: set to None/NaN (movie was broadcast but not in catalog)
    matched_df['catalog_id'] = None
    is_matched = matched_df['_merge'] == 'both'
    matched_df.loc[is_matched, 'catalog_id'] = matched_df.loc[is_matched, 'catalog_lookup_id']

    tmdb_matched_count = is_matched.sum()
    logger.info(f"TMDB-based matches: {tmdb_matched_count} records ({100*tmdb_matched_count/len(matched_df):.1f}%)")

    # Step 2: FALLBACK - Title matching for ALL unmatched records
    # This handles:
    # 1. Records without TMDB ID (failed enrichment)
    # 2. Records with TMDB ID but different TMDB match between catalog and programming
    unmatched_mask = matched_df['catalog_id'].isna()
    unmatched_count = unmatched_mask.sum()

    if unmatched_count > 0 and 'title' in matched_df.columns and 'title' in catalog_df.columns:
        logger.info(f"Attempting title-based fallback for {unmatched_count} unmatched records...")

        # Create title lookup dictionary from catalog using original title (more reliable than processed_title)
        # For duplicate titles, prefer TMDB IDs over WO_* IDs
        catalog_title_map = {}
        for catalog_id, row in catalog_df.iterrows():
            title = row.get('title')
            if pd.notna(title) and str(title).strip():
                title_lower = str(title).strip().lower()
                # Add to map if:
                # - Title not yet in map, OR
                # - Title exists but current entry is WO_* and new one is TMDB-based
                if title_lower not in catalog_title_map:
                    catalog_title_map[title_lower] = catalog_id
                elif catalog_title_map[title_lower].startswith('WO_') and not str(catalog_id).startswith('WO_'):
                    # Replace WO_* with TMDB-based ID
                    catalog_title_map[title_lower] = catalog_id

        # Match by title
        title_matches = 0
        for idx in matched_df[unmatched_mask].index:
            prog_title = matched_df.loc[idx, 'title']
            if pd.notna(prog_title) and str(prog_title).strip():
                prog_title_lower = str(prog_title).strip().lower()
                if prog_title_lower in catalog_title_map:
                    matched_df.loc[idx, 'catalog_id'] = catalog_title_map[prog_title_lower]
                    title_matches += 1

        if title_matches > 0:
            logger.info(f"Title-based matches: {title_matches} additional records ({100*title_matches/unmatched_count:.1f}% of unmatched)")
        else:
            logger.info(f"Title-based matches: 0 additional records")

    # Drop the temporary columns
    matched_df = matched_df.drop(columns=['catalog_lookup_id', '_merge'], errors='ignore')

    # Count final matches
    has_catalog_id = matched_df['catalog_id'].notna()
    logger.info(f"Total successfully matched: {has_catalog_id.sum()} records ({100*has_catalog_id.sum()/len(matched_df):.1f}%)")
    logger.info(f"Unmatched (not in catalog): {(~has_catalog_id).sum()} records ({100*(~has_catalog_id).sum()/len(matched_df):.1f}%)")
    if has_catalog_id.sum() > 0:
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


