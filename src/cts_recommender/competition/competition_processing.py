"""
Competition Schedule Processing

Converts scraped competitor XML data into historical programming format,
matching titles to the WhatsOn catalog using TMDB.
"""

from pathlib import Path
from typing import Dict, List, Callable
import pandas as pd
import logging
from datetime import datetime

from cts_recommender.adapters.tmdb.tmdb import TMDB_API
from cts_recommender.utils import text_cleaning
from cts_recommender.features.catalog_schema import HISTORICAL_PROGRAMMING_DTYPES, enforce_dtypes
from cts_recommender.preprocessing import dates
from cts_recommender.io.readers import read_json
from cts_recommender.settings import get_settings

logger = logging.getLogger(__name__)

# Load holiday data and build index for fast lookup
cfg = get_settings()
holidays = read_json(cfg.reference_dir / "holidays.json")
holiday_index = dates.build_holiday_index(holidays)


def parse_competitor_date_time(date_str: str, time_str: str, channel: str) -> tuple[pd.Timestamp, str, int]:
    """
    Parse competitor date/time strings into standardized format.

    Handles different formats:
    - TF1: date='09/08/2025', time='21.10'
    - M6:  date='2025-08-09', time='21:10'

    Parameters:
    date_str (str): Date string from XML
    time_str (str): Time string from XML
    channel (str): Channel name (TF1, M6, etc.)

    Returns:
    tuple: (pd.Timestamp, start_time_str, hour_int)
    """
    # Parse date based on channel format
    if channel == 'TF1':
        # Format: DD/MM/YYYY
        date_obj = pd.to_datetime(date_str, format='%d/%m/%Y')
        # Format: HH.MM
        hour = int(time_str.split('.')[0])
        minute = int(time_str.split('.')[1])
    else:  # M6, France2, France3
        # Format: YYYY-MM-DD
        date_obj = pd.to_datetime(date_str, format='%Y-%m-%d')
        # Format: HH:MM
        hour = int(time_str.split(':')[0])
        minute = int(time_str.split(':')[1])

    start_time_str = f"{hour:02d}:{minute:02d}:00"

    return date_obj, start_time_str, hour


def match_title_to_catalog(
    title: str,
    catalog_df: pd.DataFrame,
    tmdb_api: TMDB_API,
    use_tmdb_fallback: bool = True
) -> tuple[str | None, int | None]:
    """
    Match competitor movie title to catalog_id.

    Optimized Strategy:
    1. TMDB ID lookup (fastest): Search TMDB first, then match to catalog by tmdb_id
    2. Exact match: Try exact match on catalog.processed_title (for older/TV movies)
    3. Fuzzy match: Try fuzzy match on catalog titles (for older/TV movies)

    This prioritizes direct TMDB ID matching which is more reliable and efficient
    for newer movies, then falls back to text matching for older content or TV movies
    that may not have TMDB metadata yet.

    Parameters:
    title (str): Raw title from competitor XML
    catalog_df (pd.DataFrame): WhatsOn catalog indexed by catalog_id
    tmdb_api (TMDB_API): TMDB API client for fallback search
    use_tmdb_fallback (bool): Whether to use TMDB search if no catalog match

    Returns:
    tuple: (catalog_id, tmdb_id) or (None, None) if no match
    """
    if pd.isna(title) or not title.strip():
        return None, None

    normalized_title = text_cleaning.normalize(title.strip())

    # Strategy 1: TMDB ID lookup (fastest and most reliable)
    if use_tmdb_fallback:
        try:
            logger.debug(f"Trying TMDB lookup for '{title}'...")
            tmdb_id = tmdb_api.find_best_match(title, known_runtime=0)

            if tmdb_id:
                # Check if this TMDB ID exists in catalog
                tmdb_id_str = str(tmdb_id)
                if tmdb_id_str in catalog_df.index:
                    logger.debug(f"TMDB ID match: '{title}' → catalog_id={tmdb_id_str}")
                    return tmdb_id_str, tmdb_id
                else:
                    logger.debug(f"TMDB found {tmdb_id} but not in catalog: '{title}', trying text matching...")
        except Exception as e:
            logger.warning(f"TMDB search failed for '{title}': {e}, falling back to text matching...")

    # Strategy 2: Exact match on processed_title (for older/TV movies without TMDB)
    if 'processed_title' in catalog_df.columns:
        for catalog_id, row in catalog_df.iterrows():
            if pd.notna(row.get('processed_title')):
                catalog_normalized = text_cleaning.normalize(row['processed_title'])
                if normalized_title == catalog_normalized:
                    logger.debug(f"Exact text match: '{title}' → catalog_id={catalog_id}")
                    return catalog_id, row.get('tmdb_id')

    # Strategy 3: Fuzzy match on title or best_title (for older/TV movies)
    title_cols = ['title', 'best_title', 'original_title']
    for catalog_id, row in catalog_df.iterrows():
        for col in title_cols:
            if col in catalog_df.columns and pd.notna(row.get(col)):
                catalog_normalized = text_cleaning.normalize(row[col])
                # Bidirectional substring match
                if (normalized_title in catalog_normalized or
                    catalog_normalized in normalized_title):
                    logger.debug(f"Fuzzy text match: '{title}' → catalog_id={catalog_id} (via {col})")
                    return catalog_id, row.get('tmdb_id')

    logger.debug(f"No match found for: '{title}'")
    return None, None


def flatten_schedule(
    scraped_schedules: Dict[str, Dict[str, List[Dict]]],
    catalog_df: pd.DataFrame,
    tmdb_api: TMDB_API | None = None,
    use_tmdb_fallback: bool = True
) -> pd.DataFrame:
    """
    Convert scraped competitor schedules to historical programming format.

    Input structure:
    {
        'TF1': {
            '41': [{'date': '09/08/2025', 'time': '21.10', 'title': 'Thor'}, ...],
            '42': [...]
        },
        'M6': {
            '41': [{'date': '2025-08-09', 'time': '21:10', 'title': 'Inception'}, ...],
            '42': [...]
        }
    }

    Output: DataFrame with historical programming schema

    Parameters:
    scraped_schedules (dict): Nested dict from CompetitorDataScraper.scraped_schedules
    catalog_df (pd.DataFrame): WhatsOn catalog for title matching
    tmdb_api (TMDB_API): TMDB API client for fallback matching (optional)
    use_tmdb_fallback (bool): Whether to use TMDB for unmatched titles

    Returns:
    pd.DataFrame: Competition data in historical programming format
    """
    logger.info("Flattening competitor schedules...")

    # Initialize TMDB API if not provided
    if tmdb_api is None and use_tmdb_fallback:
        logger.info("Initializing TMDB API for title matching...")
        tmdb_api = TMDB_API()

    rows = []
    total_entries = 0
    matched_entries = 0

    for channel, weeks in scraped_schedules.items():
        logger.info(f"Processing channel: {channel}")

        for week, movies in weeks.items():
            for movie in movies:
                total_entries += 1

                # Parse date and time
                try:
                    date, start_time, hour = parse_competitor_date_time(
                        movie['date'],
                        movie['time'],
                        channel
                    )
                except Exception as e:
                    logger.warning(f"Failed to parse date/time for {movie}: {e}")
                    continue

                # Match title to catalog
                catalog_id, tmdb_id = match_title_to_catalog(
                    title=movie['title'],
                    catalog_df=catalog_df,
                    tmdb_api=tmdb_api,
                    use_tmdb_fallback=use_tmdb_fallback
                )

                if catalog_id:
                    matched_entries += 1

                # Build row matching historical programming schema
                # Initialize with None for all columns, then populate what we have
                row = {col: None for col in HISTORICAL_PROGRAMMING_DTYPES.keys()}

                # Populate fields available from competitor data
                row.update({
                    'catalog_id': catalog_id,
                    'tmdb_id': tmdb_id,
                    'missing_tmdb_id': tmdb_id is None,
                    'date': date,
                    'start_time': start_time,
                    'channel': f"{channel}_T_PL",
                    'title': movie['title'],
                    'processed_title': text_cleaning.normalize(movie['title']),
                    'hour': hour,
                    'weekday': date.weekday(),
                    'is_weekend': date.weekday() >= 5,
                    'season': dates.get_season(date),
                    'public_holiday': dates.is_holiday_indexed(date, holiday_index),
                })

                rows.append(row)

    logger.info(f"Processed {total_entries} competitor movie entries")
    logger.info(f"Matched to catalog: {matched_entries} ({100*matched_entries/total_entries:.1f}%)")
    logger.info(f"Unmatched: {total_entries - matched_entries}")

    if not rows:
        logger.warning("No competitor data to flatten!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Enforce dtypes from schema
    df = enforce_dtypes(df, HISTORICAL_PROGRAMMING_DTYPES, skip_missing=True)

    # Sort by date and time
    df = df.sort_values(['date', 'start_time']).reset_index(drop=True)

    logger.info(f"Created competitor DataFrame: {len(df)} rows, {len(df.columns)} columns")

    return df


def merge_with_historical(
    competition_df: pd.DataFrame,
    historical_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge competition data with existing historical programming data.

    Deduplicates on (catalog_id, date, channel, start_time, title) to avoid duplicates.

    Note: 'title' is included in deduplication to handle cases where catalog_id=None
    (unmatched movies). Without title, two different unmatched movies airing at the
    same time on the same channel would be incorrectly treated as duplicates.

    Parameters:
    competition_df (pd.DataFrame): New competition data
    historical_df (pd.DataFrame): Existing historical programming data

    Returns:
    pd.DataFrame: Merged and deduplicated historical programming
    """
    logger.info("Merging competition data with historical programming...")

    # Concatenate
    merged_df = pd.concat([historical_df, competition_df], ignore_index=True)

    logger.info(f"Before deduplication: {len(merged_df)} rows")

    # Deduplicate on key fields (keep first occurrence)
    # Include 'title' to handle cases where catalog_id=None (different unmatched movies)
    dedup_cols = ['catalog_id', 'date', 'channel', 'start_time', 'title']
    available_dedup_cols = [col for col in dedup_cols if col in merged_df.columns]

    merged_df = merged_df.drop_duplicates(subset=available_dedup_cols, keep='first')

    logger.info(f"After deduplication: {len(merged_df)} rows")

    # Sort by date
    merged_df = merged_df.sort_values(['date', 'start_time']).reset_index(drop=True)

    return merged_df
