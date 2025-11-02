"""
Competition Service - Scrapes and processes competitor TV schedules

Wraps the CompetitorDataScraper and provides a clean interface for the API.
"""

import logging
from typing import List, Optional
from pathlib import Path
import pandas as pd

from cts_recommender.competition.competition_scraping import CompetitorDataScraper
from cts_recommender.competition.competition_processing import flatten_schedule, merge_with_historical
from cts_recommender.adapters.tmdb.tmdb import TMDB_API
from cts_recommender.settings import get_settings

from api.schemas.competition import (
    CompetitionSyncRequest,
    CompetitionSyncResponse,
    CompetitionScheduleResponse,
    CompetitionScheduleItem
)

logger = logging.getLogger(__name__)


def scrape_competition_schedules(
    request: CompetitionSyncRequest,
    catalog_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None
) -> CompetitionSyncResponse:
    """
    Scrape competitor schedules and save to storage.

    Args:
        request: CompetitionSyncRequest with scraping parameters
        catalog_df: WhatsOn catalog for title matching
        historical_df: Optional existing historical programming data

    Returns:
        CompetitionSyncResponse with scraping results
    """
    logger.info(f"Starting competition schedule scraping: {request.days_ahead} days ahead")

    try:
        # Initialize scraper
        scraper = CompetitorDataScraper()

        # Scrape upcoming schedules (downloads XMLs)
        logger.info("Scraping competitor schedules...")
        scraper.scrape_upcoming_schedules(days_ahead=request.days_ahead)

        # Parse downloaded XMLs to extract movie schedules
        logger.info("Parsing downloaded XMLs...")
        cfg = get_settings()
        competition_folder = cfg.project_root / "competition_data"
        if competition_folder.exists():
            scraper.process_competition_folder(competition_folder)
        else:
            logger.warning(f"Competition data folder not found: {competition_folder}")

        # Flatten and process schedules
        logger.info("Processing scraped schedules...")
        tmdb_api = TMDB_API()

        competition_df = flatten_schedule(
            scraped_schedules=scraper.scraped_schedules,
            catalog_df=catalog_df,
            tmdb_api=tmdb_api,
            use_tmdb_fallback=True
        )

        if len(competition_df) == 0:
            return CompetitionSyncResponse(
                status="failed",
                message="No competition data scraped",
                scraped_entries=0,
                matched_entries=0,
                found_in_tmdb_only=0,
                not_found=0,
                match_rate=0.0,
                channels_scraped=[]
            )

        # Compute statistics
        total_scraped = len(competition_df)
        matched_to_catalog = competition_df['catalog_id'].notna().sum()

        # Count entries found in TMDB but not in catalog (has tmdb_id but no catalog_id)
        found_in_tmdb_only = competition_df[
            (competition_df['tmdb_id'].notna()) &
            (competition_df['catalog_id'].isna())
        ].shape[0]

        # Count entries not found anywhere (no tmdb_id and no catalog_id)
        not_found = competition_df[
            (competition_df['tmdb_id'].isna()) &
            (competition_df['catalog_id'].isna())
        ].shape[0]

        match_rate = matched_to_catalog / total_scraped if total_scraped > 0 else 0.0
        channels_scraped = list(scraper.scraped_schedules.keys())

        # Save to processed directory
        cfg = get_settings()
        output_path = cfg.processed_dir / "competition_schedules.parquet"

        # Merge with existing historical data if provided
        if historical_df is not None:
            logger.info("Merging with existing historical programming...")
            competition_df = merge_with_historical(
                competition_df=competition_df,
                historical_df=historical_df
            )

        # Save processed schedules
        output_path.parent.mkdir(parents=True, exist_ok=True)
        competition_df.to_parquet(output_path)
        logger.info(f"Saved competition schedules to {output_path}")

        return CompetitionSyncResponse(
            status="success",
            message=f"Successfully scraped {total_scraped} entries from {len(channels_scraped)} channels. "
                    f"Matched {matched_to_catalog} to catalog, {found_in_tmdb_only} found in TMDB only, {not_found} not found.",
            scraped_entries=total_scraped,
            matched_entries=int(matched_to_catalog),
            found_in_tmdb_only=int(found_in_tmdb_only),
            not_found=int(not_found),
            match_rate=float(match_rate),
            channels_scraped=channels_scraped
        )

    except Exception as e:
        logger.error(f"Competition scraping failed: {e}", exc_info=True)
        return CompetitionSyncResponse(
            status="failed",
            message=f"Scraping failed: {str(e)}",
            scraped_entries=0,
            matched_entries=0,
            found_in_tmdb_only=0,
            not_found=0,
            match_rate=0.0,
            channels_scraped=[]
        )


def get_competition_schedules(
    catalog_df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    channels: Optional[List[str]] = None,
    limit: int = 100
) -> CompetitionScheduleResponse:
    """
    Get stored competition schedules with optional filtering.

    Args:
        catalog_df: WhatsOn catalog (for metadata enrichment)
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
        channels: Filter by channels
        limit: Max results to return

    Returns:
        CompetitionScheduleResponse with schedule items
    """
    cfg = get_settings()
    competition_file = cfg.processed_dir / "competition_schedules.parquet"

    if not competition_file.exists():
        logger.warning(f"Competition schedules file not found: {competition_file}")
        return CompetitionScheduleResponse(
            schedules=[],
            total_count=0,
            date_range={"start_date": None, "end_date": None}
        )

    # Load competition data
    df = pd.read_parquet(competition_file)

    # Apply filters
    if start_date:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['date'] <= pd.Timestamp(end_date)]
    if channels:
        df = df[df['channel'].isin(channels)]

    # Limit results
    df = df.head(limit)

    # Build schedule items
    schedules = []
    for _, row in df.iterrows():
        item = CompetitionScheduleItem(
            channel=str(row['channel']),
            date=row['date'].strftime('%Y-%m-%d'),
            start_time=str(row['start_time']),
            title=str(row['title']),
            catalog_id=str(row['catalog_id']) if pd.notna(row.get('catalog_id')) else None,
            tmdb_id=int(row['tmdb_id']) if pd.notna(row.get('tmdb_id')) else None
        )
        schedules.append(item)

    # Compute date range
    if len(df) > 0:
        date_range = {
            "start_date": df['date'].min().strftime('%Y-%m-%d'),
            "end_date": df['date'].max().strftime('%Y-%m-%d')
        }
    else:
        date_range = {"start_date": None, "end_date": None}

    return CompetitionScheduleResponse(
        schedules=schedules,
        total_count=len(schedules),
        date_range=date_range
    )
