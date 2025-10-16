"""
Historical Programming Pipeline

Orchestrates the extraction of historical TV programming data by matching
broadcast records to the WhatsOn catalog.

Architecture:
- Historical programming is a FACT TABLE (star schema)
- Contains broadcast events: when, where, and what was broadcast
- Links to catalog via catalog_id (no feature duplication)
- TMDB features (genres, ratings, etc.) live in the catalog only
- Join historical_df with catalog_df to get movie features
"""

from pathlib import Path
import pandas as pd
import logging

from cts_recommender.preprocessing import historical_programming
from cts_recommender.io.writers import atomic_write_parquet
from cts_recommender.settings import get_settings

cfg = get_settings()
logger = logging.getLogger(__name__)


def run_historical_programming_pipeline(
    programming_file: Path | None = None,
    catalog_file: Path | None = None,
    out_file: Path | None = None
) -> tuple[pd.DataFrame, Path]:
    """
    Build historical programming dataset by matching TV broadcasts to catalog movies.

    This pipeline:
    1. Loads enriched programming data (with TMDB IDs for matching)
    2. Loads WhatsOn catalog (indexed by catalog_id)
    3. Matches broadcasts to catalog entries using TMDB ID
    4. Creates historical programming with catalog_id and broadcast metadata ONLY
       (no TMDB features - those live in the catalog)

    Output schema includes:
    - catalog_id: Link to catalog (dimension table)
    - Broadcast metadata: date, channel, time, title
    - Audience metrics: rt_m, pdm
    - Temporal features: hour, weekday, season, holidays
    - tmdb_id: For matching purposes only

    To get movie features, join with catalog:
        enriched = historical_df.merge(catalog_df, left_on='catalog_id',
                                       right_index=True, how='left')

    Parameters:
    programming_file (Path): Path to programming_enriched.parquet
    catalog_file (Path): Path to whatson_catalog.parquet
    out_file (Path): Path to save historical programming

    Returns:
    tuple[pd.DataFrame, Path]: Historical programming DataFrame and output file path
    """
    if programming_file is None:
        programming_file = cfg.processed_dir / "programming" / "programming_enriched.parquet"
    if catalog_file is None:
        catalog_file = cfg.processed_dir / "whatson" / "whatson_catalog.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "programming" / "historical_programming.parquet"

    logger.info("=== Historical Programming Pipeline ===")
    logger.info(f"Programming data: {programming_file}")
    logger.info(f"Catalog data: {catalog_file}")

    # Load data
    programming_df = historical_programming.load_enriched_programming(programming_file)
    catalog_df = historical_programming.load_whatson_catalog(catalog_file)

    # Build historical programming
    historical_df = historical_programming.build_historical_programming(
        programming_df=programming_df,
        catalog_df=catalog_df
    )

    # Save to parquet
    atomic_write_parquet(historical_df, out_file)
    logger.info(f"Historical programming saved to {out_file}")

    # Show records with and without catalog match
    has_catalog = historical_df['catalog_id'].notna()
    logger.info(f"\nRecords matched to catalog: {has_catalog.sum()} ({100*has_catalog.sum()/len(historical_df):.1f}%)")
    logger.info(f"Records without catalog match: {(~has_catalog).sum()} ({100*(~has_catalog).sum()/len(historical_df):.1f}%)")

    return historical_df, out_file
