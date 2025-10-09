"""
Historical Programming Pipeline

Orchestrates the extraction of historical TV programming data by matching
broadcast records to the WhatsOn catalog.
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
    out_file: Path | None = None,
    out_stats_file: Path | None = None
) -> tuple[pd.DataFrame, Path]:
    """
    Build historical programming dataset by matching TV broadcasts to catalog movies.

    This pipeline:
    1. Loads enriched programming data (with TMDB IDs)
    2. Loads WhatsOn catalog (indexed by catalog_id)
    3. Matches broadcasts to catalog entries using TMDB ID
    4. Creates historical programming with catalog_id and broadcast metadata


    Parameters:
    programming_file (Path): Path to programming_enriched.parquet
    catalog_file (Path): Path to whatson_catalog.parquet
    out_file (Path): Path to save historical programming
    out_stats_file (Path): Path to save broadcast statistics (optional)

    Returns:
    tuple[pd.DataFrame, Path]: Historical programming DataFrame and output file path
    """
    if programming_file is None:
        programming_file = cfg.processed_dir / "programming" / "programming_enriched.parquet"
    if catalog_file is None:
        catalog_file = cfg.processed_dir / "whatson" / "whatson_catalog.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "programming" / "historical_programming.parquet"
    if out_stats_file is None:
        out_stats_file = cfg.processed_dir / "whatson" / "broadcast_statistics.parquet"

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

    # Compute and save broadcast statistics
    stats = historical_programming.compute_broadcast_statistics(historical_df)
    atomic_write_parquet(stats, out_stats_file)
    logger.info(f"Broadcast statistics saved to {out_stats_file}")

    # Log summary statistics
    logger.info("\n=== Broadcast Statistics Summary ===")
    logger.info(f"Total unique movies broadcast: {len(stats)}")
    logger.info(f"Total broadcasts: {stats['times_shown'].sum()}")
    logger.info(f"Average broadcasts per movie: {stats['times_shown'].mean():.2f}")
    logger.info(f"Median broadcasts per movie: {stats['times_shown'].median():.0f}")
    logger.info(f"Most broadcast movie: {stats['times_shown'].max()} times")

    # Show records with and without catalog match
    has_catalog = historical_df['catalog_id'].notna()
    logger.info(f"\nRecords matched to catalog: {has_catalog.sum()} ({100*has_catalog.sum()/len(historical_df):.1f}%)")
    logger.info(f"Records without catalog match: {(~has_catalog).sum()} ({100*(~has_catalog).sum()/len(historical_df):.1f}%)")

    return historical_df, out_file
