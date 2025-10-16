"""
CLI for extracting historical programming data.

This script builds a historical programming dataset by matching TV broadcast
records from the enriched programming data with the WhatsOn movie catalog.
"""

import argparse
import logging
from pathlib import Path

from cts_recommender.pipelines import historical_programming_pipeline
from cts_recommender.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build historical programming dataset by matching broadcasts to catalog"
    )

    cfg = get_settings()

    parser.add_argument(
        "--programming",
        type=Path,
        default=cfg.processed_dir / "programming" / "programming_enriched.parquet",
        help="Path to enriched programming parquet file (default: data/processed/programming/programming_enriched.parquet)"
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=cfg.processed_dir / "whatson" / "whatson_catalog.parquet",
        help="Path to WhatsOn catalog parquet file (default: data/processed/whatson/whatson_catalog.parquet)"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.processed_dir / "programming" / "historical_programming.parquet",
        help="Path to save historical programming output (default: data/processed/whatson/historical_programming.parquet)"
    )

    args = parser.parse_args()

    logger.info("Starting historical programming extraction...")
    logger.info(f"Programming file: {args.programming}")
    logger.info(f"Catalog file: {args.catalog}")
    logger.info(f"Output file: {args.out}")

    # Validate input files exist
    if not args.programming.exists():
        logger.error(f"Programming file not found: {args.programming}")
        return 1

    if not args.catalog.exists():
        logger.error(f"Catalog file not found: {args.catalog}")
        return 1

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    try:
        historical_df, out_file = historical_programming_pipeline.run_historical_programming_pipeline(
            programming_file=args.programming,
            catalog_file=args.catalog,
            out_file=args.out
        )

        logger.info("✓ Historical programming extraction completed successfully")
        logger.info(f"✓ Output saved to: {out_file}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
