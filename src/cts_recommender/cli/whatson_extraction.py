import argparse
from pathlib import Path
import logging

from cts_recommender.pipelines import whatson_extraction_pipeline
from cts_recommender import logging_setup

logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="Create WhatsOn movie catalog")
    p.add_argument("--csv", help="Path to WhatsOn CSV file")
    p.add_argument("--out_init_extraction", required=True, help="Path to output initial whatson extraction parquet")
    p.add_argument("--out_enriched", help="Path to output TMDB-enriched parquet")
    p.add_argument("--out_catalog", help="Path to output final catalog parquet")
    p.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    a = p.parse_args()

    # Initialize logging
    logging_setup.setup_logging(a.log_level)

    # Stage 1: Extract movies
    _, out_movies = whatson_extraction_pipeline.run_whatson_extraction_pipeline(
        raw_file=Path(a.csv) if a.csv else None
    )
    logger.info(f"Wrote {out_movies}")

    # Stage 2: Enrich with TMDB
    _, out_enriched = whatson_extraction_pipeline.run_enrich_catalog_with_movie_metadata_pipeline(
        catalog_file=out_movies,
        out_file=Path(a.out_enriched) if a.out_enriched else None
    )
    logger.info(f"Wrote {out_enriched}")

    # Stage 3: Build catalog features
    _, out_catalog_features = whatson_extraction_pipeline.run_whatson_feature_processing_pipeline(
        enriched_file=out_enriched,
        out_file=Path(a.out_enriched) if a.out_enriched else None
    )
    logger.info(f"Wrote catalog features to {out_catalog_features}")

    # Stage 4: Create catalog with deduplication and catalog IDs
    _, out_catalog = whatson_extraction_pipeline.run_whatson_catalog_creation_pipeline(
        in_file=out_catalog_features,
        out_file=Path(a.out_catalog)
    )
    logger.info(f"Wrote {out_catalog}")

if __name__ == "__main__":
    main()