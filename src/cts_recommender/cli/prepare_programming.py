import argparse
from pathlib import Path
import logging
from cts_recommender.pipelines import programming_processing_pipeline
from cts_recommender import logging_setup

logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="Prepare programming dataset")
    p.add_argument("--rdata", help="Path to .RData file")
    p.add_argument("--out_processed", required=True, help="Path to output processed parquet")
    p.add_argument("--out_enriched", required=True, help="Path to output enriched parquet")
    p.add_argument("--out_ml", required=True, help="Path to output ml features parquet")
    
    p.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    a = p.parse_args()

    # Initialize logging
    logging_setup.setup_logging(a.log_level)
    _, out_processed = programming_processing_pipeline.run_original_Rdata_programming_pipeline(raw_file=Path(a.rdata), out_file=Path(a.out_processed))
    logger.info(f"Wrote {out_processed}")

    _, out_enriched = programming_processing_pipeline.run_enrich_programming_with_movie_metadata_pipeline(processed_file=out_processed, out_file=Path(a.out_enriched))
    logger.info(f"Wrote {out_enriched}")

    _, out_ml = programming_processing_pipeline.run_ML_feature_processing_pipeline(processed_file=out_processed, enriched_file=out_enriched, out_file=Path(a.out_ml))
    logger.info(f"Wrote {out_ml}")

if __name__ == "__main__":
    main()