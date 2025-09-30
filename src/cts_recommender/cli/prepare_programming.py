import argparse
from pathlib import Path
import logging
from cts_recommender.pipelines import programming_pipeline 

logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="Prepare programming dataset")
    p.add_argument("--rdata", help="Path to .RData file")
    p.add_argument("--out_processed", required=True, help="Path to output processed parquet")
    p.add_argument("--out_enriched", required=True, help="Path to output enriched parquet")

    a = p.parse_args()
    _, out_processed = programming_pipeline.run_original_Rdata_programming_pipeline(raw_file=Path(a.rdata), out_file=Path(a.out_processed))
    logger.info(f"Wrote {out_processed}")

    _, out_enriched = programming_pipeline.run_enrich_programming_with_movie_metadata_pipeline(processed_file=out_processed, out_file=Path(a.out_enriched))
    logger.info(f"Wrote {out_enriched}")

    _, out_ml = programming_pipeline.run_ML_feature_processing_pipeline(processed_file=out_processed, enriched_file=out_enriched)
    logger.info(f"Wrote {out_ml}")

if __name__ == "__main__":
    main()