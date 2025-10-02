import argparse
from pathlib import Path
import logging

from cts_recommender.pipelines import whatson_extraction_pipeline
from cts_recommender import logging_setup

logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="Extracting Whatson file into a parquet file")
    p.add_argument("--file_path", required=True, help="Path to Whatson file")
    p.add_argument("--out_file", required=True, help="Path to output parquet")
    p.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    a = p.parse_args()

    # Initialize logging
    logging_setup.setup_logging(a.log_level)
    _, out_processed = whatson_extraction_pipeline.run_whatson_extration_pipeline(raw_file=Path(a.file_path), out_file=Path(a.out_file))
    logger.info(f"Wrote {out_processed}")

if __name__ == "__main__":
    main()