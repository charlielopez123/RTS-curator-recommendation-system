"""
CLI for extracting imitation learning training data from historical programming.
Trains curator logistic regression and warm-starts Contextual Thompson Sampler.
"""

import argparse
import logging
from pathlib import Path

from cts_recommender.pipelines import IL_training_data_pipeline
from cts_recommender.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Extract IL training data from historical programming"
    )

    cfg = get_settings()

    parser.add_argument(
        "--historical",
        type=Path,
        default=cfg.processed_dir / "programming" / "historical_programming.parquet",
        help="Path to historical programming parquet"
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=cfg.processed_dir / "whatson" / "whatson_catalog.parquet",
        help="Path to catalog parquet"
    )

    parser.add_argument(
        "--audience-model",
        type=Path,
        default=cfg.models_dir / "audience_ratings_model.joblib",
        help="Path to audience ratings model"
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.processed_dir / "IL" / "training_data.joblib",
        help="Path to save training data"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.6,
        help="Weight for curator selection vs pseudo-rewards (default: 0.6)"
    )

    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=5,
        help="Negative samples per positive sample (default: 5)"
    )

    parser.add_argument(
        "--split-date",
        type=str,
        default=None,
        help="Date to split train/val (YYYY-MM-DD, optional)"
    )

    args = parser.parse_args()

    try:
        training_data, out_file = IL_training_data_pipeline.run_IL_training_data_pipeline(
            historical_programming_file=args.historical,
            catalog_file=args.catalog,
            audience_model_file=args.audience_model,
            out_file=args.out,
            gamma=args.gamma,
            negative_sampling_ratio=args.negative_ratio,
            time_split_date=args.split_date
        )

        logger.info(f"SUCCESS: Training data saved to {out_file}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
