"""
CLI for training the curator logistic regression model.
"""

import argparse
import logging
from pathlib import Path

from cts_recommender.models.curator_logistic_regressor import train_curator_model
from cts_recommender.settings import get_settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Train the curator logistic regression model."
    )

    cfg = get_settings()

    parser.add_argument(
        "--training-data",
        type=Path,
        default=cfg.processed_dir / "IL" / "training_data.joblib",
        help="Path to the training data for imitation learning."
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=cfg.models_dir / "curator_logistic_model.joblib",
        help="Path to save the trained curator model."
    )

    args = parser.parse_args()

    try:
        train_curator_model(
            training_data_file=args.training_data,
            model_output_file=args.out
        )
        logger.info(f"SUCCESS: Curator model saved to {args.out}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
