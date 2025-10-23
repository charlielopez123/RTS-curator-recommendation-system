"""
CLI for interactive CTS recommendation testing.

This script allows users to interactively test the Contextual Thompson Sampler
by viewing recommendations and providing feedback through a terminal interface.
"""

import argparse
import logging
from pathlib import Path

from cts_recommender.settings import get_settings
from cts_recommender import logging_setup
from cts_recommender.io import readers
from cts_recommender.features.catalog_schema import CATALOG_DTYPES, HISTORICAL_PROGRAMMING_DTYPES, enforce_dtypes
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor
from cts_recommender.models.curator_logistic_regressor import load_curator_model
from cts_recommender.interactive.interactive_testing import InteractiveTester

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive testing of Contextual Thompson Sampler recommendations"
    )

    cfg = get_settings()

    # Model paths
    parser.add_argument(
        "--cts-model",
        type=Path,
        default=cfg.models_dir / "cts_model.npz",
        help="Path to CTS model (.npz)"
    )

    parser.add_argument(
        "--curator-model",
        type=Path,
        default=cfg.models_dir / "curator_logistic_model.joblib",
        help="Path to curator model (.joblib)"
    )

    parser.add_argument(
        "--audience-model",
        type=Path,
        default=cfg.models_dir / "audience_ratings_model.joblib",
        help="Path to audience ratings model (.joblib)"
    )

    # Data paths
    parser.add_argument(
        "--catalog",
        type=Path,
        default=cfg.processed_dir / "whatson" / "whatson_catalog.parquet",
        help="Path to movie catalog (.parquet)"
    )

    parser.add_argument(
        "--historical-programming",
        type=Path,
        default=cfg.processed_dir / "programming" / "historical_programming.parquet",
        help="Path to historical programming data (.parquet)"
    )

    # Testing parameters
    parser.add_argument(
        "--num-days",
        type=int,
        default=7,
        help="Number of days to generate test contexts for (default: 7)"
    )

    parser.add_argument(
        "--num-recommendations",
        type=int,
        default=5,
        help="Number of recommendations per slate (default: 5)"
    )

    parser.add_argument(
        "--save-updated-model",
        type=Path,
        default=None,
        help="Path to save updated CTS model after testing (optional)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging_setup.setup_logging(args.log_level)
    logger.info("Starting interactive CTS testing...")

    # Load data
    logger.info(f"Loading catalog from {args.catalog}")
    catalog_df = readers.read_parquet(args.catalog)
    catalog_df = enforce_dtypes(catalog_df, CATALOG_DTYPES, skip_missing=True)

    logger.info(f"Loading historical programming from {args.historical_programming}")
    historical_programming_df = readers.read_parquet(args.historical_programming)
    historical_programming_df = enforce_dtypes(
        historical_programming_df,
        HISTORICAL_PROGRAMMING_DTYPES,
        skip_missing=True
    )

    # Load models
    logger.info(f"Loading audience model from {args.audience_model}")
    audience_model = AudienceRatingsRegressor()
    audience_model.load_model(args.audience_model)

    logger.info(f"Loading curator model from {args.curator_model}")
    curator_model = load_curator_model(args.curator_model)

    logger.info(f"Loading CTS model from {args.cts_model}")
    cts_model = ContextualThompsonSampler.load(args.cts_model)

    # Initialize environment
    logger.info("Initializing TV programming environment...")
    env = TVProgrammingEnvironment(catalog_df, historical_programming_df, audience_model)

    # Initialize interactive tester
    logger.info("Initializing interactive tester...")
    tester = InteractiveTester(
        env=env,
        cts_model=cts_model,
        curator_model=curator_model,
        catalog_df=catalog_df
    )

    # Generate test contexts
    logger.info(f"Generating {args.num_days} test contexts...")
    contexts = tester.generate_test_contexts(num_days=args.num_days)

    # Run interactive testing loop
    logger.info("Starting interactive testing loop...")
    logger.info("Follow the on-screen prompts to select recommendations and provide feedback.\n")

    tester.run_interactive_loop(
        contexts=contexts,
        num_recommendations=args.num_recommendations
    )

    # Save updated model if requested
    if args.save_updated_model:
        logger.info(f"Saving updated CTS model to {args.save_updated_model}")
        cts_model.save(args.save_updated_model)
        logger.info("Updated model saved successfully!")

    logger.info("\nInteractive testing completed!")
    logger.info(f"Total recommendations made: {len(tester.recommendation_history)}")


if __name__ == "__main__":
    main()
