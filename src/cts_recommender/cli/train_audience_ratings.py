import argparse
from pathlib import Path
import logging
import json
from cts_recommender.pipelines.audience_ratings_regression_pipeline import train_audience_ratings_model
from cts_recommender import logging_setup

logger = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser(description="Train audience ratings regression model")
    p.add_argument("--ml_features", required=True, help="Path to ML features parquet file")
    p.add_argument("--log_level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")

    # Remove old arguments that are no longer needed
    # p.add_argument("--processed", required=True, help="Path to processed programming parquet file")
    # p.add_argument("--enriched", required=True, help="Path to enriched programming parquet file")
    p.add_argument("--model_output", help="Path to save trained model (optional)")
    p.add_argument("--metrics_output", help="Path to save metrics JSON (optional)")

    # Model hyperparameters
    p.add_argument("--n_estimators", type=int, default=100, help="Number of trees in random forest")
    p.add_argument("--max_depth", type=int, help="Maximum depth of trees")
    p.add_argument("--min_samples_split", type=int, default=2, help="Minimum samples to split node")
    p.add_argument("--min_samples_leaf", type=int, default=1, help="Minimum samples at leaf node")
    p.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")

    a = p.parse_args()

    # Initialize logging
    logging_setup.setup_logging(a.log_level)

    # Prepare model parameters
    model_params = {
        'n_estimators': a.n_estimators,
        'min_samples_split': a.min_samples_split,
        'min_samples_leaf': a.min_samples_leaf,
        'random_state': a.random_state
    }
    if a.max_depth:
        model_params['max_depth'] = a.max_depth

    # Train model
    model_output_path = Path(a.model_output) if a.model_output else None
    regressor, train_metrics, test_metrics = train_audience_ratings_model(
        ML_features_path=Path(a.ml_features),
        model_output_path=model_output_path,
        random_seed=a.random_state,
        **model_params
    )

    logger.info("Training completed successfully!")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"Test MAPE: {test_metrics['mape']:.4f}")

    # Save metrics if requested
    if a.metrics_output:
        metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'model_params': model_params
        }
        with open(a.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {a.metrics_output}")

if __name__ == "__main__":
    main()