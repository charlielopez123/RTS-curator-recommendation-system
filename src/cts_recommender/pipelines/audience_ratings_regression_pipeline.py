"""
Random Forest regressor for audience ratings prediction.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

from cts_recommender.utils.reproducibility import set_global_seed
from cts_recommender.settings import get_settings
from cts_recommender.models.audience_ratings_regressor import AudienceRatingsRegressor

logger = logging.getLogger(__name__)


def train_audience_ratings_model(
    ML_features_path: Path,
    model_output_path: Optional[Path] = None,
    random_seed: Optional[int] = None,
    **model_params
) -> Tuple[AudienceRatingsRegressor, Dict[str, Any], Dict[str, Any]]:
    """
    Train the Audience Ratings Random Forest regressor.
    
    Parameters:
    ML_features_path (Path): Path to the ML features parquet file.
    model_output_path (Optional[Path]): Path to save the trained model (optional).
    random_seed (Optional[int]): Random seed for reproducibility (optional).
    **model_params: Additional model hyperparameters.   
        
    Returns:
    Tuple[AudienceRatingsRegressor, Dict[str, Any], Dict[str, Any]]:
        The trained regressor, training metrics, and testing metrics."""
    # Set global seed for reproducibility
    if random_seed is None:
        settings = get_settings()
        random_seed = settings.random_seed

    set_global_seed(random_seed)
    logger.info(f"Using random seed: {random_seed}")

    # Ensure model uses the same seed
    if 'random_state' not in model_params:
        model_params['random_state'] = random_seed

    regressor = AudienceRatingsRegressor(**model_params)
    X_train, X_test, y_train, y_test = regressor.prepare_data(
        ML_features_path, random_state=random_seed
    )
    train_metrics = regressor.train(X_train, y_train)
    test_metrics = regressor.evaluate(X_test, y_test)

    importance_df = regressor.get_feature_importance(X_train.columns.tolist())
    logger.info("Top 10 most important features:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    if model_output_path:
        regressor.save_model(model_output_path)

    return regressor, train_metrics, test_metrics