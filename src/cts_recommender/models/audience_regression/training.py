import pandas as pd
import numpy as np
import logging

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

def train_model(model: BaseEstimator, X_train: pd.DataFrame , y_train: pd.Series, log_transform: bool = False) -> BaseEstimator:
    """
    Train a machine learning model.

    Args:
        model (BaseEstimator): The machine learning model to be trained.
        X_train: Training features.
        y_train: Training target.
        log_transform (bool): Whether to apply log(1 + y) transformation to the target variable.

    Returns:
        BaseEstimator: The trained machine learning model.
    """
    if log_transform:
        y_train = np.log1p(y_train)  # log(1 + y)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series,
                   log_transform: bool = False, eval_ranking_metrics: bool = False) -> dict:
    """
    Evaluate a machine learning model.

    Args:
        model (BaseEstimator): The machine learning model to be evaluated.
        X_test: Test features.
        y_test: Test target.
        log_transform (bool): Whether the target variable was log-transformed during training.
        eval_ranking_metrics (bool): Whether to compute ranking metrics. This can be computationally expensive for large datasets.

    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    y_pred = model.predict(X_test)
    if log_transform:
        y_test = np.log1p(y_test)  # log(1 + y)

    model_name = model.__class__.__name__

    # Compute basic regression metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    nonzero_idx = np.nonzero(y_test)[0]
    mape = mean_absolute_percentage_error(y_test.iloc[nonzero_idx], y_pred[nonzero_idx])

    errors = abs(y_test - y_pred)
    std_error = errors.std()

    # Initialize metrics dictionary
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'std_error': std_error,
        'n_samples': len(y_test)
    }

    # Log the metrics
    logger.info(f"MSE {model_name}: {mse}")
    logger.info(f"MAE {model_name}: {mae}")
    logger.info(f"MAPE {model_name}: {mape:.3f}")
    logger.info(f"R2 {model_name}: {r2}")
    logger.info(f"Standard error deviation: {std_error}")

    if eval_ranking_metrics:
            ranking_df = pd.DataFrame({
                'true_score': y_test,
                'pred_score': y_pred,
                })
            
            # Add ranks (1 = lowest, n = highest)
            ranking_df['true_rank'] = ranking_df['true_score'].rank(method='average')
            ranking_df['pred_rank'] = ranking_df['pred_score'].rank(method='average')


            # Efficient vectorized pairwise accuracy computation
            y_true_arr = y_test.values
            y_pred_arr = y_pred

            # Create all pairwise comparisons using broadcasting
            true_diffs = y_true_arr[:, None] > y_true_arr[None, :]
            pred_diffs = y_pred_arr[:, None] > y_pred_arr[None, :]

            # Only count upper triangle (avoid double counting)
            upper_triangle = np.triu(np.ones_like(true_diffs, dtype=bool), k=1)

            agreements = (true_diffs == pred_diffs) & upper_triangle
            pairwise_acc = agreements.sum() / upper_triangle.sum()

            rho, _ = spearmanr(ranking_df['true_rank'], ranking_df['pred_rank'])

            # Add ranking metrics to dictionary
            metrics.update({
                'pairwise_accuracy': pairwise_acc,
                'spearman_correlation': rho
            })

            logger.info(f"Pairwise ranking accuracy: {pairwise_acc:.3f}")
            logger.info(f"Spearman Rank Correlation: {rho:.4f}")

    return metrics


