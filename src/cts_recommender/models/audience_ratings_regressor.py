"""
Random Forest regressor for audience ratings prediction.
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cts_recommender.features.TV_programming_Rdata_schema import TARGET_FEATURE
from cts_recommender.models.training import train_model, evaluate_model
from cts_recommender.io.readers import read_parquet

logger = logging.getLogger(__name__)

class AudienceRatingsRegressor:
    """Random Forest regressor for predicting audience ratings (rt_m)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        **kwargs
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names_ = None  # Store feature names from training

    def prepare_data(
        self,
        ML_features_path: Path,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info("Loading processed and enriched programming data...")
        feature_df = read_parquet(ML_features_path)

        logger.info("Building feature matrix and targets...")

        # Separate features and target
        X = feature_df.drop(columns=[TARGET_FEATURE])
        y = feature_df[TARGET_FEATURE]

        # Encode categorical variables (one-hot encoding)
        logger.info("Encoding categorical variables...")
        X_encoded = pd.get_dummies(X, drop_first=False)

        logger.info(f"Feature matrix shape before encoding: {X.shape}")
        logger.info(f"Feature matrix shape after encoding: {X_encoded.shape}")
        logger.info(f"Target distribution: mean={y.mean():.3f}, std={y.std():.3f}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=random_state
        )

        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, scale_features: bool = False, log_transform: bool = False) -> Dict[str, Any]:
        logger.info("Training Random Forest regressor...")

        if scale_features:
            logger.info("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

        # Store feature names before training (critical for prediction)
        self.feature_names_ = list(X_train.columns)
        logger.info(f"Storing {len(self.feature_names_)} feature names for prediction")

        self.model = train_model(self.model, X_train, y_train, log_transform=log_transform)
        self.is_trained = True

        logger.info("Evaluating model on train set...")
        train_metrics = evaluate_model(self.model, X_train, y_train, log_transform=log_transform, eval_ranking_metrics=True)
        train_metrics['split'] = 'train'

        logger.info("Training completed successfully!")
        return train_metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series, scale_features: bool = False, log_transform: bool = False) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        logger.info("Evaluating model on test set...")

        if scale_features:
            X_test_scaled = self.scaler.transform(X_test)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        test_metrics = evaluate_model(self.model, X_test, y_test, log_transform=log_transform, eval_ranking_metrics=True)
        test_metrics['split'] = 'test'

        return test_metrics

    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        importance_df = pd.DataFrame({
            'feature': feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, model_path: Path) -> None:
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names_  # Save feature names
        }, model_path)
        logger.info(f"Model saved to {model_path} with {len(self.feature_names_)} features")

    def load_model(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        saved_data = joblib.load(model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.is_trained = saved_data['is_trained']
        self.feature_names_ = saved_data.get('feature_names', None)  # Load feature names (backward compatible)

        if self.feature_names_ is None:
            logger.warning("Model loaded without feature names - predictions may fail")
        else:
            logger.info(f"Model loaded from {model_path} with {len(self.feature_names_)} features")

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names expected by the model.

        Returns:
            List of feature names in the order expected by the model

        Raises:
            ValueError: If model hasn't been trained or loaded
        """
        if not self.is_trained or self.feature_names_ is None:
            raise ValueError("Model must be trained or loaded before retrieving feature names")
        return self.feature_names_
