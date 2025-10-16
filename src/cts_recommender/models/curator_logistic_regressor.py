"""
Functions for training the curator logistic regression model.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual frequencies by binning
    predictions and comparing average confidence to average accuracy in each bin.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins to use for calibration curve.

    Returns:
        ECE value (lower is better, 0 = perfect calibration).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # Last bin includes upper boundary
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)

        if np.any(mask):
            acc = y_true[mask].mean()  # Actual accuracy in bin
            conf = y_prob[mask].mean()  # Average confidence in bin
            weight = mask.mean()  # Proportion of samples in bin
            ece += np.abs(acc - conf) * weight

    return float(ece)


def _load_and_prepare_data(training_data_file: str) -> tuple:
    """
    Load training data and prepare feature matrices.

    Args:
        training_data_file: Path to joblib file containing training data.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val).
    """
    logger.info(f"Loading training data from {training_data_file}")
    training_data = joblib.load(training_data_file)

    train_data = training_data["train"]
    val_data = training_data["val"]

    # Concatenate context and movie features
    X_train = np.concatenate(
        [train_data["context_features"], train_data["movie_features"]], axis=1
    )
    y_train = train_data["curator_targets"].astype(int).ravel()

    X_val = np.concatenate(
        [val_data["context_features"], val_data["movie_features"]], axis=1
    )
    y_val = val_data["curator_targets"].astype(int).ravel()

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(
        f"Training class distribution - Positive: {np.sum(y_train)}, "
        f"Negative: {len(y_train) - np.sum(y_train)}"
    )
    logger.info(f"Validation data shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val


def _train_base_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Train base logistic regression model with balanced class weights.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Trained LogisticRegression model.
    """
    logger.info("Training curator logistic regression model...")
    model = LogisticRegression(
        class_weight="balanced",  # Handle class imbalance
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _calibrate_model(
    model: LogisticRegression, X_cal: np.ndarray, y_cal: np.ndarray
) -> CalibratedClassifierCV:
    """
    Calibrate model probabilities using isotonic regression.

    Args:
        model: Trained base model.
        X_cal: Calibration features.
        y_cal: Calibration labels.

    Returns:
        Calibrated model.
    """
    logger.info("Calibrating model with isotonic regression...")

    # sklearn 1.6+ deprecation: cv='prefit' will be removed in 1.8
    # For now, we use try/except for compatibility
    try:
        calibrated_model = CalibratedClassifierCV(
            estimator=model, method="isotonic", cv="prefit"
        )
    except TypeError:
        # Fallback for older sklearn versions
        calibrated_model = CalibratedClassifierCV(
            base_estimator=model, method="isotonic", cv="prefit"
        )

    calibrated_model.fit(X_cal, y_cal)
    return calibrated_model


def train_curator_model(training_data_file: str, model_output_file: str) -> None:
    """
    Trains and calibrates a logistic regression model on curator selection data.

    Training pipeline:
    1. Load training and validation data
    2. Train LogisticRegression with class_weight='balanced' on full training set
    3. Split validation set into calibration/test (50/50)
    4. Calibrate model using isotonic regression on calibration set
    5. Evaluate on held-out test set with comprehensive metrics
    6. Save calibrated model

    The model predicts the probability that a curator would select a given movie
    for broadcast, based on context features (time, day, etc.) and movie features
    (genre, ratings, etc.).

    Args:
        training_data_file: Path to joblib file containing training/validation data.
        model_output_file: Path where the calibrated model will be saved.
    """
    # Step 1: Load and prepare data
    X_train, y_train, X_val, y_val = _load_and_prepare_data(training_data_file)

    # Step 2: Split validation into calibration and test sets
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42, stratify=y_val
    )
    logger.info(f"Calibration data shape: {X_cal.shape}")
    logger.info(f"Test data shape: {X_test.shape}")

    # Step 3: Train base model
    model = _train_base_model(X_train, y_train)

    # Step 4: Calibrate model
    calibrated_model = _calibrate_model(model, X_cal, y_cal)

    # Step 5: Evaluate on test set
    logger.info("Evaluating calibrated model on test set...")
    y_pred = calibrated_model.predict(X_test)
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

    # Calculate standard classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba, labels=[0, 1])
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    ece = _expected_calibration_error(y_test, y_pred_proba, n_bins=10)

    # Log overall metrics
    logger.info(f"\nTest Accuracy@0.5: {accuracy:.4f}")
    logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
    logger.info(f"Test Log Loss: {logloss:.4f}")
    logger.info(f"Test Brier Score: {brier:.4f}")
    logger.info(f"Expected Calibration Error (ECE@10): {ece:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

    # Step 6: Save model
    logger.info(f"Saving trained and calibrated model to {model_output_file}")
    joblib.dump(calibrated_model, model_output_file)
    logger.info("Curator model training and calibration complete.")
