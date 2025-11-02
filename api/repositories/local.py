"""
Local File Repository - File-based data access implementation

Loads data from local parquet and joblib files.
Environment-agnostic: paths are configured via settings (reads from .env).
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import date, datetime

import pandas as pd
import joblib

from cts_recommender.settings import get_settings
from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor
from cts_recommender.features.catalog_schema import CATALOG_DTYPES, HISTORICAL_PROGRAMMING_DTYPES, enforce_dtypes
from api.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class LocalFileRepository(BaseRepository):
    """Repository implementation using local file storage"""

    def __init__(self):
        self.settings = get_settings()
        self._catalog_cache: Optional[pd.DataFrame] = None
        self._historical_cache: Optional[pd.DataFrame] = None
        self._audience_model_cache: Optional[AudienceRatingsRegressor] = None
        self._cts_model_cache: Optional[ContextualThompsonSampler] = None

        # File paths (actual names from data directory)
        self.catalog_path = self.settings.processed_dir / "whatson" / "whatson_catalog.parquet"
        self.historical_path = self.settings.processed_dir / "programming" / "historical_programming.parquet"
        self.audience_model_path = self.settings.models_dir / "audience_ratings_model.joblib"
        self.cts_model_path = self.settings.models_dir / "cts_model"  # Will be .npz + .rng.pkl
        self.feedback_log_path = self.settings.data_root / "feedback_log.jsonl"

        logger.info(f"LocalFileRepository initialized with data_root: {self.settings.data_root}")

    def get_catalog(self) -> pd.DataFrame:
        """Load WhatsOn catalog from parquet file"""
        if self._catalog_cache is not None:
            logger.debug("Returning cached catalog")
            return self._catalog_cache

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog file not found: {self.catalog_path}. "
                "Run 'make extract-whatson' to generate it."
            )

        logger.info(f"Loading catalog from {self.catalog_path}")
        self._catalog_cache = pd.read_parquet(self.catalog_path)
        # Enforce schema to ensure date columns are datetime64[ns] not strings
        self._catalog_cache = enforce_dtypes(self._catalog_cache, CATALOG_DTYPES, skip_missing=True)
        logger.info(f"Loaded {len(self._catalog_cache)} catalog items")
        return self._catalog_cache

    def get_historical_programming(self) -> pd.DataFrame:
        """Load historical programming data from parquet file"""
        if self._historical_cache is not None:
            logger.debug("Returning cached historical programming")
            return self._historical_cache

        if not self.historical_path.exists():
            raise FileNotFoundError(
                f"Historical programming file not found: {self.historical_path}. "
                "Run 'make historical-programming' to generate it."
            )

        logger.info(f"Loading historical programming from {self.historical_path}")
        self._historical_cache = pd.read_parquet(self.historical_path)
        # Enforce schema to ensure date columns are datetime64[ns] not strings
        self._historical_cache = enforce_dtypes(self._historical_cache, HISTORICAL_PROGRAMMING_DTYPES, skip_missing=True)
        logger.info(f"Loaded {len(self._historical_cache)} programming records")
        return self._historical_cache

    def get_audience_model(self) -> AudienceRatingsRegressor:
        """Load trained audience ratings model from joblib file"""
        if self._audience_model_cache is not None:
            logger.debug("Returning cached audience model")
            return self._audience_model_cache

        if not self.audience_model_path.exists():
            raise FileNotFoundError(
                f"Audience model not found: {self.audience_model_path}. "
                "Run 'make train-audience-ratings' to generate it."
            )

        logger.info(f"Loading audience model from {self.audience_model_path}")
        # Create instance and load from file (handles dict unpacking)
        model = AudienceRatingsRegressor()
        model.load_model(self.audience_model_path)
        self._audience_model_cache = model
        logger.info("Audience model loaded successfully")
        return self._audience_model_cache

    def get_cts_model(self) -> ContextualThompsonSampler:
        """
        Load trained CTS model using its built-in load() method.

        CTS model saves as two files:
        - cts_model.npz: Model parameters (U, b, h_U, h_b) and hyperparameters
        - cts_model.rng.pkl: Random number generator state
        """
        if self._cts_model_cache is not None:
            logger.debug("Returning cached CTS model")
            return self._cts_model_cache

        # Check if .npz file exists (CTS saves as .npz + .rng.pkl)
        npz_path = self.cts_model_path.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(
                f"CTS model not found: {npz_path}. "
                "Run CTS training pipeline to generate it."
            )

        logger.info(f"Loading CTS model from {self.cts_model_path}")
        # Use CTS's built-in load method
        self._cts_model_cache = ContextualThompsonSampler.load(self.cts_model_path)
        # Get dimensions from model parameter shapes (num_signals, context_dim not stored as attrs)
        num_signals = self._cts_model_cache.U.shape[0]
        context_dim = self._cts_model_cache.U.shape[1]
        logger.info(f"CTS model loaded: {num_signals} signals, {context_dim} context dimensions")
        return self._cts_model_cache

    def save_cts_model(self, model: ContextualThompsonSampler, version: str) -> None:
        """
        Save CTS model using its built-in save() method with versioning.

        Creates versioned copy and updates main model file.
        """
        # Save to versioned path
        version_dir = self.settings.models_dir / "versions" / version
        version_dir.mkdir(parents=True, exist_ok=True)
        version_path = version_dir / "cts_model"

        logger.info(f"Saving CTS model version {version} to {version_path}")
        # Use CTS's built-in save method (creates .npz + .rng.pkl)
        model.save(version_path)

        # Also update the main model file
        logger.info(f"Updating main CTS model at {self.cts_model_path}")
        model.save(self.cts_model_path)
        logger.info(f"CTS model saved successfully")

        # Invalidate cache
        self._cts_model_cache = None

    def get_competition_schedules(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Load competitor schedules from stored data.

        Note: Competition data is typically scraped dynamically via
        CompetitorDataScraper. This method loads previously saved
        competition data if available.
        """
        competition_file = self.settings.processed_dir / "competition_schedules.parquet"

        if not competition_file.exists():
            logger.warning(
                f"Competition schedules file not found: {competition_file}. "
                "Use CompetitorDataScraper to scrape and save competition data. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        logger.info(f"Loading competition schedules from {competition_file}")
        df = pd.read_parquet(competition_file)

        # Apply date filters if provided
        if 'date' in df.columns:
            if start_date is not None:
                df = df[df['date'] >= pd.Timestamp(start_date)]
            if end_date is not None:
                df = df[df['date'] <= pd.Timestamp(end_date)]

        logger.info(f"Loaded {len(df)} competition schedule records")
        return df

    def save_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """
        Save feedback to JSONL file (append mode).

        Each line is a JSON object with feedback details.
        Future: migrate to database table.
        """
        # Ensure feedback log file exists
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert datetime to ISO string if present
        if 'timestamp' in feedback_data and isinstance(feedback_data['timestamp'], datetime):
            feedback_data['timestamp'] = feedback_data['timestamp'].isoformat()

        # Append to JSONL file
        with open(self.feedback_log_path, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')

        logger.info(f"Feedback saved to {self.feedback_log_path}")

    def get_feedback_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load feedback history from JSONL file.

        Returns most recent feedback events first.
        """
        if not self.feedback_log_path.exists():
            logger.info("No feedback history found")
            return []

        feedback_list: List[Dict[str, Any]] = []

        with open(self.feedback_log_path, 'r') as f:
            for line in f:
                if line.strip():
                    feedback_list.append(json.loads(line))

        # Sort by timestamp (most recent first)
        feedback_list.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )

        # Apply limit if specified
        if limit is not None:
            feedback_list = feedback_list[:limit]

        logger.info(f"Loaded {len(feedback_list)} feedback records")
        return feedback_list

    def clear_cache(self) -> None:
        """Clear all cached data (useful for development/testing)"""
        logger.info("Clearing repository cache")
        self._catalog_cache = None
        self._historical_cache = None
        self._audience_model_cache = None
        self._cts_model_cache = None
