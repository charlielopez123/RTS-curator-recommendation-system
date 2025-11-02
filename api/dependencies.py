"""
API Dependencies - Singleton state management and FastAPI dependency injection

Manages lazy loading of models and data with background preloading support.
Inspired by the pattern from your previous film-rec project but simplified.
"""

import asyncio
import logging
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import Depends
import pandas as pd
import joblib

from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.settings import get_settings
from api.repositories.base import BaseRepository
from api.repositories.local import LocalFileRepository

logger = logging.getLogger(__name__)


class AppState:
    """
    Global application state - holds loaded models and data.

    Singleton pattern: one instance shared across all requests.
    """

    def __init__(self):
        self.repository: Optional[BaseRepository] = None
        self.cts_model: Optional[ContextualThompsonSampler] = None
        self.catalog_df: Optional[pd.DataFrame] = None
        self.audience_model: Optional[AudienceRatingsRegressor] = None
        self.historical_df: Optional[pd.DataFrame] = None

        # State tracking for lazy initialization
        self._initialized = False
        self._initializing = False
        self._initialization_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Load models and data (async for non-blocking startup).

        Loads in order:
        1. Repository (data access layer)
        2. Catalog (needed for recommendations)
        3. CTS model (core recommendation algorithm)
        4. Audience model (optional, for signal computation)
        5. Historical data (optional, for feedback context)
        """
        # Prevent duplicate initialization
        async with self._initialization_lock:
            if self._initialized:
                logger.debug("AppState already initialized")
                return

            if self._initializing:
                logger.debug("AppState initialization already in progress")
                return

            self._initializing = True
            logger.info("Initializing AppState...")

            try:
                # Initialize repository
                logger.info("Creating repository...")
                self.repository = LocalFileRepository()

                # Load catalog (required)
                logger.info("Loading catalog...")
                self.catalog_df = self.repository.get_catalog()
                logger.info(f"Catalog loaded: {len(self.catalog_df)} items")

                # Load CTS model (required)
                logger.info("Loading CTS model...")
                self.cts_model = self.repository.get_cts_model()
                logger.info(f"CTS model loaded: {self.cts_model.num_signals} signals")

                # Load audience model (optional - may not exist yet)
                try:
                    logger.info("Loading audience model...")
                    self.audience_model = self.repository.get_audience_model()
                    logger.info("Audience model loaded")
                except FileNotFoundError as e:
                    logger.warning(f"Audience model not found: {e}. Continuing without it.")
                    self.audience_model = None

                # Load historical data (optional - may not exist yet)
                try:
                    logger.info("Loading historical programming...")
                    self.historical_df = self.repository.get_historical_programming()
                    logger.info(f"Historical data loaded: {len(self.historical_df)} records")
                except FileNotFoundError as e:
                    logger.warning(f"Historical data not found: {e}. Continuing without it.")
                    self.historical_df = None

                # Load curator model (optional)
                try:
                    logger.info("Loading curator model...")
                    cfg = get_settings()
                    curator_model_path = cfg.models_dir / "curator_logistic_model.joblib"
                    self.curator_model = joblib.load(curator_model_path)
                    logger.info("Curator model loaded")
                except Exception as e:
                    logger.warning(f"Curator model not found or failed to load: {e}. Continuing without it.")
                    self.curator_model = None

                # Initialize TV Programming Environment
                logger.info("Initializing TVProgrammingEnvironment...")
                self.env = TVProgrammingEnvironment(
                    catalog_df=self.catalog_df,
                    historical_programming_df=self.historical_df,
                    audience_model=self.audience_model
                )
                logger.info("TVProgrammingEnvironment initialized")

                self._initialized = True
                self._initializing = False
                logger.info("AppState initialization complete!")

            except Exception as e:
                self._initializing = False
                logger.error(f"Failed to initialize AppState: {e}", exc_info=True)
                raise

    def is_ready(self) -> bool:
        """Check if app is ready to serve requests"""
        return self._initialized and self.cts_model is not None and self.catalog_df is not None

    def get_status(self) -> dict:
        """Get current initialization status"""
        return {
            "initialized": self._initialized,
            "initializing": self._initializing,
            "ready": self.is_ready(),
            "cts_model_loaded": self.cts_model is not None,
            "catalog_loaded": self.catalog_df is not None,
            "audience_model_loaded": self.audience_model is not None,
            "historical_loaded": self.historical_df is not None,
        }


# Global singleton instance
app_state = AppState()


async def preload_app_state() -> None:
    """
    Background task to preload heavy environment.

    Called during FastAPI startup to load models in background
    while the app binds to port immediately.
    """
    logger.info("Starting background preload of app state...")
    try:
        await app_state.initialize()
        logger.info("Background preload complete!")
    except Exception as e:
        logger.error(f"Background preload failed: {e}", exc_info=True)


def get_app_state() -> AppState:
    """
    FastAPI dependency to access app state.

    Usage in routers:
        @router.get("/example")
        async def example(state: AppState = Depends(get_app_state)):
            model = state.cts_model
            ...
    """
    # Ensure initialization (blocking if not initialized yet)
    if not app_state._initialized and not app_state._initializing:
        logger.warning("AppState not initialized, initializing synchronously...")
        # For sync initialization (fallback)
        asyncio.run(app_state.initialize())

    return app_state


def get_repository() -> BaseRepository:
    """
    FastAPI dependency to access repository.

    Usage in routers:
        @router.get("/example")
        async def example(repo: BaseRepository = Depends(get_repository)):
            catalog = repo.get_catalog()
            ...
    """
    state = get_app_state()
    if state.repository is None:
        raise RuntimeError("Repository not initialized")
    return state.repository


@asynccontextmanager
async def lifespan_handler(app):
    """
    FastAPI lifespan context manager for startup/shutdown.

    Usage in main.py:
        app = FastAPI(lifespan=lifespan_handler)
    """
    # Startup: preload environment in background
    logger.info("FastAPI starting up...")
    preload_task = asyncio.create_task(preload_app_state())

    yield  # App is now running

    # Shutdown: cleanup if needed
    logger.info("FastAPI shutting down...")
    # Wait for preload to finish if still running
    if not preload_task.done():
        logger.info("Waiting for preload task to finish...")
        await preload_task
    logger.info("Shutdown complete")
