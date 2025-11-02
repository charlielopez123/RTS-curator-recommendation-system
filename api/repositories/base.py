"""
Base Repository - Abstract interface for data access

This defines the contract that all repository implementations must follow.
Allows swapping between local files (current) and database (future) without
changing the rest of the API code.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import date, datetime
import pandas as pd

if TYPE_CHECKING:
    from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
    from cts_recommender.models.audience_regression.audience_ratings_regressor import AudienceRatingsRegressor


class BaseRepository(ABC):
    """Abstract base class for data repositories"""

    @abstractmethod
    def get_catalog(self) -> pd.DataFrame:
        """
        Get the complete WhatsOn movie catalog.

        Returns:
            DataFrame with columns: catalog_id, title, tmdb_id, genres, etc.
        """
        pass

    @abstractmethod
    def get_historical_programming(self) -> pd.DataFrame:
        """
        Get historical programming data (curator decisions).

        Returns:
            DataFrame with columns: date, channel, catalog_id, broadcast_time, etc.
        """
        pass

    @abstractmethod
    def get_audience_model(self) -> "AudienceRatingsRegressor":
        """
        Load the trained audience ratings prediction model.

        Returns:
            Trained AudienceRatingsRegressor model
        """
        pass

    @abstractmethod
    def get_cts_model(self) -> "ContextualThompsonSampler":
        """
        Load the trained Contextual Thompson Sampler model.

        Returns:
            ContextualThompsonSampler instance
        """
        pass

    @abstractmethod
    def save_cts_model(self, model: "ContextualThompsonSampler", version: str) -> None:
        """
        Save CTS model to storage.

        Args:
            model: ContextualThompsonSampler instance
            version: Version identifier (e.g., "v1.0.0")
        """
        pass

    @abstractmethod
    def get_competition_schedules(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Get competitor TV schedules.

        Args:
            start_date: Filter schedules from this date onwards
            end_date: Filter schedules up to this date

        Returns:
            DataFrame with competitor movie schedules
        """
        pass

    @abstractmethod
    def save_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """
        Save curator feedback (selection event).

        Args:
            feedback_data: Dict containing:
                - date: str
                - hour: int
                - channel: str
                - selected_movie_id: str
                - unselected_movie_ids: List[str]
                - context_features: List[float]
                - timestamp: datetime
        """
        pass

    @abstractmethod
    def get_feedback_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical feedback events.

        Args:
            limit: Max number of recent events to return

        Returns:
            List of feedback dictionaries sorted by timestamp (most recent first)
        """
        pass
