"""
Recommendation API Schemas - Request/response models for recommendation endpoints

Field types match catalog_schema.py definitions for data consistency.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    """Request to generate movie recommendations"""

    date: str = Field(
        ...,
        description="Date for recommendations in ISO format (YYYY-MM-DD)"
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day for recommendations (0-23)"
    )
    channel: str = Field(
        ...,
        description="Channel identifier (e.g., 'RTS 1', 'RTS 2')"
    )
    top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )
    exclude_movie_ids: List[str] = Field(
        default_factory=list,
        description="Catalog IDs to exclude from recommendations"
    )

    @field_validator('date')
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v: str) -> str:
        """Validate channel is one of the supported channels"""
        valid_channels = ['RTS 1', 'RTS 2', 'RTS 1_T_PL', 'RTS 2_T_PL']
        if v not in valid_channels:
            raise ValueError(f"Channel must be one of {valid_channels}")
        return v


class Recommendation(BaseModel):
    """
    A single movie recommendation with metadata and scoring details.

    Field types match catalog_schema.py:
    - catalog_id: string (index)
    - tmdb_id: Int64 (nullable)
    - genres: list[str]
    - duration_min: float64
    - production_year: float64
    - tv_rights_start/end: datetime64[ns]
    - director, actors: string
    """

    # Core identifiers (from CATALOG_DTYPES)
    catalog_id: str = Field(..., description="Unique catalog identifier (string)")
    title: str = Field(..., description="Movie title")
    tmdb_id: Optional[int] = Field(None, description="The Movie Database ID (nullable)")

    # Movie metadata (from CATALOG_DTYPES)
    genres: Optional[List[str]] = Field(None, description="Movie genres (list)")
    director: Optional[str] = Field(None, description="Director name (string)")
    actors: Optional[str] = Field(None, description="Main actors (string, comma-separated)")
    production_year: Optional[float] = Field(None, description="Production year (float64, nullable)")
    duration_min: Optional[float] = Field(None, description="Runtime in minutes (float64)")
    short_description: Optional[str] = Field(None, description="Plot summary")
    original_language: Optional[str] = Field(None, description="Original language code")

    # Rights availability (from CATALOG_DTYPES: datetime64[ns])
    tv_rights_start: Optional[str] = Field(None, description="Rights start date (ISO format)")
    tv_rights_end: Optional[str] = Field(None, description="Rights end date (ISO format)")

    # TMDB features (from CATALOG_DTYPES)
    vote_average: Optional[float] = Field(None, description="TMDB vote average (float64)")
    popularity: Optional[float] = Field(None, description="TMDB popularity score (float64)")

    # Scoring details (computed by CTS)
    total_score: float = Field(..., description="Final weighted score")
    signal_contributions: Dict[str, float] = Field(
        ...,
        description="Raw signal values (before weighting)"
    )
    signal_weights: Dict[str, float] = Field(
        ...,
        description="CTS-learned weights for this context"
    )
    weighted_signals: Dict[str, float] = Field(
        ...,
        description="Signal contributions × weights"
    )

    # Predictions (from audience model)
    predicted_audience: Optional[float] = Field(
        None,
        description="Predicted audience size (rt_m in thousands)"
    )


class RecommendationResponse(BaseModel):
    """Response containing list of recommendations with metadata"""

    recommendations: List[Recommendation] = Field(
        ...,
        description="Ordered list of recommended movies (best first)"
    )
    context: Dict[str, Any] = Field(
        ...,
        description="Context used for recommendations (date, hour, weekday, etc.)"
    )
    model_version: str = Field(
        ...,
        description="CTS model version used"
    )
    candidates_considered: int = Field(
        ...,
        description="Number of movies considered before filtering"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when recommendations were generated"
    )
