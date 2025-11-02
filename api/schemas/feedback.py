"""
Feedback API Schemas - Request/response models for curator feedback

Used to record curator selections and update the CTS model (future feature).
"""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    """
    Curator feedback on recommendations.

    Records which movie was selected and which were rejected,
    allowing the CTS model to learn curator preferences.
    """

    date: str = Field(..., description="Date of selection (YYYY-MM-DD)")
    hour: int = Field(..., ge=0, le=23, description="Hour of selection (0-23)")
    channel: str = Field(..., description="Channel (e.g., 'RTS 1')")

    selected_movie_id: str = Field(
        ...,
        description="Catalog ID of the movie selected by curator"
    )
    unselected_movie_ids: List[str] = Field(
        ...,
        description="Catalog IDs of movies that were shown but not selected"
    )

    context_features: Optional[List[float]] = Field(
        None,
        description="Context feature vector (optional, for advanced usage)"
    )


class FeedbackResponse(BaseModel):
    """Response after submitting feedback"""

    status: str = Field(..., description="Status of feedback submission")
    message: str = Field(..., description="Human-readable message")
    feedback_id: Optional[str] = Field(None, description="Unique feedback ID (future)")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When feedback was recorded"
    )
