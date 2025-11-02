"""
Feedback Router - Curator feedback endpoints (prepared for future implementation)
"""

import logging
from fastapi import APIRouter, Depends

from api.schemas.feedback import FeedbackRequest, FeedbackResponse
from api.dependencies import get_repository
from api.repositories.base import BaseRepository

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    repo: BaseRepository = Depends(get_repository)
) -> FeedbackResponse:
    """
    Submit curator feedback on recommendations.

    Currently: Logs feedback to file for later analysis.
    Future: Updates CTS model in real-time and triggers retraining.
    """
    logger.info(f"Feedback received for {request.date} @ {request.hour}:00 on {request.channel}")
    logger.info(f"Selected: {request.selected_movie_id}")
    logger.info(f"Unselected: {request.unselected_movie_ids}")

    # Save feedback to storage
    feedback_data = {
        "date": request.date,
        "hour": request.hour,
        "channel": request.channel,
        "selected_movie_id": request.selected_movie_id,
        "unselected_movie_ids": request.unselected_movie_ids,
        "context_features": request.context_features
    }

    try:
        repo.save_feedback(feedback_data)
        return FeedbackResponse(
            status="success",
            message="Feedback recorded successfully. Model will be retrained in next batch."
        )
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}", exc_info=True)
        return FeedbackResponse(
            status="error",
            message=f"Failed to save feedback: {str(e)}"
        )
