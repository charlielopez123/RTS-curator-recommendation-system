"""
Recommendations Router - Main endpoint for generating movie recommendations
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from api.schemas.recommendations import RecommendationRequest, RecommendationResponse
from api.services import recommendation_service
from api.dependencies import get_app_state, AppState

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    state: AppState = Depends(get_app_state)
) -> RecommendationResponse:
    """
    Generate movie recommendations for a given context.

    This endpoint uses the Contextual Thompson Sampler (CTS) model to score
    and rank candidate movies based on multiple value signals (audience, competition,
    diversity, novelty, rights, curator preference).

    Args:
        request: RecommendationRequest with date, hour, channel, and filtering options
        state: Application state (injected dependency)

    Returns:
        RecommendationResponse with ordered list of recommendations

    Raises:
        HTTPException: If models not loaded or no candidates available
    """
    # Check if system is ready
    if not state.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Models are still loading."
        )

    # Check required components
    if state.env is None:
        raise HTTPException(
            status_code=500,
            detail="TV Programming Environment not initialized"
        )

    if state.curator_model is None:
        logger.warning("Curator model not available")
        raise HTTPException(
            status_code=500,
            detail="Curator model not loaded. Cannot compute recommendations."
        )

    try:
        # Generate recommendations
        response = recommendation_service.generate_recommendations(
            request=request,
            env=state.env,
            cts_model=state.cts_model,
            curator_model=state.curator_model,
            catalog_df=state.catalog_df
        )

        logger.info(
            f"Successfully generated {len(response.recommendations)} recommendations "
            f"for {request.date} @ {request.hour}:00 on {request.channel}"
        )

        return response

    except ValueError as e:
        # User-facing errors (no candidates, invalid request, etc.)
        logger.warning(f"Recommendation request failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected errors
        logger.error(f"Recommendation generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )
