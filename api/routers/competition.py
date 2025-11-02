"""
Competition Router - Competitor schedule scraping endpoints
"""

import logging
from fastapi import APIRouter, Depends, HTTPException

from api.schemas.competition import (
    CompetitionSyncRequest,
    CompetitionSyncResponse,
    CompetitionScheduleResponse
)
from api.services import competition_service
from api.dependencies import get_app_state, AppState

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sync", response_model=CompetitionSyncResponse)
async def sync_competition_schedules(
    request: CompetitionSyncRequest,
    state: AppState = Depends(get_app_state)
) -> CompetitionSyncResponse:
    """
    Scrape competitor TV schedules and save to storage.

    This endpoint scrapes upcoming movie schedules from competitor channels
    (TF1, M6, etc.) and matches them to the WhatsOn catalog.
    """
    if not state.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Models are still loading."
        )

    try:
        response = competition_service.scrape_competition_schedules(
            request=request,
            catalog_df=state.catalog_df,
            historical_df=state.historical_df
        )
        return response

    except Exception as e:
        logger.error(f"Competition sync failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to sync competition schedules: {str(e)}"
        )


@router.get("/schedules", response_model=CompetitionScheduleResponse)
async def get_competition_schedules(
    start_date: str = None,
    end_date: str = None,
    limit: int = 100,
    state: AppState = Depends(get_app_state)
) -> CompetitionScheduleResponse:
    """
    Get stored competition schedules with optional filtering.
    """
    if not state.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )

    try:
        response = competition_service.get_competition_schedules(
            catalog_df=state.catalog_df,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        return response

    except Exception as e:
        logger.error(f"Failed to get competition schedules: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve schedules: {str(e)}"
        )
