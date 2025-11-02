"""
Health Router - Health checks and system status endpoints
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any

from api.dependencies import get_app_state, AppState

router = APIRouter()


@router.get("/ready")
async def health_check_ready(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    """
    Kubernetes readiness probe.

    Returns ready=True when all models are loaded and system is ready to serve requests.
    """
    status = state.get_status()

    return {
        "ready": state.is_ready(),
        "details": status
    }


@router.get("/model-info")
async def get_model_info(state: AppState = Depends(get_app_state)) -> Dict[str, Any]:
    """
    Get information about loaded models and data sources.
    """
    info = {
        "cts_model": None,
        "catalog": None,
        "audience_model": None,
        "historical_data": None
    }

    if state.cts_model is not None:
        info["cts_model"] = {
            "loaded": True,
            "num_signals": state.cts_model.num_signals,
            "context_dim": state.cts_model.context_dim,
            "learning_rate": state.cts_model.lr,
            "exploration_scale": state.cts_model.expl_scale
        }

    if state.catalog_df is not None:
        info["catalog"] = {
            "loaded": True,
            "total_movies": len(state.catalog_df),
            "columns": list(state.catalog_df.columns)
        }

    if state.audience_model is not None:
        info["audience_model"] = {
            "loaded": True
        }

    if state.historical_df is not None:
        info["historical_data"] = {
            "loaded": True,
            "total_records": len(state.historical_df)
        }

    return info
