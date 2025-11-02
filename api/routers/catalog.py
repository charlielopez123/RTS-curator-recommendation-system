"""
Catalog Router - Browse WhatsOn movie catalog
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
import pandas as pd

from api.schemas.catalog import CatalogResponse, CatalogItem
from api.dependencies import get_app_state, AppState

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/catalog", response_model=CatalogResponse)
async def browse_catalog(
    genre: Optional[str] = Query(None, description="Filter by genre"),
    min_year: Optional[int] = Query(None, description="Minimum production year"),
    max_year: Optional[int] = Query(None, description="Maximum production year"),
    search: Optional[str] = Query(None, description="Search in titles"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    state: AppState = Depends(get_app_state)
) -> CatalogResponse:
    """
    Browse the WhatsOn movie catalog with optional filtering and pagination.
    """
    if not state.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )

    try:
        df = state.catalog_df.copy()
        total_count = len(df)

        # Apply filters
        if genre:
            df = df[df['genres'].apply(lambda x: genre in x if isinstance(x, list) else False)]

        if min_year:
            df = df[df['production_year'] >= min_year]

        if max_year:
            df = df[df['production_year'] <= max_year]

        if search:
            search_lower = search.lower()
            df = df[
                df['title'].str.lower().str.contains(search_lower, na=False) |
                df['best_title'].str.lower().str.contains(search_lower, na=False)
            ]

        filtered_count = len(df)

        # Pagination
        df = df.iloc[offset:offset+limit]

        # Build catalog items
        items = []
        for catalog_id, row in df.iterrows():
            item = CatalogItem(
                catalog_id=str(catalog_id),
                title=str(row['title']),
                best_title=str(row['best_title']) if 'best_title' in row and not pd.isna(row['best_title']) else None,
                original_title=str(row['original_title']) if 'original_title' in row and not pd.isna(row['original_title']) else None,
                tmdb_id=int(row['tmdb_id']) if 'tmdb_id' in row and not pd.isna(row['tmdb_id']) else None,
                genres=row['genres'] if isinstance(row.get('genres'), list) else None,
                director=str(row['director']) if 'director' in row and not pd.isna(row['director']) else None,
                actors=str(row['actors']) if 'actors' in row and not pd.isna(row['actors']) else None,
                production_year=float(row['production_year']) if 'production_year' in row and not pd.isna(row['production_year']) else None,
                duration_min=float(row['duration_min']) if 'duration_min' in row and not pd.isna(row['duration_min']) else None,
                short_description=str(row['short_description']) if 'short_description' in row and not pd.isna(row['short_description']) else None,
                tv_rights_start=str(row['tv_rights_start']) if 'tv_rights_start' in row and not pd.isna(row['tv_rights_start']) else None,
                tv_rights_end=str(row['tv_rights_end']) if 'tv_rights_end' in row and not pd.isna(row['tv_rights_end']) else None,
                total_broadcasts=int(row['total_broadcasts']) if 'total_broadcasts' in row and not pd.isna(row['total_broadcasts']) else None,
                available_broadcasts=int(row['available_broadcasts']) if 'available_broadcasts' in row and not pd.isna(row['available_broadcasts']) else None,
                times_shown=int(row['times_shown']) if 'times_shown' in row and not pd.isna(row['times_shown']) else None,
                vote_average=float(row['vote_average']) if 'vote_average' in row and not pd.isna(row['vote_average']) else None,
                popularity=float(row['popularity']) if 'popularity' in row and not pd.isna(row['popularity']) else None
            )
            items.append(item)

        return CatalogResponse(
            items=items,
            total_count=total_count,
            filtered_count=filtered_count,
            returned_count=len(items),
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Catalog browse failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to browse catalog: {str(e)}"
        )
