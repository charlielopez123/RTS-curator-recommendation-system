"""
Catalog API Schemas - Response models for catalog browsing

Field types match catalog_schema.py definitions.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CatalogItem(BaseModel):
    """
    A single movie from the WhatsOn catalog.

    Field types match CATALOG_DTYPES from catalog_schema.py.
    """

    catalog_id: str = Field(..., description="Unique catalog identifier")
    title: str = Field(..., description="Movie title")
    best_title: Optional[str] = Field(None, description="Best available title")
    original_title: Optional[str] = Field(None, description="Original title")
    tmdb_id: Optional[int] = Field(None, description="TMDB ID (nullable)")

    # Metadata
    genres: Optional[List[str]] = Field(None, description="Movie genres")
    director: Optional[str] = Field(None, description="Director")
    actors: Optional[str] = Field(None, description="Main actors")
    production_year: Optional[float] = Field(None, description="Production year")
    production_regions: Optional[str] = Field(None, description="Production regions")
    duration_min: Optional[float] = Field(None, description="Duration in minutes")
    short_description: Optional[str] = Field(None, description="Short description")

    # Rights and broadcasts
    tv_rights_start: Optional[str] = Field(None, description="TV rights start date")
    tv_rights_end: Optional[str] = Field(None, description="TV rights end date")
    total_broadcasts: Optional[int] = Field(None, description="Total broadcasts allowed")
    available_broadcasts: Optional[int] = Field(None, description="Remaining broadcasts")
    times_shown: Optional[int] = Field(None, description="Times already broadcast")

    # TMDB features
    vote_average: Optional[float] = Field(None, description="TMDB vote average")
    popularity: Optional[float] = Field(None, description="TMDB popularity")
    release_date: Optional[str] = Field(None, description="Release date (YYYY-MM-DD)")
    original_language: Optional[str] = Field(None, description="Original language")


class CatalogQuery(BaseModel):
    """Query parameters for catalog search"""

    genre: Optional[str] = Field(None, description="Filter by genre")
    min_year: Optional[int] = Field(None, description="Minimum production year")
    max_year: Optional[int] = Field(None, description="Maximum production year")
    available_on: Optional[str] = Field(None, description="Filter by availability date (YYYY-MM-DD)")
    search: Optional[str] = Field(None, description="Search in titles")
    limit: int = Field(default=50, ge=1, le=500, description="Max results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")


class CatalogResponse(BaseModel):
    """Response for catalog queries"""

    items: List[CatalogItem] = Field(..., description="Catalog items matching query")
    total_count: int = Field(..., description="Total items in catalog")
    filtered_count: int = Field(..., description="Items matching filters")
    returned_count: int = Field(..., description="Items in this response")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")
