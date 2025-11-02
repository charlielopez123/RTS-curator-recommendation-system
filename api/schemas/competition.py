"""
Competition API Schemas - Request/response models for competitor schedule scraping
"""

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class CompetitionSyncRequest(BaseModel):
    """Request to scrape competitor schedules"""

    days_ahead: int = Field(
        default=21,
        ge=1,
        le=90,
        description="Number of days ahead to scrape (1-90)"
    )
    channels: Optional[List[str]] = Field(
        default=None,
        description="Specific channels to scrape (None = all channels)"
    )


class CompetitionScheduleItem(BaseModel):
    """A single competitor movie schedule entry"""

    channel: str = Field(..., description="Competitor channel (e.g., 'TF1', 'M6')")
    date: str = Field(..., description="Broadcast date (YYYY-MM-DD)")
    start_time: str = Field(..., description="Start time (HH:MM:SS)")
    title: str = Field(..., description="Movie title")
    catalog_id: Optional[str] = Field(None, description="Matched catalog ID (if found)")
    tmdb_id: Optional[int] = Field(None, description="TMDB ID (if matched)")


class CompetitionSyncResponse(BaseModel):
    """Response after syncing competitor schedules"""

    status: str = Field(..., description="Sync status (success, partial, failed)")
    message: str = Field(..., description="Human-readable message")
    scraped_entries: int = Field(..., description="Total entries scraped")
    matched_entries: int = Field(..., description="Entries matched to WhatsOn catalog")
    found_in_tmdb_only: int = Field(default=0, description="Found in TMDB but not in catalog (movie exists but not available on RTS)")
    not_found: int = Field(default=0, description="Not found in TMDB or catalog")
    match_rate: float = Field(..., description="Catalog match rate (0-1)")
    channels_scraped: List[str] = Field(..., description="Channels successfully scraped")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When scraping completed"
    )


class CompetitionScheduleResponse(BaseModel):
    """Response for getting competition schedules"""

    schedules: List[CompetitionScheduleItem] = Field(
        ...,
        description="Competitor movie schedules"
    )
    total_count: int = Field(..., description="Total schedule entries")
    date_range: Dict[str, str] = Field(
        ...,
        description="Date range covered (start_date, end_date)"
    )
