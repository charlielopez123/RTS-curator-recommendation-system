from datetime import datetime, timedelta
from typing import Any, Dict, List
import pandas as pd

from cts_recommender.competition.competition_scraping import CompetitorDataScraper
from cts_recommender.competition.competition_processing import flatten_schedule, merge_with_historical
from cts_recommender.adapters.tmdb.tmdb import TMDB_API

class MovieCompetitorContext:
    def __init__(self, movie_id: str, reference_date: datetime):
        self.movie_id = movie_id
        self.reference_date = reference_date
        self.competitor_showings: List[Dict[str, Any]] = []  # List of competitor airings

    def add_competitor_showing(self, channel: str, air_date: datetime, air_hour: int):
        """Add a competitor showing of this specific movie"""
        self.competitor_showings.append({
            'channel': channel,
            'air_date': air_date,
            'air_hour': air_hour,
        })

class CompetitorDataManager:
    def __init__(self, 
                competition_historical_programming_df: pd.DataFrame,
                ):

        self.competition_historical_df = competition_historical_programming_df
        self.competition_scraper = CompetitorDataScraper()

    def get_movie_competitor_context(self, 
                                    movie_id: str,
                                    reference_date: datetime, 
                                    window_days_back=365//2, # Check 6 months back and 3 weeks ahead
                                    window_days_ahead=21) -> MovieCompetitorContext: 
        """Retrieve competitor context for a given movie around a reference date."""

        competitor_context = MovieCompetitorContext(movie_id, reference_date)

        # Search historical data
        if self.competition_historical_df is not None:
            if reference_date > datetime.now() + timedelta(days=window_days_ahead):
                windowed_competitor_historical_data = self.competition_historical_df[
                    ( self.competition_historical_df['date'] >= (reference_date - timedelta(days=window_days_back)) ) &
                    ( self.competition_historical_df['date'] <= (reference_date + timedelta(days=window_days_ahead)) )
                    ]
            else:
                windowed_competitor_historical_data = self.competition_historical_df[
                    (self.competition_historical_df['date'] >= (reference_date - timedelta(days=window_days_back))) &
                    (self.competition_historical_df['date'] <= (reference_date + timedelta(days=window_days_ahead)))
                    ]
                
            competitor_historical_showings = windowed_competitor_historical_data[windowed_competitor_historical_data['catalog_id'] == movie_id]

            for _, showing in competitor_historical_showings.iterrows():
                competitor_context.add_competitor_showing(
                    channel=showing['channel'],
                    air_date=showing['date'],
                    air_hour=showing['hour'],
                )
        return competitor_context
    

    def update_competition_historical_data(self):
        self.competition_scraper.scrape_upcoming_schedules()
        self.competition_scraper.process_competition_folder()
        future_showings_df = flatten_schedule(self.competition_scraper.scraped_schedules)
        self.competitor_historical_data = merge_with_historical(self.competitor_historical_data, future_showings_df)


    def saturday_within_three_weeks(from_date=datetime.today()) -> datetime:
        # Returns date for the saturday within 3 weeks i.e the furthest saturday from which we can scrape competitor showings
        today = datetime.today()
        target = today + timedelta(weeks=2)

        # Saturday is weekday 5 (Mon=0); find offset to next Saturday (0 if already Saturday)
        days_until_sat = (5 - target.weekday()) % 7
        return target + timedelta(days=days_until_sat)
