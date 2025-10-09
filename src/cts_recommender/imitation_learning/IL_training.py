import pandas as pd
import logging
from tqdm.auto import tqdm
from typing import Dict



from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.environments.schemas import Context, Season
from cts_recommender.RTS_constants import INTEREST_CHANNELS


logger = logging.getLogger(__name__)

class HistoricalDataProcessor:
    """
    Processes historical programming data for offline training, using past programming decisions 
    to train the curator model and warm start the Contextual Thompson Sampler weights
    """
    def __init__(self, 
                 environment: TVProgrammingEnvironment,
                 historical_data = pd.DataFrame):
        
        self.env = environment
        self.historical_df = historical_data

    def prepare_training_data(self,
                            negative_sampling_ratio: float = 5,
                            time_split_date: str = None) -> Dict:
        """
        Prepare training data from historical programming decisions
            
        Args:
            negative_sampling_ratio: Ratio of negative samples to positive samples
            time_split_date: Split date for temporal validation (format: 'YYYY-MM-DD')
            
        Returns:
            Dictionary with training data for both networks
        """

        logger.info("Processing historical programming data for RTS 1 and RTS 2...")

        # Keep historical only from RTS channels of interest
        interest_channel_historical_df: pd.DataFrame = self.historical_df[self.historical_df['channel'].isin(INTEREST_CHANNELS)].copy()

        logger.info(f"Found {len(interest_channel_historical_df)} programming decisions for {INTEREST_CHANNELS}")

        # Prepare positive samples (movies that were actually shown)
        all_samples = []
        # Keep track of all pseudo reward signals (value signals) for each programmming decision
        all_rewards = []
        # Keep track of number of positive samples considered
        num_successful_samples = 0

        # reset memory
        self.env.memory = []
        self.env.reward.memory = self.env.memory

        # Iterate through each row of the interest channel's historical decisions
        for _, row in tqdm(interest_channel_historical_df.iterrows(), total=len(interest_channel_historical_df), desc="Processing rows"): 

            # Create context from historical data
            context = self._create_context_from_row(row)

            air_date = row['date']

            # Set historical data based off what is available for current date
            available_historical_df = self.historical_df[self.historical_df['date'] < air_date].copy()

            movie_id = row['catalog_id']

            # Store movie ID in memory
            self.env.update_memory(movie_id) 

            # Set movie times shown based on historical data if available
            if pd.isna(self.env.catalog_df.loc[movie_id, 'date_diff_1']):
                self.env.catalog_df.loc[movie_id, 'times_shown'] = 0
            else:
                show_cols = ['date_diff_1', 'date_rediff_1', 'date_rediff_2', 'date_rediff_3', 'date_rediff_4']
                mask = self.env.catalog_df.loc[movie_id, show_cols].lt(air_date)
                self.env.catalog_df.loc[movie_id, "times_shown"] = mask.sum()
            
            if movie_id not in self.env.catalog_df.index:
                    logger.info(f"Movie ID {movie_id} not found in catalog, skipping row")
                    continue
            
            context_features, context_cache_key = self.env.get_context_features(context)

            


    def _create_context_from_row(self, row: pd.Series) -> Context:
        """Create Context object from historical data row"""
        
        # Map time to time slot
        hour = row.get('hour')
        day_of_week=row.get('weekday')
        
        # Map month to season
        month = row.get('month')
        if month in [3, 4, 5]:
            season = Season.SPRING
        elif month in [6, 7, 8]:
            season = Season.SUMMER
        elif month in [9, 10, 11]:
            season = Season.AUTUMN
        else:
            season = Season.WINTER
        
        return Context(
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            season=season,
        )
    
    