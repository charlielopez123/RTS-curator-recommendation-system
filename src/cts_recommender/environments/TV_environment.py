import pandas as pd
from typing import Optional
from datetime import datetime
from typing import Union, Tuple, Dict
import numpy as np

from cts_recommender.environments.schemas import Context, TimeSlot, Season

class TVProgrammingEnvironment:
    def __init__(
            self,
            catalog_df: pd.DataFrame,
            historical_programming_df: Optional[pd.DataFrame] = None,
            ):
        
        self.catalog_df = catalog_df
        self.historical_programming_df = historical_programming_df

        self.memory_size = 100  # Max memory size
        self.memory = []

        # Movies available to be shown, initialised to empty
        self.available_movies = []

        # State tracking
        self.context_features_cache: set[Tuple] = {}
        

    def get_available_movies(self, date: datetime):
        """Get movies available for the given context (rights not expired), updates self.available_movies"""

        available_mask = ((self.catalog_df['end_rights'] > date) &
                            (self.catalog_df['start_rights'] < date) &
                            (self.catalog_df['available_num_diff'] > 0))
        
        self.available_movies = self.catalog_df[available_mask].index.tolist()


    def update_memory(self, movie_id: int):
        """Update self.memory with the given movie_id, maintaining the memory size limit"""
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append(movie_id)

    def get_context_features(self, context: Union[Context, Tuple]) -> np.ndarray:

        """
        Convert context to feature vector from either a given Context or previous context_cache_key

        feature_vector = [time_slot_hot (4,), day_of_week_hot (7,), is_weekend, season_one_hot(4,1)] -> shape: (16, 1)
        """
        # Cache key for efficiency
        if context is Tuple:
            cache_key = context
        else:
            cache_key = (context.hour, context.day_of_week, 
                        context.month, context.season.value)
        
        if cache_key in self.context_features_cache:
            return cache_key
        
        features = []

        # Slot hour of showing into 4 different TimeSlot
        try:
            time_slot_value = self.get_time_slot(context.hour).value
        except:
            print(context.hour)
        time_slots = [time_slot.value for time_slot in TimeSlot]
        
        time_slot_features = [1 if time_slot == time_slot_value else 0 
                          for time_slot in time_slots]
        features.extend(time_slot_features)

        # Day-of-week one-hot (0=Monday ... 6=Sunday)
        dow_one_hot = [1 if context.day_of_week == i else 0 for i in range(7)]
        features.extend(dow_one_hot)
        
        # Weekend flag
        is_weekend = 1 if context.day_of_week >= 5 else 0
        features.append(is_weekend)
        
        # Season one-hot
        seasons = [season.value for season in Season]
        season_features = [1 if season == context.season.value else 0 
                          for season in seasons]
        features.extend(season_features)
        
        feature_vector = np.array(features, dtype=np.float32)
        self.context_features_cache[cache_key] = feature_vector
        return feature_vector, cache_key
    
    def get_time_slot(self, hour: int) -> TimeSlot | None:

        if 6 <= hour < 14:
            return TimeSlot.MORNING
        elif 14 <= hour < 19:
            return TimeSlot.AFTERNOON
        elif 19 <= hour < 22:
            return TimeSlot.PRIME_TIME
        elif 22 <= hour < 26:
            return TimeSlot.LATE_NIGHT
        else:
            return None  # outside defined slots