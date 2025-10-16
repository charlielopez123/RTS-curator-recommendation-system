"""
Times Shown Tracker for Imitation Learning

Dynamically tracks how many times each movie has been broadcast during
offline IL training, combining catalog broadcast dates and historical
programming records filtered by reference date.
"""

import pandas as pd
from typing import List, Optional

from cts_recommender.features.whatson_schema import SHOWINGS_COLUMNS


class TimesShownTracker:
    """
    Dynamically tracks times_shown for each catalog item during IL training.

    Counts broadcasts from:
    1. Catalog broadcast date columns (first_broadcast_date, rebroadcast_date_1-4)
    2. Historical programming records from interest channels only

    The tracker is read-only - it computes times_shown on-demand by filtering
    broadcasts before a given reference_date. As the reference_date advances
    chronologically during iteration, the count naturally increases.
    """

    def __init__(self,
                 catalog_df: pd.DataFrame,
                 historical_df: pd.DataFrame,
                 interest_channels: List[str]):
        """
        Args:
            catalog_df: WhatsOn catalog with broadcast date columns
            historical_df: Historical programming data (all channels)
            interest_channels: List of channels to count (e.g., ['RTS1', 'RTS2'])
        """
        self.catalog_df = catalog_df
        self.historical_df = historical_df
        self.interest_channels = interest_channels

    def get_times_shown(self, catalog_id: str, reference_date: pd.Timestamp) -> int:
        """
        Count total broadcasts BEFORE reference_date.

        Args:
            catalog_id: Movie catalog ID
            reference_date: Count only broadcasts before this date

        Returns:
            Total number of broadcasts before reference_date
        """
        count = 0

        # 1. Count from catalog broadcast date columns
        if catalog_id in self.catalog_df.index:
            for col in SHOWINGS_COLUMNS:
                date_val: pd.Timestamp = self.catalog_df.loc[catalog_id, col]
                if pd.notna(date_val) and date_val < reference_date:
                    count += 1

        # 2. Count from historical data (interest channels only)
        hist_mask = (
            (self.historical_df['catalog_id'] == catalog_id) &
            (self.historical_df['date'] < reference_date) &
            (self.historical_df['channel'].isin(self.interest_channels))
        )
        count += hist_mask.sum()

        return int(count)

    def get_last_broadcast_date(self, catalog_id: str, reference_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Get most recent broadcast date BEFORE reference_date.

        Args:
            catalog_id: Movie catalog ID
            reference_date: Consider only broadcasts before this date

        Returns:
            Most recent broadcast date, or None if never broadcast
        """
        dates = []

        # 1. Collect dates from catalog broadcast columns
        if catalog_id in self.catalog_df.index:
            for col in SHOWINGS_COLUMNS:
                date_val = self.catalog_df.loc[catalog_id, col]
                if pd.notna(date_val) and date_val < reference_date:
                    dates.append(date_val)

        # 2. Collect dates from historical data (interest channels only)
        hist_dates = self.historical_df[
            (self.historical_df['catalog_id'] == catalog_id) &
            (self.historical_df['date'] < reference_date) &
            (self.historical_df['channel'].isin(self.interest_channels))
        ]['date'].tolist()
        dates.extend(hist_dates)

        return max(dates) if dates else None
