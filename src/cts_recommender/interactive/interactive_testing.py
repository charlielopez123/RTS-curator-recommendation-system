"""
Interactive testing framework for CTS recommendation system.

This module provides the main InteractiveTester class for running interactive
testing sessions with human feedback on CTS recommendations.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from cts_recommender.environments.schemas import Context, Season, Channel
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.RTS_constants import INTEREST_CHANNELS
from cts_recommender.utils import dates
from cts_recommender.utils.interactive import (
    display_recommendations,
    get_user_choice,
    get_signal_feedback,
    compute_signal_vector,
    print_summary,
    SIGNAL_NAMES
)


def get_channels_from_names(channel_names: List[str]) -> List[Channel]:
    """Map channel name strings to Channel enum values."""
    channel_map = {
        'RTS 1': Channel.RTS1,
        'RTS 2': Channel.RTS2
    }
    return [channel_map[name] for name in channel_names if name in channel_map]


class InteractiveTester:
    """
    Interactive testing framework for CTS recommendations.

    This class orchestrates the interactive testing loop where users can:
    - View CTS recommendations for different contexts
    - Select recommendations and provide signal feedback
    - Regenerate recommendations by rejecting current slate
    - Track recommendation history and weight evolution
    """

    def __init__(
        self,
        env: TVProgrammingEnvironment,
        cts_model: ContextualThompsonSampler,
        curator_model,
        catalog_df: pd.DataFrame,
        signal_names: Optional[List[str]] = None
    ):
        """
        Initialize the InteractiveTester.

        Args:
            env: TVProgrammingEnvironment instance
            cts_model: Contextual Thompson Sampler model
            curator_model: Curator acceptance probability model
            catalog_df: DataFrame containing movie catalog information
            signal_names: Optional list of signal names (defaults to SIGNAL_NAMES)
        """
        self.env = env
        self.cts_model = cts_model
        self.curator_model = curator_model
        self.catalog_df = catalog_df
        self.signal_names = signal_names or SIGNAL_NAMES

        # Tracking
        self.recommendation_history: List[Dict[str, Any]] = []
        self.weight_evolution: List[Dict[str, Any]] = []

    def generate_test_contexts(
        self,
        num_days: int = 7,
        hour_options: Optional[List[int]] = None,
        channel_options: Optional[List[Channel]] = None,
        start_date: Optional[datetime] = None
    ) -> List[Tuple[pd.Timestamp, Context]]:
        """
        Generate test contexts for interactive testing.

        Args:
            num_days: Number of days to generate contexts for
            hour_options: List of hours to sample from (default: [10, 21, 23])
            channel_options: List of channels to sample from (default: [RTS1, RTS2])
            start_date: Starting date (default: today)

        Returns:
            List of (test_date, context) tuples
        """
        if hour_options is None:
            hour_options = [10, 21, 23]
        if channel_options is None:
            channel_options = [Channel.RTS1, Channel.RTS2]
        if start_date is None:
            start_date = datetime.now()

        contexts = []
        for i in range(num_days, 0, -1):
            test_date = pd.Timestamp(start_date - timedelta(days=i))

            # Randomly select hour and channel
            hour = np.random.choice(hour_options)
            channel = np.random.choice(channel_options)

            # Get season
            season_str = dates.get_season(test_date)
            season_map = {
                'spring': Season.SPRING,
                'summer': Season.SUMMER,
                'fall': Season.AUTUMN,
                'winter': Season.WINTER
            }
            season = season_map[season_str]

            # Build context
            context = Context(
                hour=hour,
                day_of_week=test_date.dayofweek,
                month=test_date.month,
                season=season,
                channel=channel
            )

            contexts.append((test_date, context))

        return contexts

    def compute_signals_for_context(
        self,
        test_date: pd.Timestamp,
        context: Context,
        context_features: np.ndarray,
        rejected_movies: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute signal matrix for all available movies in a context.

        Args:
            test_date: Air date
            context: Context object
            context_features: Pre-computed context feature vector
            rejected_movies: List of catalog IDs to exclude

        Returns:
            Tuple of (signal_matrix, available_catalog_ids)
        """
        signal_matrix = []
        available_catalog_ids = []

        # Calculate expected number of movies to process
        expected_to_process = len(self.env.available_movies) - len(rejected_movies) - len(self.env.memory)

        # Compute signals for all available movies (excluding memory and rejected)
        for catalog_id in tqdm(
            self.env.available_movies,
            desc=f"Computing signals for {test_date.strftime('%Y-%m-%d')}",
            total=expected_to_process,
            unit="movie"
        ):
            # Skip movies that are in memory or rejected for this context
            if catalog_id in self.env.memory or catalog_id in rejected_movies:
                continue

            # Compute signal vector
            signal_vector = compute_signal_vector(
                catalog_id=catalog_id,
                test_date=test_date,
                context=context,
                context_features=context_features,
                env=self.env,
                curator_model=self.curator_model
            )

            signal_matrix.append(signal_vector)
            available_catalog_ids.append(catalog_id)

        return np.array(signal_matrix), available_catalog_ids

    def get_recommendations(
        self,
        context_features: np.ndarray,
        signal_matrix: np.ndarray,
        available_catalog_ids: List[str],
        num_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get top-K recommendations from CTS model.

        Args:
            context_features: Context feature vector
            signal_matrix: Matrix of signal vectors for all candidates
            available_catalog_ids: List of catalog IDs corresponding to signal_matrix rows
            num_recommendations: Number of recommendations to return

        Returns:
            List of recommendation dictionaries
        """
        # Get top-K recommendations from CTS
        topk_indices, topk_scores, sampled_weights, _ = self.cts_model.score_candidates(
            c_t=context_features,
            S_matrix=signal_matrix,
            K=min(num_recommendations, len(available_catalog_ids))
        )

        # Build recommendation slate
        recommendations = []
        for idx, score in zip(topk_indices, topk_scores):
            recommendations.append({
                'catalog_id': available_catalog_ids[idx],
                'score': score,
                'signals': signal_matrix[idx],
                'weights': sampled_weights
            })

        return recommendations

    def run_interactive_loop(
        self,
        contexts: List[Tuple[pd.Timestamp, Context]],
        num_recommendations: int = 5
    ) -> None:
        """
        Run the main interactive testing loop.

        Args:
            contexts: List of (test_date, context) tuples to test
            num_recommendations: Number of recommendations per slate
        """
        print(f"\n{'='*80}")
        print(f"Starting Interactive CTS Testing")
        print(f"Testing {len(contexts)} contexts")
        print(f"{'='*80}\n")

        for test_date, context in contexts:
            # Get available movies for this date
            self.env.get_available_movies(test_date)

            if len(self.env.available_movies) == 0:
                print(f"⚠️  No available movies for {test_date.strftime('%Y-%m-%d')}, skipping...")
                continue

            # Get context features
            context_features, _ = self.env.get_context_features(context)

            # Reset rejected movies for this new context
            rejected_movies = []

            # Inner loop to allow regeneration
            while True:
                # Compute signals for all available movies
                signal_matrix, available_catalog_ids = self.compute_signals_for_context(
                    test_date=test_date,
                    context=context,
                    context_features=context_features,
                    rejected_movies=rejected_movies
                )

                if len(available_catalog_ids) == 0:
                    print("⚠️  No more available movies after filtering (memory + rejections), skipping this context...")
                    break

                # Get recommendations
                recommendations = self.get_recommendations(
                    context_features=context_features,
                    signal_matrix=signal_matrix,
                    available_catalog_ids=available_catalog_ids,
                    num_recommendations=num_recommendations
                )

                # Display recommendations
                display_recommendations(
                    recommendations=recommendations,
                    context=context,
                    test_date=test_date,
                    catalog_df=self.catalog_df,
                    env=self.env,
                    signal_names=self.signal_names
                )

                # Get user input
                user_input = get_user_choice()

                if user_input == 'q':
                    print("\nExiting recommendation loop...\n")
                    print_summary(self.recommendation_history, self.signal_names)
                    return

                elif user_input == 's':
                    print("Skipping this context...\n")
                    break

                elif user_input == 'r':
                    # Mark all current recommendations as rejected for this context
                    rejected_movies.extend([rec['catalog_id'] for rec in recommendations])
                    print(f"Regenerating without {len(rejected_movies)} rejected movies...\n")
                    continue

                else:
                    # Try to parse as recommendation selection
                    if not self._handle_recommendation_selection(
                        user_input=user_input,
                        recommendations=recommendations,
                        test_date=test_date,
                        context=context,
                        context_features=context_features
                    ):
                        # Invalid input, continue loop
                        continue

                    # Valid selection, break inner loop and move to next context
                    break

        # Print final summary
        print_summary(self.recommendation_history, self.signal_names)

    def _handle_recommendation_selection(
        self,
        user_input: str,
        recommendations: List[Dict[str, Any]],
        test_date: pd.Timestamp,
        context: Context,
        context_features: np.ndarray
    ) -> bool:
        """
        Handle user selection of a recommendation.

        Args:
            user_input: User input string
            recommendations: List of recommendation dictionaries
            test_date: Test date
            context: Context object
            context_features: Context feature vector

        Returns:
            True if selection was valid and processed, False otherwise
        """
        try:
            choice = int(user_input)
            if 1 <= choice <= len(recommendations):
                chosen_rec = recommendations[choice - 1]

                # Get signal feedback
                movie_title = self.catalog_df.loc[chosen_rec['catalog_id']]['title']
                y_signals = get_signal_feedback(movie_title, self.signal_names)

                # Update CTS with positive reward
                self.cts_model.update(
                    c_t=context_features,
                    s_chosen=chosen_rec['signals'],
                    r=1.0,  # Accepted recommendation
                    y_signals=y_signals
                )

                # Track history
                self.recommendation_history.append({
                    'date': test_date,
                    'context': context,
                    'chosen_catalog_id': chosen_rec['catalog_id'],
                    'score': chosen_rec['score'],
                    'reward': 1.0,
                    'signals': chosen_rec['signals'].copy(),
                    'weights': chosen_rec['weights'].copy(),
                    'y_signals': y_signals
                })

                # Update environment memory
                self.env.update_memory(chosen_rec['catalog_id'])

                # Store weight evolution
                mean_w, _ = self.cts_model._compute_w(self.cts_model.U, self.cts_model.b, context_features)
                self.weight_evolution.append({
                    'date': test_date,
                    'weights': mean_w.copy()
                })

                print(f"✓ Recommendation accepted and CTS updated (added to memory)\n")
                return True
            else:
                print("Invalid choice, please try again\n")
                return False
        except ValueError:
            print("Invalid input, please try again\n")
            return False
