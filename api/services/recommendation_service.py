"""
Recommendation Service - Core business logic for generating recommendations

Bridges the API layer with the CTS recommendation system.
Uses TVProgrammingEnvironment to compute signals and CTS model to score candidates.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from cts_recommender.environments.TV_environment import TVProgrammingEnvironment
from cts_recommender.environments.schemas import Context, Season, Channel
from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.utils import dates
from cts_recommender.utils.interactive import compute_signal_vector, SIGNAL_NAMES

from api.schemas.recommendations import (
    RecommendationRequest,
    RecommendationResponse,
    Recommendation
)

logger = logging.getLogger(__name__)


def channel_name_to_enum(channel_name: str) -> Channel:
    """Map channel name string to Channel enum"""
    channel_map = {
        'RTS 1': Channel.RTS1,
        'RTS 2': Channel.RTS2,
        'RTS 1_T_PL': Channel.RTS1,
        'RTS 2_T_PL': Channel.RTS2,
    }
    return channel_map.get(channel_name, Channel.RTS1)


def create_context_from_request(
    request: RecommendationRequest
) -> Tuple[datetime, Context]:
    """
    Create Context object from API request.

    Args:
        request: RecommendationRequest with date, hour, channel

    Returns:
        Tuple of (datetime, Context)
    """
    # Parse date
    date_obj = datetime.strptime(request.date, '%Y-%m-%d')

    # Create Context
    season = dates.get_season(pd.Timestamp(date_obj))
    season_enum = Season(season)

    channel_enum = channel_name_to_enum(request.channel)

    context = Context(
        hour=request.hour,
        day_of_week=date_obj.weekday(),
        month=date_obj.month,
        season=season_enum,
        channel=channel_enum
    )

    return date_obj, context


def compute_signals_for_candidates(
    env: TVProgrammingEnvironment,
    curator_model: Any,
    test_date: pd.Timestamp,
    context: Context,
    context_features: np.ndarray,
    exclude_catalog_ids: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute signal matrix for all available candidates.

    Args:
        env: TVProgrammingEnvironment instance
        curator_model: Curator acceptance probability model
        test_date: Air date
        context: Context object
        context_features: Pre-computed context feature vector
        exclude_catalog_ids: Catalog IDs to exclude (globally excluded + user specified)

    Returns:
        Tuple of (signal_matrix, available_catalog_ids)
    """
    signal_matrix = []
    available_catalog_ids = []

    logger.info(f"Computing signals for {len(env.available_movies)} available movies")

    for catalog_id in tqdm(env.available_movies, desc="Computing signals", unit="movie"):
        # Skip excluded movies (memory + user exclusions)
        if catalog_id in env.memory or catalog_id in exclude_catalog_ids:
            continue

        # Compute signal vector for this candidate
        try:
            signal_vector = compute_signal_vector(
                catalog_id=catalog_id,
                test_date=test_date,
                context=context,
                context_features=context_features,
                env=env,
                curator_model=curator_model
            )
            signal_matrix.append(signal_vector)
            available_catalog_ids.append(catalog_id)
        except Exception as e:
            logger.warning(f"Failed to compute signals for {catalog_id}: {e}")
            continue

    if len(signal_matrix) == 0:
        logger.warning("No candidates with computable signals")
        return np.array([]), []

    signal_matrix = np.array(signal_matrix)
    logger.info(f"Computed signals for {len(available_catalog_ids)} candidates")

    return signal_matrix, available_catalog_ids


def build_recommendation_response(
    request: RecommendationRequest,
    cts_model: ContextualThompsonSampler,
    env: TVProgrammingEnvironment,
    catalog_df: pd.DataFrame,
    topk_indices: np.ndarray,
    topk_scores: np.ndarray,
    sampled_weights: np.ndarray,
    signal_matrix: np.ndarray,
    available_catalog_ids: List[str],
    context_dict: Dict[str, Any],
    total_candidates: int
) -> RecommendationResponse:
    """
    Build API response from CTS recommendation results.

    Args:
        request: Original request
        cts_model: CTS model (for version info)
        env: TVProgrammingEnvironment
        catalog_df: Catalog DataFrame (indexed by catalog_id)
        topk_indices: Indices of top-K candidates
        topk_scores: Scores for top-K candidates
        sampled_weights: CTS-sampled signal weights
        signal_matrix: Full signal matrix
        available_catalog_ids: List of catalog IDs corresponding to signal_matrix
        context_dict: Context metadata
        total_candidates: Total candidates considered

    Returns:
        RecommendationResponse
    """
    recommendations = []

    for idx, score in zip(topk_indices, topk_scores):
        catalog_id = available_catalog_ids[idx]
        signal_values = signal_matrix[idx]

        # Get movie metadata from catalog
        if catalog_id not in catalog_df.index:
            logger.warning(f"Catalog ID {catalog_id} not found in catalog")
            continue

        movie_row = catalog_df.loc[catalog_id]

        # Build signal contribution dict
        signal_contributions = {
            name: float(val) for name, val in zip(SIGNAL_NAMES, signal_values)
        }

        # Build signal weights dict
        signal_weights_dict = {
            name: float(w) for name, w in zip(SIGNAL_NAMES, sampled_weights)
        }

        # Compute weighted signals (element-wise multiply)
        weighted_signals = {
            name: float(signal_values[i] * sampled_weights[i])
            for i, name in enumerate(SIGNAL_NAMES)
        }

        # Extract movie metadata
        recommendation = Recommendation(
            catalog_id=catalog_id,
            title=str(movie_row.get('title', 'Unknown')),
            tmdb_id=int(movie_row['tmdb_id']) if pd.notna(movie_row.get('tmdb_id')) else None,
            genres=movie_row.get('genres') if isinstance(movie_row.get('genres'), list) else None,
            director=str(movie_row['director']) if pd.notna(movie_row.get('director')) else None,
            actors=str(movie_row['actors']) if pd.notna(movie_row.get('actors')) else None,
            production_year=float(movie_row['production_year']) if pd.notna(movie_row.get('production_year')) else None,
            duration_min=float(movie_row['duration_min']) if pd.notna(movie_row.get('duration_min')) else None,
            short_description=str(movie_row['short_description']) if pd.notna(movie_row.get('short_description')) else None,
            original_language=str(movie_row['original_language']) if pd.notna(movie_row.get('original_language')) else None,
            tv_rights_start=movie_row['tv_rights_start'].isoformat() if pd.notna(movie_row.get('tv_rights_start')) else None,
            tv_rights_end=movie_row['tv_rights_end'].isoformat() if pd.notna(movie_row.get('tv_rights_end')) else None,
            vote_average=float(movie_row['vote_average']) if pd.notna(movie_row.get('vote_average')) else None,
            popularity=float(movie_row['popularity']) if pd.notna(movie_row.get('popularity')) else None,
            total_score=float(score),
            signal_contributions=signal_contributions,
            signal_weights=signal_weights_dict,
            weighted_signals=weighted_signals,
            predicted_audience=signal_contributions.get('audience')  # Audience signal is the prediction
        )

        recommendations.append(recommendation)

    # Build response
    response = RecommendationResponse(
        recommendations=recommendations,
        context=context_dict,
        model_version="v1.0.0",  # TODO: Get from model metadata
        candidates_considered=total_candidates,
        timestamp=datetime.now()
    )

    return response


def generate_recommendations(
    request: RecommendationRequest,
    env: TVProgrammingEnvironment,
    cts_model: ContextualThompsonSampler,
    curator_model: Any,
    catalog_df: pd.DataFrame
) -> RecommendationResponse:
    """
    Main entry point for generating recommendations.

    Args:
        request: RecommendationRequest from API
        env: TVProgrammingEnvironment instance
        cts_model: Contextual Thompson Sampler model
        curator_model: Curator acceptance probability model
        catalog_df: Catalog DataFrame

    Returns:
        RecommendationResponse

    Raises:
        ValueError: If no candidates available or invalid request
    """
    logger.info(f"Generating recommendations for {request.date} @ {request.hour}:00 on {request.channel}")

    # Create context
    date_obj, context = create_context_from_request(request)
    test_date = pd.Timestamp(date_obj)

    # Get available movies for this date
    env.get_available_movies(test_date)

    if len(env.available_movies) == 0:
        raise ValueError(f"No movies available for {request.date}")

    logger.info(f"Found {len(env.available_movies)} available movies")

    # Get context features
    context_features, _ = env.get_context_features(context)

    # Compute signals for all candidates
    signal_matrix, available_catalog_ids = compute_signals_for_candidates(
        env=env,
        curator_model=curator_model,
        test_date=test_date,
        context=context,
        context_features=context_features,
        exclude_catalog_ids=request.exclude_movie_ids
    )

    if len(available_catalog_ids) == 0:
        raise ValueError("No candidates available after filtering")

    # Get top-K recommendations from CTS
    K = min(request.top_n, len(available_catalog_ids))
    topk_indices, topk_scores, sampled_weights, _ = cts_model.score_candidates(
        c_t=context_features,
        S_matrix=signal_matrix,
        K=K
    )

    logger.info(f"CTS selected top {K} recommendations")

    # Build context metadata
    context_dict = {
        "date": request.date,
        "hour": request.hour,
        "channel": request.channel,
        "weekday": date_obj.weekday(),
        "is_weekend": date_obj.weekday() >= 5,
        "season": context.season.value
    }

    # Build response
    response = build_recommendation_response(
        request=request,
        cts_model=cts_model,
        env=env,
        catalog_df=catalog_df,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        sampled_weights=sampled_weights,
        signal_matrix=signal_matrix,
        available_catalog_ids=available_catalog_ids,
        context_dict=context_dict,
        total_candidates=len(env.available_movies)
    )

    return response
