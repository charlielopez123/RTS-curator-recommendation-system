from typing import List

def compute_genre_diversity(current_genres: List[str] | None, recent_genres: List[List[str]] | None, decay: float = 0.9) -> float:
    """
        Calculate genre diversity score based on overlap frequency and recency.

        Args:
            current_genres: List of genres of the current movie.
            recent_genres: List of genre lists of recently shown movies, ordered from most recent to oldest.
            decay: Exponential decay factor for recency weighting (e.g., 0.9).

        Returns:
            A float in [0, 1], where 1 = completely novel, 0 = heavily repetitive.
    """

    # If no recent movie genres shown or current movie genres shown default to 0.5
    if (not recent_genres) or (not current_genres):
        return 0.5
    
    current_genres_set = set(current_genres)

    genre_penalty = 0.0
    max_penalty = 0.0 # Normalize penalty by maximum possible penalty

    # Iterate over recent movies
    for i, genres in enumerate(recent_genres):
        recency_weight = decay ** i  # More recent = higher weight
        for genre in genres:
            if genre in current_genres_set:
                genre_penalty += recency_weight
        max_penalty += recency_weight * len(current_genres_set)

    # Avoid division by zero
    if max_penalty == 0:
        return 1.0
    
    normalized_penalty = genre_penalty / max_penalty
    return 1.0 - normalized_penalty  # Higher = more diverse

def calculate_language_diversity(
    current_lang: str,
    recent_langs: List[str],
    decay: float = 0.9) -> float:
    """
    Calculate country diversity score with recency-based decay penalty.
    Args:
        current_lang: Language of the current movie.
        recent_langs: List of countries of recently shown movies (most recent first).
        decay: Recency decay factor (e.g., 0.9).
    Returns:
        A float in [0, 1], where 1 = highly diverse (never/long ago shown), 0 = heavily repetitive.
    """
    
    if not recent_langs:
        return 1.0
    
    penalty = 0.0
    max_penalty = 0.0
    for i, language in enumerate(recent_langs):
        
        weight = decay ** (i+1)
        if language == current_lang:
            penalty += weight
        max_penalty += weight
    diversity = 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)
    return diversity


def calculate_temporal_diversity(
    current_release_year: int | None,
    recent_release_years: List[int | None],
    decay: float = 0.9,
    distance_penalty_factor: float = 1.0,
    max_distance: int = 100
    ) -> float:
        """
        Calculates temporal diversity based on recency and distance of movie release years.

        Args:
            current_release_year: Current movie's release year (or None if missing).
            recent_release_years: List of release years of recently shown movies (most recent first).
                                 Can contain None for movies with missing release dates.
            decay: Recency weighting factor (e.g., 0.9).
            distance_penalty_factor: Multiplier for how much to penalize close decades.
        Returns:
            A float in [0, 1] where 1 = highly diverse, 0 = repetitive.
        """
        if not current_release_year or current_release_year is None:
            return 0.5  # Neutral if no release year
        if not recent_release_years:  # No recent eras, fully diverse
            return 1.0
        penalty = 0.0
        max_penalty = 0.0
        for i, recent_release_year in enumerate(recent_release_years):
            # Skip movies with missing release years
            if recent_release_year is None:
                continue
            recency_weight = decay ** i
            # Smaller distance = higher penalty
            distance = abs(current_release_year - recent_release_year)/max_distance
            distance_penalty = max(0, 1 - (distance_penalty_factor * distance))
            # Clamp between 0 and 1
            distance_penalty = min(1.0, max(0.0, distance_penalty))
            penalty += recency_weight * distance_penalty
            max_penalty += recency_weight  # full penalty if same era
        return 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)


def calculate_voting_average_diversity(
    current_vote_average: float | None,
    recent_vote_averages: List[float | None],
    decay: float = 0.9,
    distance_penalty_factor: float = 1.0,
    max_distance: float = 10.0
) -> float:
    """
    Calculate voting average diversity based on recency and distance from recent movies' ratings.

    Args:
        current_vote_average: Current movie's vote average (e.g., 0-10 scale).
        recent_vote_averages: List of vote averages of recently shown movies (most recent first).
        decay: Recency weighting factor (e.g., 0.9).
        distance_penalty_factor: Multiplier for how much to penalize similar ratings.
        max_distance: Maximum expected distance between ratings (default 10.0 for 0-10 scale).

    Returns:
        A float in [0, 1] where 1 = highly diverse, 0 = repetitive.
    """
    if current_vote_average is None:
        return 0.5  # Neutral if no rating
    if not recent_vote_averages:
        return 1.0  # No recent ratings, fully diverse

    penalty = 0.0
    max_penalty = 0.0

    for i, recent_vote in enumerate(recent_vote_averages):
        if recent_vote is None:
            continue

        recency_weight = decay ** i
        # Smaller distance = higher penalty
        distance = abs(current_vote_average - recent_vote) / max_distance
        distance_penalty = max(0, 1 - (distance_penalty_factor * distance))
        # Clamp between 0 and 1
        distance_penalty = min(1.0, max(0.0, distance_penalty))
        penalty += recency_weight * distance_penalty
        max_penalty += recency_weight  # full penalty if same rating

    return 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)


def calculate_revenue_diversity(
    current_revenue: float | None,
    recent_revenues: List[float | None],
    decay: float = 0.9,
    distance_penalty_factor: float = 1.0,
    max_distance: float = 1e9
) -> float:
    """
    Calculate revenue diversity based on recency and distance from recent movies' revenues.

    Args:
        current_revenue: Current movie's revenue (e.g., in USD).
        recent_revenues: List of revenues of recently shown movies (most recent first).
        decay: Recency weighting factor (e.g., 0.9).
        distance_penalty_factor: Multiplier for how much to penalize similar revenue levels.
        max_distance: Maximum expected distance between revenues (default 1e9 for billion-dollar scale).

    Returns:
        A float in [0, 1] where 1 = highly diverse, 0 = repetitive.
    """
    if current_revenue is None:
        return 0.5  # Neutral if no revenue data
    if not recent_revenues:
        return 1.0  # No recent revenues, fully diverse

    penalty = 0.0
    max_penalty = 0.0

    for i, recent_rev in enumerate(recent_revenues):
        if recent_rev is None:
            continue

        recency_weight = decay ** i
        # Smaller distance = higher penalty
        distance = abs(current_revenue - recent_rev) / max_distance
        distance_penalty = max(0, 1 - (distance_penalty_factor * distance))
        # Clamp between 0 and 1
        distance_penalty = min(1.0, max(0.0, distance_penalty))
        penalty += recency_weight * distance_penalty
        max_penalty += recency_weight  # full penalty if same revenue

    return 1.0 - (penalty / max_penalty if max_penalty > 0 else 0)

