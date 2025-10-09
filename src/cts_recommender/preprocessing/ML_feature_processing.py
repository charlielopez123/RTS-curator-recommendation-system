from pathlib import Path
import pandas as pd
import logging
from cts_recommender import RTS_constants
from cts_recommender.features.TV_programming_Rdata_schema import ML_FEATURES
from cts_recommender.io.readers import read_parquet
from cts_recommender.preprocessing import feature_transformations

logger = logging.getLogger(__name__)

# Load the preprocessed programming file into a DataFrame
def load_processed_and_enriched_programming(data_path_processed: Path, data_path_enriched: Path) -> pd.DataFrame:
    processed_df = read_parquet(data_path_processed)
    enriched_df = read_parquet(data_path_enriched)
    return processed_df, enriched_df


# Import shared transformation functions
process_movie_tmdb_features = feature_transformations.process_tmdb_features
add_genre_features = feature_transformations.add_genre_features
add_movie_age_feature = feature_transformations.add_movie_age_feature


def finalize_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final processing: convert booleans to int and validate no NaNs remain.

    Parameters:
    df (pd.DataFrame): DataFrame with all features

    Returns:
    pd.DataFrame: DataFrame ready for ML
    """
    # Convert booleans using shared utility
    df = feature_transformations.convert_bool_to_int(df)

    # Check for any remaining NaNs
    num_nans = df.isna().sum().sum()
    if num_nans > 0:
        logger.warning(f"DataFrame contains {num_nans} NaN values after processing. Please check the data.")
        logger.warning(df.isna().sum()[df.isna().sum() > 0])

    return df


def build_X_features_movies_only(df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML features for movies-only datasets (e.g., WhatsOn catalog).
    Use this when you don't have non-movie content to merge.

    Parameters:
    df_enriched (pd.DataFrame): The enriched movie features DataFrame with TMDB data.

    Returns:
    pd.DataFrame: The feature-ready DataFrame for movies.
    """
    # Process TMDB features
    df = process_movie_tmdb_features(df_enriched, is_movie=True)

    # Select columns needed for processing
    # Note: We don't filter by ML_FEATURES here since we need genres and release_date
    # for processing, and they'll be transformed/dropped later

    # Add genre one-hot encoding
    df = add_genre_features(df)

    # Add movie age
    df = add_movie_age_feature(df)

    # Drop columns that are not ML features (e.g., tmdb_id used for processing)
    cols_to_drop = [col for col in ['tmdb_id'] if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Final cleanup
    df = finalize_ml_features(df)

    return df


def build_X_features(processed_df: pd.DataFrame, df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set X by merging the programming DataFrame with the enriched movie features DataFrame.
    Use this for programming data that contains both movies and non-movies.

    Parameters:
    processed_df (pd.DataFrame): The preprocessed programming DataFrame (includes non-movies).
    df_enriched (pd.DataFrame): The enriched movie features DataFrame.

    Returns:
    pd.DataFrame: The feature set X combining both movies and non-movies.
    """

    # Remove movies from processed_df (they're in df_enriched)
    nonmovie_mask = ~processed_df['class_key'].isin(RTS_constants.ALL_MOVIE_CLASSKEYS)
    nonmovie_df = processed_df.loc[nonmovie_mask].copy()

    # Process TMDB features for both movies and non-movies
    nonmovie_df = process_movie_tmdb_features(nonmovie_df, is_movie=False)
    df_enriched = process_movie_tmdb_features(df_enriched, is_movie=True)

    # Select only ML features that exist in both datasets plus processing columns
    # This ensures consistent column ordering for sklearn models
    available_features = [col for col in ML_FEATURES if col in nonmovie_df.columns and col in df_enriched.columns]
    # Add columns needed for processing (will be processed/dropped later)
    processing_features = available_features + ['genres', 'release_date']

    nonmovie_df = nonmovie_df[processing_features]
    df_enriched = df_enriched[processing_features]

    # Combine movies and non-movies
    full_df = pd.concat([nonmovie_df, df_enriched])

    # Add genre one-hot encoding
    full_df = add_genre_features(full_df)

    # Add movie age
    full_df = add_movie_age_feature(full_df)

    # Final cleanup
    full_df = finalize_ml_features(full_df)

    return full_df