from pathlib import Path
import pandas as pd
import numpy as np
import logging
from cts_recommender import RTS_constants
from cts_recommender.features.schemas import ML_FEATURES

logger = logging.getLogger(__name__)

# Load the preprocessed programming file into a DataFrame
def load_processed_and_enriched_programming(data_path_processed: Path, data_path_enriched: Path) -> pd.DataFrame:
    processed_df = pd.read_parquet(data_path_processed)
    enriched_df = pd.read_parquet(data_path_enriched)
    return processed_df, enriched_df


def build_X_features(processed_df: pd.DataFrame, df_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature set X by merging the programming DataFrame with the enriched movie features DataFrame
    
    Parameters:
    processed_df (pd.DataFrame): The preprocessed programming DataFrame.
    df_enriched (pd.DataFrame): The enriched movie features DataFrame.
    
    Returns:
    pd.DataFrame: The feature set X.
"""
    
    #Remove movies from processed_df
    # --- make explicit copies before mutating to avoid SettingWithCopyWarning ---
    nonmovie_mask = ~processed_df['class_key'].isin(RTS_constants.ALL_MOVIE_CLASSKEYS)
    nonmovie_df = processed_df.loc[nonmovie_mask].copy()   # <— explicit copy
    df_enriched = df_enriched.copy()                       # <— safe to mutate


    # Feature preprocessing for handling missing values
    logger.info(nonmovie_df.columns)
    logger.info(df_enriched.columns)
    nonmovie_df.loc[:, 'adult'] = False
    df_enriched['adult'] = np.where(
        df_enriched['adult'].isna(),
        False,
        df_enriched['adult']
    )
    nonmovie_df.loc[:, 'original_language'] = 'unknown'
    df_enriched['original_language'] = np.where(
        df_enriched['original_language'].isna(),
        'unknown',
        df_enriched['original_language']
    )

    # Genres
    # Add empty genres list for non-movies
    nonmovie_df.loc[:, 'genres'] = [[] for _ in range(len(nonmovie_df))]
    df_enriched['genres'] = df_enriched['genres'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Release date
    nonmovie_df.loc[:, 'release_date'] = '1900-01-01'
    nonmovie_df.loc[:,'missing_release_date']= True

    df_enriched['release_date'] = np.where(
        df_enriched['release_date'].isna(),
        '1900-01-01',
        df_enriched['release_date']
    )
    df_enriched.loc[:, 'missing_release_date'] = np.where(
        df_enriched['release_date'].isna(),
        False,
        True
    )
    df_enriched.loc[:, 'missing_release_date'] = df_enriched.loc[:, 'missing_release_date'].apply(lambda s: False if s == '' else True)
    df_enriched.loc[:, 'release_date'] = df_enriched.loc[:, 'release_date'].apply(lambda s: '1900-01-01' if s == '' else s)

    # Missing Revenue put as 0
    nonmovie_df.loc[:, 'revenue'] = 0
    df_enriched['revenue'] = np.where(
        df_enriched['revenue'].isna(),
        0,
        df_enriched['revenue']
    )

    # missing tmdb id feature
    nonmovie_df.loc[:, 'missing_tmdb']=True
    df_enriched.loc[:, 'missing_tmdb'] = np.where(
        df_enriched['tmdb_id'].isna(),
        True,
        False
    )

    # Add vote average as median or zero
    df_enriched.loc[:, 'vote_average'] = df_enriched['vote_average'].fillna(0)
    nonmovie_df.loc[:, 'vote_average'] = 0


    #  Add popularity as median or zero 
    df_enriched.loc[:, 'popularity'] = df_enriched['popularity'].fillna(0)
    nonmovie_df.loc[:, 'popularity'] = 0

    # Separate Movies and TV Shows
    nonmovie_df.loc[:, 'is_movie'] = False
    df_enriched.loc[:, 'is_movie'] = True

    # Select only ML features in the correct order for both DataFrames
    # This ensures consistent column ordering for sklearn models
    available_features = [col for col in ML_FEATURES if col in nonmovie_df.columns and col in df_enriched.columns]

    nonmovie_df = nonmovie_df[available_features]
    df_enriched = df_enriched[available_features]

    # full_df includes all movies and non-movies
    full_df = pd.concat([nonmovie_df, df_enriched])

    return full_df