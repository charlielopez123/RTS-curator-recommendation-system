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
    nonmovie_df.loc[:, 'adult'] = False
    df_enriched['adult'] = np.where(
        df_enriched['adult'].isna(),
        False,
        df_enriched['adult']
    ).astype(bool)
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
        lambda x: x.tolist() if isinstance(x, np.ndarray) else (x if isinstance(x, list) else [])
    )

    # Release date
    nonmovie_df.loc[:, 'release_date'] = '1900-01-01'
    nonmovie_df.loc[:,'missing_release_date']= True

    df_enriched['release_date'] = np.where(
        df_enriched['release_date'].isna(),
        '1900-01-01',
        df_enriched['release_date']
    )
    df_enriched.loc[:, 'missing_release_date'] = (
        df_enriched['release_date'].isna() | (df_enriched['release_date'] == '')
    )
    df_enriched.loc[:, 'release_date'] = df_enriched['release_date'].fillna('1900-01-01').replace('', '1900-01-01')

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

    # Select only ML features that exist in both datasets plus processing columns
    # This ensures consistent column ordering for sklearn models
    available_features = [col for col in ML_FEATURES if col in nonmovie_df.columns and col in df_enriched.columns]
    # Add columns needed for processing (will be processed/dropped later)
    processing_features = available_features + ['genres', 'release_date']

    nonmovie_df = nonmovie_df[processing_features]  # nonmovie_df now has all processing features
    df_enriched = df_enriched[processing_features]  # enriched_df has all features

    # full_df includes all movies and non-movies
    full_df = pd.concat([nonmovie_df, df_enriched])

    # One-hot encoding for categorical features
    full_df.loc[:, 'genre_names'] = full_df['genres'].apply(lambda lst: [d['name'] for d in lst])
    exploded = full_df.explode(['genre_names'])
    dummies = pd.get_dummies(
        exploded["genre_names"],
        prefix="genre"
    )
    genre_dummies = dummies.groupby(level=0).sum()
    full_df = pd.concat([full_df, genre_dummies], axis=1)

    # Drop the intermediate genre columns - we now have the one-hot encoded versions
    full_df = full_df.drop(columns=['genres', "genre_names"])

    full_df["release_date_dt"] = pd.to_datetime(
        full_df["release_date"],
        format="%Y-%m-%d",
    )

    full_df['movie_age'] = pd.Timestamp('today').year - pd.to_datetime(full_df['release_date'], format='%Y-%m-%d').dt.year
    # Drop intermediate columns not needed for modeling
    full_df = full_df.drop(columns=['release_date', 'release_date_dt'])

    # Convert all bool columns → int (0/1)
    bool_cols = full_df.select_dtypes(include='bool').columns
    full_df[bool_cols] = full_df[bool_cols].astype(int)

    # Check for any remaining NaNs
    num_nans = full_df.isna().sum().sum()
    assert num_nans == 0, f"DataFrame contains {num_nans} NaN values after processing."
    if num_nans > 0:
        logger.warning(f"DataFrame contains {num_nans} NaN values after processing. Please check the data.")
        logger.warning(full_df.isna().sum())

    return full_df