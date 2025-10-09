import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import logging
import re

from cts_recommender.adapters.tmdb import tmdb
from cts_recommender import RTS_constants
from cts_recommender.io.readers import read_parquet
from cts_recommender.preprocessing.whatson_extraction import parse_duration_minutes
from cts_recommender.features.tmdb_extract import BASIC_FEATURES



logger = logging.getLogger(__name__)


# Initialize TMDB API client
tmdb_api: tmdb.TMDB_API = tmdb.TMDB_API()

# Load the preprocessed programming file into a DataFrame
def load_processed_programming(data_path: Path) -> pd.DataFrame:
    df = read_parquet(data_path)
    return df

def search_movie_id_row(row: pd.Series, title_column: str = 'title') -> pd.Series:
    """
    Search for the best matching TMDB movie ID for a given row in the programming DataFrame.
    """

    title = row[title_column]
    
    # Remove patterns like '1/2' or '2/2' from the title
    title = re.sub(r'\s*\d+/\d+\s*', ' ', title).strip()

    # Handle duration - programming data has 'duration_min', whatson has 'duration' as string
    if 'duration_min' in row.index:
        known_runtime = row['duration_min']
    elif 'duration' in row.index:
        # Parse HH:MM:SS format from whatson
        known_runtime = parse_duration_minutes(row['duration'])
    else:
        known_runtime = None

    row['tmdb_id'] = tmdb_api.find_best_match(title, known_runtime)
    if row['tmdb_id'] is not None:
        row['processed_title'] = tmdb_api.get_movie_title(row['tmdb_id']) # Keep title from TMDB for consistency throughout all channels
        row['missing_tmdb_id'] = False
    else:
        # Build log message with collection if available
        log_msg = f'No TMDB ID found for title: {title}'
        if 'collection' in row.index and pd.notna(row['collection']):
            log_msg += f' (collection: {row["collection"]})'
        logger.info(log_msg)

        row['processed_title'] = row['title'] # Fallback to original title if no match found
        row['missing_tmdb_id'] = True
    return row


def enrich_movie_feature_row(row):
    """
    Enrich a single row with movie features from TMDB API based on the tmdb_id"""
    tmdb_id = row["tmdb_id"]
    if pd.notna(tmdb_id): # Only proceed if tmdb_id is not NaN
        feats = tmdb_api.get_movie_features(tmdb_id)
        if feats is not None:
            for k, v in feats.items():
                row[k] = v
        else:
            # Set features to default values if API call failed
            for k, (_, _, default) in BASIC_FEATURES.items():
                row[k] = default
    else:
        # Set features to default values if no tmdb_id
        for k, (_, _, default) in BASIC_FEATURES.items():
            row[k] = default

    return row


def enrich_programming_with_movie_metadata(df: pd.DataFrame, extract_movies: bool = True, title_column: str = 'title') -> pd.DataFrame:
    """
    Enrich the programming DataFrame with movie metadata from TMDB API.

    Parameters:
    df (pd.DataFrame): The preprocessed programming DataFrame.
    extract_movies (bool): Whether to select solely the movies from df Dataframe
    title_column (str): Which df column to consider for title to tmdb id search

    Returns:
    pd.DataFrame: The enriched DataFrame with movie metadata.
    """



    # Extract the movies from the programming dataset using the relevant broadcast class key values
    if extract_movies:
        movies_df: pd.DataFrame = df[df['class_key'].isin(RTS_constants.ALL_MOVIE_CLASSKEYS)]
    else:
        # For the case of the whatson catalogue, movie selection alreay applied
        movies_df = df.copy()

    # Certain movies titles are not in the defined column but actually in the description columns, for specific cases retrieve from 'description' column
    # Handle different schemas: programming data has 'description', whatson has 'short_description'
    description_col = 'description' if 'description' in movies_df.columns else 'short_description'
    if description_col in movies_df.columns:
        movies_df.loc[:, 'title'] = np.where(
            movies_df['title'].isin(RTS_constants.RTS_SPECIAL_MOVIE_NAMES),  # condition per-row
            movies_df[description_col],                 # value if True
            movies_df['title']                        # value if False
        )

    ## TMDB API search of movie metadata
    tqdm.pandas(desc="Searching movie IDs...")
    movies_df = movies_df.progress_apply(search_movie_id_row, axis=1, args = (title_column,))

    logger.info(f"{len(movies_df[movies_df['tmdb_id'].isna()])}/{len(movies_df)} TMDB IDs not found ")

    tqdm.pandas(desc="Extracting movie features...")
    movies_df = movies_df.progress_apply(enrich_movie_feature_row, axis=1)

    return movies_df