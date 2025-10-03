import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import logging

from cts_recommender.adapters.tmdb import tmdb
from cts_recommender import RTS_constants
from cts_recommender.io.readers import read_parquet



logger = logging.getLogger(__name__)


# Initialize TMDB API client
tmdb_api: tmdb.TMDB_API = tmdb.TMDB_API()

# Load the preprocessed programming file into a DataFrame
def load_processed_programming(data_path: Path) -> pd.DataFrame:
    df = read_parquet(data_path)
    return df

def search_movie_id_row(row: pd.Series) -> pd.Series:
    """
    Search for the best matching TMDB movie ID for a given row in the programming DataFrame.
    """

    title = row['title']
    known_runtime = row['duration_min']
    row['tmdb_id'] = tmdb_api.find_best_match(title, known_runtime)
    if row['tmdb_id'] is not None:
        row['processed_title'] = tmdb_api.get_movie_title(row['tmdb_id']) # Keep title from TMDB for consistency throughout all channels
        row['missing_tmdb_id'] = False
    else:
        logger.info(f'No TMDB ID found for title: {title}')
        row['processed_title'] = row['title'] # Fallback to original title if no match found
        row['missing_tmdb_id'] = True
    return row


def enrich_movie_feature_row(row):
    """
    Enrich a single row with movie features from TMDB API based on the tmdb_id"""
    tmdb_id = row["tmdb_id"]
    if pd.notna(tmdb_id): # Only proceed if tmdb_id is not NaN
        feats = tmdb_api.get_movie_features(tmdb_id)
        for k, v in feats.items():
            row[k] = v

    return row


def enrich_programming_with_movie_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the programming DataFrame with movie metadata from TMDB API.

    Parameters:
    df (pd.DataFrame): The preprocessed programming DataFrame.

    Returns:
    pd.DataFrame: The enriched DataFrame with movie metadata.
    """



    # Extract the movies from the programming dataset using the relevant broadcast class key values
    movies_df: pd.DataFrame = df[df['class_key'].isin(RTS_constants.ALL_MOVIE_CLASSKEYS)]

    # Certain movies titles are not in the defined column but actually in the description columns, for specific cases retrieve from 'description' column
    movies_df.loc[:, 'title'] = np.where(
        movies_df['title'].isin(RTS_constants.RTS_SPECIAL_MOVIE_NAMES),  # condition per-row
        movies_df['description'],                 # value if True
        movies_df['title']                        # value if False
    )

    ## TMDB API search of movie metadata
    tqdm.pandas(desc="Searching movie IDs...")
    movies_df = movies_df.progress_apply(search_movie_id_row, axis=1)

    logger.info(f"{len(movies_df[movies_df['tmdb_id'].isna()])}/{len(movies_df)} TMDB IDs not found ")

    tqdm.pandas(desc="Extracting movie features...")
    movies_df = movies_df.progress_apply(enrich_movie_feature_row, axis=1)

    return movies_df