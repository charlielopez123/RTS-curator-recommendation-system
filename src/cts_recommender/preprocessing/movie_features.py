import pandas as pd
from data.reference import RTS_constants
import numpy as np

# Load the preprocessed programmming file into a DataFrame
df = pd.read_csv('data/preprocessed_programming_dataframe.parquet')

# Extract the movies from the programming dataset
relevant_keys = RTS_constants.movies_BrdCstClassKey + RTS_constants.competitor_movie_codes
movies_df = df[df['BrdCstClassKey'].isin(relevant_keys)]

# Certain movies titles are not in the defined column but actually in the description columns, for specific cases retrieve from 'description' column
movies_df.loc[:, 'title'] = np.where(
    movies_df['title'].isin(RTS_constants.special_movie_names),  # condition per-row
    movies_df['description'],                 # value if True
    movies_df['title']                        # value if False
)
movies_df = movies_df.drop(columns=['description']) # description column can now be removed

## TMDB API search of movie metadata
