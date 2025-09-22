import pyreadr
from pathlib import Path
import pandas as pd
import json
from cts_recommender.io.readers import read_rdata, read_json
from cts_recommender.preprocessing import dates
from cts_recommender.settings import get_settings

cfg = get_settings()

# Load JSON file containing holidays
holidays = read_json(cfg.reference_dir / "holidays.json")
# Build the holiday index for fast lookup
index = dates.build_holiday_index(holidays)

# Opening the original R data file into a dataframe
def load_original_programming(rdata_path: Path, key: str = 'mydata') -> pd.DataFrame:
    df = read_rdata(rdata_path, key)

    # Rename columns of df
    df.columns = ['date','title','description','channel','start_time','end_time',
                'duration','net_duration','class_key','target_audience','activity',
                'rt_m','pdm']
    return df

def preprocess_programming(df: pd.DataFrame) -> pd.DataFrame:

    # Preprocessing
    df = df[df['activity'] == 'Overnight+7']  # Keep only the Overnight+7
    df = df[df['target_audience'] == 'Personnes 3+'] # Keep only for target audience Personnes 3+, which includes all ages

    # Remove superfluous columns
    df = df.drop(columns=['net_duration', 'end_time'])

    # Keep column names in a list for further use
    column_names = df.columns.tolist()

    # Convert dates and timestamps into datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Convert durations to timedelta
    df['duration'] = pd.to_timedelta(df['duration'])

    # Convert target features into numeric values
    df['rt_m'] = pd.to_numeric(df['rt_m'])
    df['pdm'] = pd.to_numeric(df['pdm'])


    ## Feature Engineering

    df['hour'] = df['start_time'].str.split(':').str[0].astype(int) # Extract the starting hour from start_time
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'] >= 5
    df['duration_min'] = df['duration'].dt.total_seconds() / 60


    # Retrieve season for each date
    df['season'] = df['date'].apply(dates.get_season)

    # Apply the function to the 'date' column
    df['public_holiday'] = df['date'].apply(lambda x: dates.is_holiday_indexed(x=x, index=index))

    # Remove columns that are not needed anymore
    df = df.drop(columns = ['target_audience', 'activity', 'duration'])

    return df

