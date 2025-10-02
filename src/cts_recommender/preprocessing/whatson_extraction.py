from pathlib import Path
import pandas as pd

from cts_recommender.io import readers
from cts_recommender.settings import get_settings


def load_whatson_catalog_csv(csv_path: Path) -> pd.DataFrame: 
    whatson_df = readers.read_csv(csv_path, sep='\t')
    return whatson_df

