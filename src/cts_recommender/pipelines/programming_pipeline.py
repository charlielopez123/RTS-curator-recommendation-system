from pathlib import Path
from cts_recommender.preprocessing.programming import *
from cts_recommender.io.writers import atomic_write_parquet
from cts_recommender.settings import get_settings

cfg = get_settings()

def run_original_Rdata_programming_pipeline( raw_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:

    if raw_file is None:
        raw_file = cfg.original_rdata_programming
    holidays_file = cfg.reference_dir / "holidays.json"
    if out_file is None:
        out_file = cfg.processed_dir / "preprocessed_programming.parquet"
    df_raw = load_original_programming(raw_file)
    df = preprocess_programming(df_raw)

    atomic_write_parquet(df, out_file)
    return df, out_file


