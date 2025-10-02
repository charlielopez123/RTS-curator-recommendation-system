from pathlib import Path
import pandas as pd

from cts_recommender.preprocessing import whatson_extraction
from cts_recommender.io.writers import atomic_write_parquet
from cts_recommender.settings import get_settings

cfg = get_settings()

def run_whatson_extration_pipeline(raw_file: Path | None = None, out_file: Path |None = None) -> tuple[pd.DataFrame, Path]:
    if raw_file is None:
        raw_file = cfg.original_rdata_programming
    if out_file is None:
        out_file = cfg.processed_dir / "whatson.parquet"

    whatson_df = whatson_extraction.load_whatson_catalog_csv(raw_file)
    return whatson_df, out_file