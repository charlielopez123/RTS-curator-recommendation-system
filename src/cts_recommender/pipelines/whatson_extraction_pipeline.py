from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import logging

from cts_recommender.preprocessing import whatson_extraction
from cts_recommender.io.writers import atomic_write_parquet
from cts_recommender.settings import get_settings
from cts_recommender.features.whatson_csv_schema import ORIGINAL_WHATSON_COLUMNS, WHATSON_RENAME_MAP

cfg = get_settings()

logger = logging.getLogger(__name__)

def run_whatson_extration_pipeline(raw_file: Path | None = None, out_file: Path |None = None) -> tuple[pd.DataFrame, Path]:
    if raw_file is None:
        raw_file = cfg.original_rdata_programming
    if out_file is None:
        out_file = cfg.processed_dir / "whatson.parquet"

    whatson_df = whatson_extraction.load_whatson_catalog_csv(raw_file)
    logger.info(f"Loaded {len(whatson_df)} records")

    assert whatson_df.columns.tolist() == ORIGINAL_WHATSON_COLUMNS, "Column mismatch!"
    logger.info("✓ Columns validated")

    # Rename columns immediately
    whatson_df = whatson_df.rename(columns=WHATSON_RENAME_MAP)
    logger.info("✓ Columns renamed")

    # Create a copy for processing
    processing_df = whatson_df.copy()

    # Pre-compute title counts for Rule 3
    title_counts = whatson_df['title'].value_counts()

    # Pre-compute collection counts for Rule 4
    collection_counts = whatson_df['collection'].value_counts()

    # Apply filter, keeps only the rows considered as movies
    tqdm.pandas(desc="Selecting movies from Whats'On catalogue...")
    movies_only_df = processing_df[processing_df.progress_apply(whatson_extraction.should_keep_as_movie, axis=1, args=(title_counts, collection_counts))].copy()

    # Apply the title selection
    processed_movies_only_df = movies_only_df.copy()
    tqdm.pandas(desc="Finding the best titles available...")
    processed_movies_only_df['best_title'] = movies_only_df.apply(whatson_extraction.select_best_title, axis=1)

    logger.info(f"Total records: {len(whatson_df)}")
    logger.info(f"Movies kept: {len(processed_movies_only_df)}")
    logger.info(f"Filtered out: {len(whatson_df) - len(processed_movies_only_df)} ({100*(len(whatson_df) - len(processed_movies_only_df))/len(whatson_df):.1f}%)")

    logger.info(f"\nContent codes distribution:")
    logger.info(processed_movies_only_df['class_key'].value_counts())
    logger.info(f"\nCollection value counts (top 20):")
    logger.info(processed_movies_only_df['collection'].value_counts().head(20))

    # Show statistics about filtering
    logger.info(f"\n=== FILTERING BREAKDOWN ===")
    logger.info(f"Titles with dates: {whatson_df['title'].apply(whatson_extraction.has_date_pattern).sum()}")
    logger.info(f"Titles appearing >3 times: {sum(1 for t in whatson_df['title'] if title_counts.get(t, 0) > 3)}")
    logger.info(f"Collections appearing >2 times without film keywords: {sum(1 for c, count in collection_counts.items() if count > 2 and not whatson_extraction.is_film_collection(c))}")
    logger.info(f"Titles == Original titles: {sum(1 for _, row in whatson_df.iterrows() if pd.notna(row['title']) and pd.notna(row['original_title']) and str(row['title']).strip() == str(row['original_title']).strip())}")
    
    atomic_write_parquet(processed_movies_only_df, out_file)

    return processed_movies_only_df, out_file