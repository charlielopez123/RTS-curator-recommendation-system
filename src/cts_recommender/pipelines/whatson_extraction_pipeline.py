from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import logging

from cts_recommender.preprocessing import whatson_extraction, movie_features, catalog_features
from cts_recommender.io.readers import read_parquet
from cts_recommender.io.writers import atomic_write_parquet
from cts_recommender.settings import get_settings
from cts_recommender.features.whatson_schema import ORIGINAL_WHATSON_COLUMNS, WHATSON_RENAME_MAP

cfg = get_settings()

logger = logging.getLogger(__name__)



def run_whatson_extraction_pipeline(raw_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:
    if raw_file is None:
        raw_file = cfg.original_rdata_programming
    if out_file is None:
        out_file = cfg.processed_dir / "whatson_catalogue.parquet"

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


def run_enrich_catalog_with_movie_metadata_pipeline(catalog_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """
    Enrich WhatsOn movie catalog with TMDB metadata.

    Parameters:
    catalog_file (Path): Path to the processed movie catalog parquet file
    out_file (Path): Path to save the enriched catalog

    Returns:
    tuple[pd.DataFrame, Path]: The enriched DataFrame and output file path
    """
    if catalog_file is None:
        catalog_file = cfg.processed_dir / "whatson_catalogue.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "whatson_catalogue_enriched_tmdb.parquet"

    whatson_catalogue_df = whatson_extraction.load_whatson_movie_catalogue(catalog_file)
    df_enriched = movie_features.enrich_programming_with_movie_metadata(whatson_catalogue_df, extract_movies=False, title_column='best_title')

    atomic_write_parquet(df_enriched, out_file)

    return df_enriched, out_file


def run_whatson_feature_processing_pipeline(enriched_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """
    Build catalog features for WhatsOn movie catalog.

    Parameters:
    enriched_file (Path): Path to the TMDB-enriched catalog parquet file
    out_file (Path): Path to save the catalog feature dataset

    Returns:
    tuple[pd.DataFrame, Path]: The catalog feature DataFrame and output file path
    """
    if enriched_file is None:
        enriched_file = cfg.processed_dir / "whatson_catalogue_enriched_tmdb.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "whatson_catalogue_enriched_tmdb.parquet"

    enriched_df = read_parquet(enriched_file)
    catalog_features_df = catalog_features.build_catalog_features(enriched_df)

    atomic_write_parquet(catalog_features_df, out_file)

    return catalog_features_df, out_file


def run_whatson_catalog_creation_pipeline(in_file: Path | None = None, out_file: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """
    Create a deduplicated WhatsOn movie catalog with catalog IDs.

    Parameters:
    in_file (Path): Path to the TMDB-enriched and additional catalog features parquet file
    out_file (Path): Path to save the catalog

    Returns:
    tuple[pd.DataFrame, Path]: The catalog DataFrame and output file path
    """
    if in_file is None:
        in_file = cfg.processed_dir / "whatson/whatson_catalogue_enriched_tmdb.parquet"
    if out_file is None:
        out_file = cfg.processed_dir / "whatson/whatson_catalogue.parquet"

    logger.info(f"Loading enriched WhatsOn data from {in_file}")
    df_enriched = read_parquet(in_file)
    logger.info(f"Loaded {len(df_enriched)} enriched movie records")

    logger.info("Creating catalog with deduplication and catalog IDs...")
    catalog_df = whatson_extraction.create_whatson_catalog(df_enriched)

    logger.info(f"Catalog created: {len(catalog_df)} unique movies")
    logger.info(f"TMDB IDs: {(~catalog_df.index.str.startswith('WO_')).sum()}")
    logger.info(f"Custom IDs: {catalog_df.index.str.startswith('WO_').sum()}")

    atomic_write_parquet(catalog_df, out_file)
    logger.info(f"Catalog saved to {out_file}")

    return catalog_df, out_file