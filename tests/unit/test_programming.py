"""Unit tests for programming data loading and preprocessing."""
import pytest
import pandas as pd
from unittest.mock import patch
from pathlib import Path

from cts_recommender.preprocessing.programming import load_original_programming
from cts_recommender.features.TV_programming_Rdata_schema import (
    ORIGINAL_PROGRAMMING_COLUMNS,
    PROGRAMMING_RENAME_MAP,
)


class TestLoadOriginalProgramming:
    """Test suite for load_original_programming function."""

    def test_sanity_check_passes_with_all_columns(self):
        """Test that loading succeeds when all expected columns are present."""
        # Create a mock dataframe with all expected columns
        mock_df = pd.DataFrame(columns=list(ORIGINAL_PROGRAMMING_COLUMNS))

        # Mock the read_rdata function to return our test dataframe
        with patch('cts_recommender.preprocessing.programming.read_rdata', return_value=mock_df):
            result = load_original_programming(Path("dummy.RData"))

            # Verify result has renamed columns in correct order
            expected_columns = [PROGRAMMING_RENAME_MAP[c] for c in ORIGINAL_PROGRAMMING_COLUMNS]
            assert list(result.columns) == expected_columns

    def test_sanity_check_fails_with_missing_column(self):
        """Test that KeyError is raised when expected columns are missing."""
        # Create a mock dataframe missing the "Title" column
        columns = [c for c in ORIGINAL_PROGRAMMING_COLUMNS if c != "Title"]
        mock_df = pd.DataFrame(columns=columns)

        with patch('cts_recommender.preprocessing.programming.read_rdata', return_value=mock_df):
            with pytest.raises(KeyError, match="Missing expected RData columns.*Title"):
                load_original_programming(Path("dummy.RData"))

    def test_sanity_check_fails_with_multiple_missing_columns(self):
        """Test that KeyError lists all missing columns."""
        # Create a mock dataframe missing multiple columns
        columns = [c for c in ORIGINAL_PROGRAMMING_COLUMNS if c not in ["Title", "Channel", "Duration"]]
        mock_df = pd.DataFrame(columns=columns)

        with patch('cts_recommender.preprocessing.programming.read_rdata', return_value=mock_df):
            with pytest.raises(KeyError) as exc_info:
                load_original_programming(Path("dummy.RData"))

            # Verify all missing columns are mentioned
            error_message = str(exc_info.value)
            assert "Channel" in error_message
            assert "Duration" in error_message
            assert "Title" in error_message

    def test_sanity_check_with_extra_columns(self):
        """Test that loading succeeds with extra unexpected columns."""
        # Create a mock dataframe with all expected columns plus extras
        columns = list(ORIGINAL_PROGRAMMING_COLUMNS) + ["Extra1", "Extra2"]
        mock_df = pd.DataFrame(columns=columns)

        with patch('cts_recommender.preprocessing.programming.read_rdata', return_value=mock_df):
            result = load_original_programming(Path("dummy.RData"))

            # Should succeed and only include expected columns (renamed)
            expected_columns = [PROGRAMMING_RENAME_MAP[c] for c in ORIGINAL_PROGRAMMING_COLUMNS]
            assert list(result.columns) == expected_columns
            assert "Extra1" not in result.columns
            assert "Extra2" not in result.columns

    def test_column_renaming(self):
        """Test that columns are renamed correctly."""
        # Create a mock dataframe with sample data
        mock_df = pd.DataFrame({
            col: [f"value_{i}"] for i, col in enumerate(ORIGINAL_PROGRAMMING_COLUMNS)
        })

        with patch('cts_recommender.preprocessing.programming.read_rdata', return_value=mock_df):
            result = load_original_programming(Path("dummy.RData"))

            # Check specific renames
            assert "date" in result.columns
            assert "title" in result.columns
            assert "class_key" in result.columns
            assert "Date" not in result.columns
            assert "Title" not in result.columns
            assert "BrdCstClassKey" not in result.columns

    def test_column_ordering_preserved(self):
        """Test that column order matches ORIGINAL_PROGRAMMING_COLUMNS order."""
        # Create a mock dataframe with shuffled columns
        shuffled_columns = list(reversed(ORIGINAL_PROGRAMMING_COLUMNS))
        mock_df = pd.DataFrame(columns=shuffled_columns)

        with patch('cts_recommender.preprocessing.programming.read_rdata', return_value=mock_df):
            result = load_original_programming(Path("dummy.RData"))

            # Result should have columns in the original order (renamed)
            expected_columns = [PROGRAMMING_RENAME_MAP[c] for c in ORIGINAL_PROGRAMMING_COLUMNS]
            assert list(result.columns) == expected_columns


class TestSchemaIntegrity:
    """Tests to ensure schema changes are tracked and handled properly."""

    def test_rename_map_covers_all_original_columns(self):
        """Ensure PROGRAMMING_RENAME_MAP has entries for all ORIGINAL_PROGRAMMING_COLUMNS."""
        for col in ORIGINAL_PROGRAMMING_COLUMNS:
            assert col in PROGRAMMING_RENAME_MAP, (
                f"Column '{col}' in ORIGINAL_PROGRAMMING_COLUMNS is missing from PROGRAMMING_RENAME_MAP. "
                f"If you've added a new column, update the rename map accordingly."
            )

    def test_rename_map_no_orphaned_entries(self):
        """Ensure PROGRAMMING_RENAME_MAP doesn't have extra entries not in ORIGINAL_PROGRAMMING_COLUMNS."""
        for col in PROGRAMMING_RENAME_MAP.keys():
            assert col in ORIGINAL_PROGRAMMING_COLUMNS, (
                f"Column '{col}' in PROGRAMMING_RENAME_MAP is not in ORIGINAL_PROGRAMMING_COLUMNS. "
                f"If you've removed a column, clean up the rename map."
            )

    def test_no_duplicate_renamed_columns(self):
        """Ensure no two original columns map to the same renamed column."""
        renamed_values = list(PROGRAMMING_RENAME_MAP.values())
        duplicates = [x for x in renamed_values if renamed_values.count(x) > 1]
        assert len(duplicates) == 0, (
            f"Duplicate renamed columns found: {set(duplicates)}. "
            f"Each original column must map to a unique renamed column."
        )

    def test_expected_column_count(self):
        """Test that we have exactly 13 columns as per the original schema."""
        assert len(ORIGINAL_PROGRAMMING_COLUMNS) == 13, (
            f"Expected 13 columns in ORIGINAL_PROGRAMMING_COLUMNS, got {len(ORIGINAL_PROGRAMMING_COLUMNS)}. "
            f"If you've added/removed columns, update this test and verify downstream impacts."
        )
        assert len(PROGRAMMING_RENAME_MAP) == 13, (
            f"Expected 13 entries in PROGRAMMING_RENAME_MAP, got {len(PROGRAMMING_RENAME_MAP)}. "
            f"Ensure the rename map is in sync with ORIGINAL_PROGRAMMING_COLUMNS."
        )
