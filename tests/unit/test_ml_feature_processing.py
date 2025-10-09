"""Unit tests for ML feature processing functions."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from cts_recommender.preprocessing.ML_feature_processing import (
    process_movie_tmdb_features,
    add_genre_features,
    add_movie_age_feature,
    finalize_ml_features,
    build_X_features_movies_only,
)


class TestProcessMovieTmdbFeatures:
    """Tests for process_movie_tmdb_features function."""

    def test_movie_with_complete_tmdb_data(self):
        """Test processing movie with all TMDB fields populated."""
        df = pd.DataFrame({
            'adult': [False],
            'original_language': ['en'],
            'genres': [[{'id': 28, 'name': 'Action'}, {'id': 12, 'name': 'Adventure'}]],
            'release_date': ['2020-05-15'],
            'revenue': [1000000],
            'tmdb_id': [12345],
            'vote_average': [7.5],
            'popularity': [100.0],
        })

        result = process_movie_tmdb_features(df, is_movie=True)

        assert result['adult'].iloc[0] == False
        assert result['original_language'].iloc[0] == 'en'
        assert result['release_date'].iloc[0] == '2020-05-15'
        assert result['revenue'].iloc[0] == 1000000
        assert result['missing_tmdb'].iloc[0] == False
        assert result['vote_average'].iloc[0] == 7.5
        assert result['popularity'].iloc[0] == 100.0
        assert result['is_movie'].iloc[0] == True
        assert result['missing_release_date'].iloc[0] == False

    def test_movie_with_missing_tmdb_data(self):
        """Test processing movie with missing TMDB fields."""
        df = pd.DataFrame({
            'adult': [np.nan],
            'original_language': [np.nan],
            'genres': [[]], # Empty list, not NaN
            'release_date': [np.nan],
            'revenue': [np.nan],
            'tmdb_id': [np.nan],
            'vote_average': [np.nan],
            'popularity': [np.nan],
        })

        result = process_movie_tmdb_features(df, is_movie=True)

        assert result['adult'].iloc[0] == False
        assert result['original_language'].iloc[0] == 'unknown'
        assert result['genres'].iloc[0] == []
        assert result['release_date'].iloc[0] == '1900-01-01'
        assert result['revenue'].iloc[0] == 0
        assert result['missing_tmdb'].iloc[0] == True
        assert result['vote_average'].iloc[0] == 0
        assert result['popularity'].iloc[0] == 0
        assert result['is_movie'].iloc[0] == True
        # Note: missing_release_date is False because NaN was replaced before the check
        assert result['missing_release_date'].iloc[0] == False

    def test_nonmovie_defaults(self):
        """Test that non-movies get appropriate default values."""
        df = pd.DataFrame({
            'title': ['Some TV Show'],
        })

        result = process_movie_tmdb_features(df, is_movie=False)

        assert result['adult'].iloc[0] == False
        assert result['original_language'].iloc[0] == 'unknown'
        assert result['genres'].iloc[0] == []
        assert result['release_date'].iloc[0] == '1900-01-01'
        assert result['revenue'].iloc[0] == 0
        assert result['missing_tmdb'].iloc[0] == True
        assert result['vote_average'].iloc[0] == 0
        assert result['popularity'].iloc[0] == 0
        assert result['is_movie'].iloc[0] == False
        assert result['missing_release_date'].iloc[0] == True

    def test_empty_release_date_handling(self):
        """Test that empty string release dates are handled correctly."""
        df = pd.DataFrame({
            'adult': [False],
            'original_language': ['en'],
            'genres': [[{'name': 'Action'}]],
            'release_date': [''],
            'revenue': [100],
            'tmdb_id': [123],
            'vote_average': [7.0],
            'popularity': [50.0],
        })

        result = process_movie_tmdb_features(df, is_movie=True)

        assert result['release_date'].iloc[0] == '1900-01-01'
        assert result['missing_release_date'].iloc[0] == True

    def test_genres_as_numpy_array(self):
        """Test that genres as numpy array are converted to list."""
        df = pd.DataFrame({
            'adult': [False],
            'original_language': ['en'],
            'genres': [np.array([{'name': 'Comedy'}, {'name': 'Drama'}])],
            'release_date': ['2020-01-01'],
            'revenue': [100],
            'tmdb_id': [123],
            'vote_average': [7.0],
            'popularity': [50.0],
        })

        result = process_movie_tmdb_features(df, is_movie=True)

        assert isinstance(result['genres'].iloc[0], list)
        assert len(result['genres'].iloc[0]) == 2

    def test_multiple_rows(self):
        """Test processing multiple rows correctly."""
        df = pd.DataFrame({
            'adult': [False, False, False],
            'original_language': ['en', 'fr', 'de'],
            'genres': [
                [{'name': 'Action'}],
                [{'name': 'Drama'}],
                [{'name': 'Comedy'}],
            ],
            'release_date': ['2020-01-01', '2019-01-01', '2021-01-01'],
            'revenue': [100, np.nan, 200],
            'tmdb_id': [123, np.nan, 456],
            'vote_average': [7.0, 6.0, 8.0],
            'popularity': [50.0, 40.0, 60.0],
        })

        result = process_movie_tmdb_features(df, is_movie=True)

        assert len(result) == 3
        assert result['missing_tmdb'].iloc[0] == False
        assert result['missing_tmdb'].iloc[1] == True
        assert result['missing_tmdb'].iloc[2] == False


class TestAddGenreFeatures:
    """Tests for add_genre_features function."""

    def test_single_movie_single_genre(self):
        """Test one-hot encoding for single movie with one genre."""
        df = pd.DataFrame({
            'title': ['Movie A'],
            'genres': [[{'name': 'Action'}]],
        })

        result = add_genre_features(df)

        assert 'genre_Action' in result.columns
        assert result['genre_Action'].iloc[0] == 1
        assert 'genres' not in result.columns
        assert 'genre_names' not in result.columns

    def test_single_movie_multiple_genres(self):
        """Test one-hot encoding for single movie with multiple genres."""
        df = pd.DataFrame({
            'title': ['Movie A'],
            'genres': [[{'name': 'Action'}, {'name': 'Comedy'}]],
        })

        result = add_genre_features(df)

        assert 'genre_Action' in result.columns
        assert 'genre_Comedy' in result.columns
        assert result['genre_Action'].iloc[0] == 1
        assert result['genre_Comedy'].iloc[0] == 1

    def test_multiple_movies_different_genres(self):
        """Test one-hot encoding for multiple movies with different genres."""
        df = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'genres': [
                [{'name': 'Action'}],
                [{'name': 'Comedy'}],
                [{'name': 'Action'}, {'name': 'Drama'}],
            ],
        })

        result = add_genre_features(df)

        assert 'genre_Action' in result.columns
        assert 'genre_Comedy' in result.columns
        assert 'genre_Drama' in result.columns

        # Movie A has Action
        assert result['genre_Action'].iloc[0] == 1
        assert result['genre_Comedy'].iloc[0] == 0
        assert result['genre_Drama'].iloc[0] == 0

        # Movie B has Comedy
        assert result['genre_Action'].iloc[1] == 0
        assert result['genre_Comedy'].iloc[1] == 1
        assert result['genre_Drama'].iloc[1] == 0

        # Movie C has Action and Drama
        assert result['genre_Action'].iloc[2] == 1
        assert result['genre_Comedy'].iloc[2] == 0
        assert result['genre_Drama'].iloc[2] == 1

    def test_movie_with_no_genres(self):
        """Test handling of movies with empty genre list."""
        df = pd.DataFrame({
            'title': ['Movie A', 'Movie B'],
            'genres': [[], [{'name': 'Action'}]],
        })

        result = add_genre_features(df)

        assert 'genre_Action' in result.columns
        # Movie A should have 0 for all genres
        assert result['genre_Action'].iloc[0] == 0
        # Movie B should have 1 for Action
        assert result['genre_Action'].iloc[1] == 1


class TestAddMovieAgeFeature:
    """Tests for add_movie_age_feature function."""

    def test_recent_movie(self):
        """Test age calculation for recent movie."""
        current_year = datetime.now().year
        df = pd.DataFrame({
            'release_date': [f'{current_year}-01-01'],
        })

        result = add_movie_age_feature(df)

        assert 'movie_age' in result.columns
        assert result['movie_age'].iloc[0] == 0
        assert 'release_date' not in result.columns
        assert 'release_date_dt' not in result.columns

    def test_old_movie(self):
        """Test age calculation for old movie."""
        current_year = datetime.now().year
        df = pd.DataFrame({
            'release_date': ['1990-06-15'],
        })

        result = add_movie_age_feature(df)

        expected_age = current_year - 1990
        assert result['movie_age'].iloc[0] == expected_age

    def test_multiple_movies_different_ages(self):
        """Test age calculation for multiple movies."""
        current_year = datetime.now().year
        df = pd.DataFrame({
            'release_date': [
                f'{current_year}-01-01',
                '2010-05-20',
                '1985-12-25',
            ],
        })

        result = add_movie_age_feature(df)

        assert result['movie_age'].iloc[0] == 0
        assert result['movie_age'].iloc[1] == current_year - 2010
        assert result['movie_age'].iloc[2] == current_year - 1985

    def test_default_date_handling(self):
        """Test handling of default '1900-01-01' date."""
        current_year = datetime.now().year
        df = pd.DataFrame({
            'release_date': ['1900-01-01'],
        })

        result = add_movie_age_feature(df)

        expected_age = current_year - 1900
        assert result['movie_age'].iloc[0] == expected_age


class TestFinalizeMLFeatures:
    """Tests for finalize_ml_features function."""

    def test_boolean_to_int_conversion(self):
        """Test that boolean columns are converted to int."""
        df = pd.DataFrame({
            'is_movie': [True, False, True],
            'adult': [False, True, False],
            'numeric_col': [1.5, 2.5, 3.5],
        })

        result = finalize_ml_features(df)

        assert result['is_movie'].dtype == int
        assert result['adult'].dtype == int
        assert result['is_movie'].iloc[0] == 1
        assert result['is_movie'].iloc[1] == 0
        assert result['adult'].iloc[0] == 0
        assert result['adult'].iloc[1] == 1
        # Numeric columns should remain unchanged
        assert result['numeric_col'].dtype == float

    def test_no_nans_validation_passes(self):
        """Test that validation passes when no NaNs present."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
        })

        result = finalize_ml_features(df)
        assert result is not None

    def test_nans_validation_warns(self):
        """Test that warning is logged when NaNs are present (no assertion error)."""
        df = pd.DataFrame({
            'col1': [1, np.nan, 3],
            'col2': ['a', 'b', 'c'],
        })

        # Should log warning but not raise error
        result = finalize_ml_features(df)

        # Result should still be returned with NaNs
        assert result is not None
        assert 'col1' in result.columns
        assert pd.isna(result['col1'].iloc[1])


class TestBuildXFeaturesMoviesOnly:
    """Tests for build_X_features_movies_only function."""

    def test_complete_pipeline_single_movie(self):
        """Test complete feature building pipeline for single movie."""
        current_year = datetime.now().year
        df = pd.DataFrame({
            'title': ['Test Movie'],
            'adult': [False],
            'original_language': ['en'],
            'genres': [[{'name': 'Action'}, {'name': 'Comedy'}]],
            'release_date': ['2020-05-15'],
            'revenue': [1000000],
            'tmdb_id': [12345],
            'vote_average': [7.5],
            'popularity': [100.0],
        })

        result = build_X_features_movies_only(df)

        # Check all processing happened
        assert 'genre_Action' in result.columns
        assert 'genre_Comedy' in result.columns
        assert 'movie_age' in result.columns
        assert result['movie_age'].iloc[0] == current_year - 2020
        assert 'genres' not in result.columns
        assert 'release_date' not in result.columns
        assert result['is_movie'].iloc[0] == 1
        assert result['adult'].iloc[0] == 0

    def test_complete_pipeline_multiple_movies(self):
        """Test complete feature building pipeline for multiple movies."""
        df = pd.DataFrame({
            'title': ['Movie A', 'Movie B'],
            'adult': [False, True],
            'original_language': ['en', 'fr'],
            'genres': [
                [{'name': 'Action'}],
                [{'name': 'Drama'}],
            ],
            'release_date': ['2020-05-15', '2015-08-20'],
            'revenue': [1000000, 500000],
            'tmdb_id': [123, 456],
            'vote_average': [7.5, 6.0],
            'popularity': [100.0, 50.0],
        })

        result = build_X_features_movies_only(df)

        assert len(result) == 2
        assert 'genre_Action' in result.columns
        assert 'genre_Drama' in result.columns
        assert result['genre_Action'].iloc[0] == 1
        assert result['genre_Drama'].iloc[0] == 0
        assert result['genre_Action'].iloc[1] == 0
        assert result['genre_Drama'].iloc[1] == 1

    def test_movie_with_missing_tmdb_data(self):
        """Test pipeline handles missing TMDB data gracefully."""
        df = pd.DataFrame({
            'adult': [np.nan],
            'original_language': [np.nan],
            'genres': [[]], # Empty list instead of NaN
            'release_date': [np.nan],
            'revenue': [np.nan],
            'tmdb_id': [np.nan],
            'vote_average': [np.nan],
            'popularity': [np.nan],
        })

        result = build_X_features_movies_only(df)

        # Should have defaults filled in
        assert result['adult'].iloc[0] == 0
        assert result['original_language'].iloc[0] == 'unknown'
        assert result['revenue'].iloc[0] == 0
        assert result['missing_tmdb'].iloc[0] == 1
        assert result['is_movie'].iloc[0] == 1
        # Should have no NaNs
        assert result.isna().sum().sum() == 0
