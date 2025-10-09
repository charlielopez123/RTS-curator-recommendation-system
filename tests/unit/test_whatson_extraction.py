"""Unit tests for WhatsOn catalog extraction functions."""
import pytest
import pandas as pd
import numpy as np

from cts_recommender.preprocessing.whatson_extraction import (
    is_episode_title,
    has_ignore_pattern,
    has_date_pattern,
    parse_duration_minutes,
    is_film_collection,
    select_best_title,
    should_keep_as_movie,
)


class TestIsEpisodeTitle:
    """Tests for is_episode_title function."""

    def test_episode_pattern_ep_dot_number(self):
        """Test detection of 'ép.1' pattern."""
        assert is_episode_title('ép.1') == True
        assert is_episode_title('ép. 2') == True
        assert is_episode_title('ep.1') == True
        assert is_episode_title('ep. 3') == True

    def test_episode_pattern_episode_word(self):
        """Test detection of 'épisode 1' pattern."""
        assert is_episode_title('épisode 1') == True
        assert is_episode_title('episode 2') == True
        assert is_episode_title('èpisode 3') == True

    def test_episode_pattern_season_episode(self):
        """Test detection of S01E01 pattern."""
        assert is_episode_title('S01E01') == True
        assert is_episode_title('S1E2') == True
        assert is_episode_title('S03E15') == True

    def test_episode_pattern_e_number(self):
        """Test detection of E1 pattern at start."""
        assert is_episode_title('E1') == True
        assert is_episode_title('E23') == True

    def test_episode_pattern_just_number(self):
        """Test detection of titles that are just numbers."""
        assert is_episode_title('1') == True
        assert is_episode_title('42') == True

    def test_episode_pattern_number_dash(self):
        """Test detection of '1 - Title' pattern."""
        assert is_episode_title('1 - The Beginning') == True
        assert is_episode_title('2- Second Episode') == True

    def test_episode_pattern_part(self):
        """Test detection of 'partie 1' and 'part 1' patterns."""
        assert is_episode_title('partie 1') == True
        assert is_episode_title('part 2') == True

    def test_non_episode_titles(self):
        """Test that normal movie titles are not detected as episodes."""
        assert is_episode_title('Inception') == False
        assert is_episode_title('The Matrix') == False
        assert is_episode_title('Le Fabuleux Destin') == False

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        assert is_episode_title(np.nan) == False
        assert is_episode_title(None) == False


class TestHasIgnorePattern:
    """Tests for has_ignore_pattern function."""

    def test_doublon_pattern(self):
        """Test detection of 'doublon' pattern."""
        assert has_ignore_pattern('doublon') == True
        assert has_ignore_pattern('doublons') == True
        assert has_ignore_pattern('Film (doublon)') == True

    def test_serie_pattern(self):
        """Test detection of 'série' pattern."""
        assert has_ignore_pattern('serie') == True
        assert has_ignore_pattern('série') == True
        assert has_ignore_pattern('series') == True
        assert has_ignore_pattern('séries') == True

    def test_clean_titles(self):
        """Test that clean titles don't trigger ignore patterns."""
        assert has_ignore_pattern('Inception') == False
        # Note: 'Series' with word boundary will match 'series' pattern
        assert has_ignore_pattern('The Amazing Movie') == False

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        assert has_ignore_pattern(np.nan) == False


class TestHasDatePattern:
    """Tests for has_date_pattern function."""

    def test_date_pattern_yyyy_mm_dd(self):
        """Test detection of YYYY-MM-DD pattern."""
        assert has_date_pattern('Box office - 2018-06-11') == True
        assert has_date_pattern('2020-01-15') == True

    def test_date_pattern_yy_mm_dd(self):
        """Test detection of YY-MM-DD pattern."""
        assert has_date_pattern('Show - 18-06-11') == True
        assert has_date_pattern('20-01-15') == True

    def test_date_pattern_dots(self):
        """Test detection of date with dots."""
        assert has_date_pattern('Show - 18.06.11') == True
        assert has_date_pattern('20.01.15') == True

    def test_date_pattern_slashes(self):
        """Test detection of date with slashes."""
        assert has_date_pattern('Show - 18/06/11') == True
        assert has_date_pattern('20/01/15') == True

    def test_non_date_titles(self):
        """Test that normal titles don't trigger date patterns."""
        assert has_date_pattern('Inception') == False
        assert has_date_pattern('The Matrix 2003') == False  # just year, not date

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        assert has_date_pattern(np.nan) == False


class TestParseDurationMinutes:
    """Tests for parse_duration_minutes function."""

    def test_parse_hh_mm_ss_format(self):
        """Test parsing HH:MM:SS format."""
        assert parse_duration_minutes('01:30:00') == 90
        assert parse_duration_minutes('02:15:30') == 135
        assert parse_duration_minutes('00:45:00') == 45

    def test_parse_mm_ss_format(self):
        """Test parsing MM:SS format."""
        assert parse_duration_minutes('30:00') == 30
        assert parse_duration_minutes('15:30') == 15

    def test_parse_zero_duration(self):
        """Test parsing 00:00:00."""
        assert parse_duration_minutes('00:00:00') == 0

    def test_parse_invalid_format(self):
        """Test that invalid formats return None."""
        assert parse_duration_minutes('invalid') is None
        assert parse_duration_minutes('1:2:3:4') is None
        assert parse_duration_minutes('abc:def:ghi') is None

    def test_parse_nan(self):
        """Test that NaN returns None."""
        assert parse_duration_minutes(np.nan) is None
        assert parse_duration_minutes(None) is None


class TestIsFilmCollection:
    """Tests for is_film_collection function."""

    def test_exact_match_film(self):
        """Test exact matches for film collections."""
        assert is_film_collection('Film') == True
        assert is_film_collection('Téléfilm') == True
        assert is_film_collection('Fiction') == True
        assert is_film_collection('Cinéma') == True

    def test_keyword_match(self):
        """Test keyword matching for film collections."""
        assert is_film_collection('Film d\'action') == True
        assert is_film_collection('Les classiques du cinéma') == True
        assert is_film_collection('Nocturne fiction') == True

    def test_non_film_collections(self):
        """Test that non-film collections return False."""
        assert is_film_collection('Camping Paradis') == False
        assert is_film_collection('Alex Hugo') == False
        assert is_film_collection('Top Models') == False

    def test_case_insensitive_keywords(self):
        """Test that keyword matching is case-insensitive."""
        assert is_film_collection('FILM') == True
        assert is_film_collection('film') == True
        assert is_film_collection('Film') == True

    def test_nan_handling(self):
        """Test that NaN returns False."""
        assert is_film_collection(np.nan) == False


class TestSelectBestTitle:
    """Tests for select_best_title function."""

    def test_priority_1_original_title(self):
        """Test that original_title is preferred when valid."""
        row = pd.Series({
            'title': 'Titre Français',
            'original_title': 'Original English Title',
            'collection': 'Film'
        })
        assert select_best_title(row) == 'Original English Title'

    def test_priority_2_title_when_original_is_episode(self):
        """Test that title is used when original_title is an episode."""
        row = pd.Series({
            'title': 'Good Title',
            'original_title': 'ép.1',
            'collection': 'Film'
        })
        assert select_best_title(row) == 'Good Title'

    def test_priority_3_collection_fallback(self):
        """Test that collection is used as fallback."""
        row = pd.Series({
            'title': 'ép.1',
            'original_title': np.nan,
            'collection': 'Addict'
        })
        assert select_best_title(row) == 'Addict'

    def test_title_preferred_over_collection(self):
        """Test that valid title is preferred over collection."""
        row = pd.Series({
            'title': 'Beethoven',
            'original_title': np.nan,
            'collection': 'Film'
        })
        assert select_best_title(row) == 'Beethoven'

    def test_original_title_with_matching_title(self):
        """Test when original_title and title are the same."""
        row = pd.Series({
            'title': 'Inception',
            'original_title': 'Inception',
            'collection': 'Film'
        })
        assert select_best_title(row) == 'Inception'

    def test_empty_title_handling(self):
        """Test handling of empty strings."""
        row = pd.Series({
            'title': '',
            'original_title': '',
            'collection': 'Film'
        })
        assert select_best_title(row) == 'Film'


class TestShouldKeepAsMovie:
    """Tests for should_keep_as_movie function."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample title and collection counts
        self.title_counts = pd.Series({
            'Episode 1': 40,
            'Top models': 34,
            'Inception': 1,
            'The Matrix': 2,
        })

        self.collection_counts = pd.Series({
            'Camping Paradis': 100,
            'Rosamunde Pilcher': 78,
            'Film': 8003,
            'Fiction': 3522,
            'Inception': 1,
        })

    def test_rule_1_episode_in_title(self):
        """Test Rule 1: Episode pattern in title."""
        row = pd.Series({
            'title': 'ép.1',
            'original_title': np.nan,
            'collection': 'Film',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_rule_1_episode_in_original_title(self):
        """Test Rule 1: Episode pattern in original_title."""
        row = pd.Series({
            'title': 'Something',
            'original_title': 'S01E01',
            'collection': 'Film',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_rule_2_date_in_title(self):
        """Test Rule 2: Date pattern in title."""
        row = pd.Series({
            'title': 'Box office - 2018-06-11',
            'original_title': np.nan,
            'collection': 'Film',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_rule_3_title_appears_many_times(self):
        """Test Rule 3: Title appears more than 3 times."""
        row = pd.Series({
            'title': 'Episode 1',
            'original_title': np.nan,
            'collection': 'Various',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_rule_4_collection_appears_many_times_no_film_keyword(self):
        """Test Rule 4: Collection appears many times without film keyword."""
        row = pd.Series({
            'title': 'Episode Title',
            'original_title': np.nan,
            'collection': 'Camping Paradis',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_rule_5_short_duration(self):
        """Test Rule 5: Duration less than 35 minutes."""
        row = pd.Series({
            'title': 'Short Film',
            'original_title': np.nan,
            'collection': 'Film',
            'duration': '00:20:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_rule_8_title_equals_original_title(self):
        """Test Rule 8: title == original_title (positive signal)."""
        row = pd.Series({
            'title': 'Inception',
            'original_title': 'Inception',
            'collection': 'Some Collection',
            'duration': '02:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == True

    def test_rule_10_film_collection(self):
        """Test Rule 10: Film collection keyword match."""
        row = pd.Series({
            'title': 'Good Movie',
            'original_title': np.nan,
            'collection': 'Film',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == True

    def test_valid_movie_all_checks_pass(self):
        """Test a valid movie that should pass all checks."""
        row = pd.Series({
            'title': 'The Matrix',
            'original_title': 'The Matrix',
            'collection': 'Fiction',
            'duration': '02:16:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == True

    def test_missing_title(self):
        """Test that missing title is rejected."""
        row = pd.Series({
            'title': np.nan,
            'original_title': np.nan,
            'collection': 'Film',
            'duration': '01:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == False

    def test_zero_duration_allowed(self):
        """Test that 00:00:00 duration is allowed (unknown)."""
        row = pd.Series({
            'title': 'Movie',
            'original_title': 'Movie',
            'collection': 'Film',
            'duration': '00:00:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == True

    def test_long_duration_passes(self):
        """Test that movies with long duration pass."""
        row = pd.Series({
            'title': 'Epic Movie',
            'original_title': 'Epic Movie',
            'collection': 'Film',
            'duration': '03:30:00',
        })
        assert should_keep_as_movie(row, self.title_counts, self.collection_counts) == True
