"""
Unit tests for TMDB API matching functionality.

Tests verify that TMDB fuzzy matching returns correct movie IDs
for known titles to prevent regression in catalog enrichment.
"""
import pytest
from cts_recommender.adapters.tmdb.tmdb import TMDB_API


@pytest.fixture
def tmdb_api():
    """Create TMDB API instance for testing."""
    return TMDB_API()


class TestTMDBMatching:
    """Test TMDB movie matching with known cases."""

    def test_the_martian_french_title(self, tmdb_api):
        """Test matching 'Seul sur Mars' returns correct TMDB ID."""
        result = tmdb_api.find_best_match("Seul sur Mars", known_runtime=135)
        assert result == 286217, "Should match The Martian (2015)"

    def test_the_martian_english_title(self, tmdb_api):
        """Test matching 'The Martian' returns correct TMDB ID."""
        result = tmdb_api.find_best_match("The Martian", known_runtime=135)
        assert result == 286217, "Should match The Martian (2015)"

    def test_once_upon_time_in_west(self, tmdb_api):
        """Test matching 'Il était une fois dans l'Ouest' returns correct TMDB ID."""
        result = tmdb_api.find_best_match("Il était une fois dans l'Ouest", known_runtime=165)
        assert result == 335, "Should match Once Upon a Time in the West (1968)"

    def test_how_to_train_your_dragon_3(self, tmdb_api):
        """Test matching 'Dragons 3' returns correct TMDB ID."""
        result = tmdb_api.find_best_match("Dragons 3 : Le monde caché", known_runtime=104)
        assert result == 166428, "Should match How to Train Your Dragon: The Hidden World (2019)"

    def test_space_jam_new_legacy(self, tmdb_api):
        """Test matching 'Space Jam : nouvelle ère' returns correct TMDB ID."""
        result = tmdb_api.find_best_match("Space Jam : nouvelle ère", known_runtime=115)
        assert result == 379686, "Should match Space Jam: A New Legacy (2021)"

    def test_bourne_identity(self, tmdb_api):
        """Test matching 'La mémoire dans la peau' returns correct TMDB ID."""
        result = tmdb_api.find_best_match("La mémoire dans la peau", known_runtime=119)
        assert result == 2501, "Should match The Bourne Identity (2002)"

    def test_matching_without_runtime(self, tmdb_api):
        """Test that matching works without runtime info."""
        result = tmdb_api.find_best_match("The Martian", known_runtime=None)
        assert result == 286217, "Should still match The Martian without runtime"

    def test_no_match_returns_none(self, tmdb_api):
        """Test that invalid titles return None."""
        result = tmdb_api.find_best_match("ZZZZZ_INVALID_MOVIE_TITLE_12345", known_runtime=None)
        assert result is None, "Should return None for non-existent movies"
