from datetime import date
from cts_recommender.preprocessing.dates import get_season, is_holiday_indexed, build_holiday_index

def test_get_season():
    assert get_season(date(2025, 6, 1)) == "summer"

def test_index_build_and_query():
    holidays = {
        "cantons": {
            "GE": {
                # allow strings for holidays
                "public_holidays": [{"date": "2025-01-01"}],
                # and dicts for vacations
                "school_vacations": [{"start": "2025-04-01", "end": "2025-04-10"}],
            }
        }
    }
    idx = build_holiday_index(holidays)

    assert is_holiday_indexed("2025-01-01", idx) is True      # holiday
    assert is_holiday_indexed("2025-04-05", idx) is True      # inside vacation
    assert is_holiday_indexed("2025-02-01", idx) is False     # normal day
