# Competition Scraping System

## Overview

Monitors French competitor TV channels (TF1, M6, France 2, France 3) to track when they air movies that might compete with RTS programming. This provides context for scheduling decisions.

**Location**: `src/cts_recommender/competition/`

**Output**: `competition_data/*.xml` files with competitor schedules

## Architecture

```
CompetitorDataScraper              # Downloads XML schedules
    ↓
Channel Parsers (TF1.py, M6.py)    # Extract cinema entries
    ↓
CompetitorDataManager              # Manage historical + future data
    ↓
MovieCompetitorContext             # Query competitor airings per movie
```

## TV Week Scheduling

Competitor schedules operate on **TV weeks**:
- **Period**: Saturday → Friday (not Mon-Sun)
- **Availability**: 3 weeks in advance
- **Publication**: Saturdays at 18:00 Europe/Zurich time

```python
# Today: Tuesday 2025-10-07
# Available weeks:
#   Week 1: Sat 2025-10-11 → Fri 2025-10-17
#   Week 2: Sat 2025-10-18 → Fri 2025-10-24
#   Week 3: Sat 2025-10-25 → Fri 2025-10-31
```

## Usage

### Scraping Schedules

```python
from cts_recommender.competition.competition_scraping import CompetitorDataScraper

scraper = CompetitorDataScraper()
scraper.scrape_upcoming_schedules(days_ahead=21)  # 3 weeks

# Downloads to: competition_data/TF1_2025_41.xml, M6_2025_41.xml, etc.
```

### Querying Competitor Context

```python
from cts_recommender.competition.competitor import CompetitorDataManager

manager = CompetitorDataManager(historical_competitor_df)

# Get competitor airings for a movie around a reference date
context = manager.get_movie_competitor_context(
    movie_id="27205",  # TMDB ID
    reference_date=datetime(2025, 10, 15),
    window_days_back=180,   # 6 months
    window_days_ahead=21    # 3 weeks
)

print(f"Competitor showings: {len(context.competitor_showings)}")
for showing in context.competitor_showings:
    print(f"{showing['channel']}: {showing['air_date']} at {showing['air_hour']}:00")
```

## Channel Scrapers

### TF1

- **Endpoint**: `https://tf1pro.com/grilles-xml/16/{date}`
- **Auth**: Requires visiting HTML page first to set cookies
- **Filter**: `typeEmission == 'Cinéma'`
- **Output**: `{'date': '09/08/2025', 'time': '21.10', 'title': 'Thor : Ragnarök'}`

### M6

- **Endpoint**: `https://pro.m6.fr/m6/grille/{year}-{week}.xml`
- **Auth**: Direct XML fetch
- **Filter**: `format == 'Long Métrage'`
- **Output**: `{'date': '2025-08-09', 'time': '21:10', 'title': 'Movie Title'}`

Both use **streaming XML parsing** with `lxml.etree.iterparse` for memory efficiency.

## File Naming Convention

Pattern: `{Channel}_{ISO_Year}_{ISO_Week}.xml`

Examples:
- `TF1_2025_31.xml` → TF1, ISO year 2025, week 31
- `M6_2025_52.xml` → M6, ISO year 2025, week 52

## Configuration

Channel configurations in [comp_constants.py](../src/cts_recommender/competition/comp_constants.py):

```python
SITES = {
    'TF1': {
        'base_html': 'https://tf1pro.com/grilles-tv/TF1/{date}',
        'base_xml': 'https://tf1pro.com/grilles-xml/16/{date}',
        'default_headers': { "User-Agent": "...", "Accept": "..." }
    },
    'M6': {
        'base_xml': 'https://pro.m6.fr/m6/grille/{year}-{week}.xml',
        'default_headers': {}
    },
}
```

## Expected Schema

Competition historical data should match RTS historical programming schema with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `catalog_id` | string | Links to WhatsOn catalog (TMDB ID or WO_XXXXXX) |
| `date` | datetime64 | Broadcast date |
| `channel` | string | Competitor channel (TF1_T_PL, M6_T_PL, etc.) |
| `hour` | int64 | Hour of broadcast |
| `title` | string | Movie title from XML |
| `tmdb_id` | Int64 | TMDB ID (nullable, for matching) |

**Note**: Movie features (genres, ratings, etc.) are stored in the catalog and accessed via `catalog_id` joins.

See [catalog_schema.py](../src/cts_recommender/features/catalog_schema.py) for full schema.

## Key Features

✅ **Timezone-aware**: Uses Europe/Zurich for schedule availability
✅ **Memory-efficient**: Streaming XML parsing, element clearing
✅ **Idempotent**: Safe to re-run, skips already-downloaded weeks
✅ **Session management**: Per-channel sessions with connection pooling
✅ **Graceful degradation**: Failures for one channel don't block others


## Related Documentation

- [Historical Programming](./PIPELINE.md) - RTS historical programming structure
- [WhatsOn Catalog](./WHATSON_CATALOG.md) - Catalog schema conventions
- [TMDB Integration](../src/cts_recommender/adapters/tmdb/) - Title matching logic
