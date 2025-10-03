# What's On Catalogue Extraction

## Overview

This document describes the process of extracting and filtering movie data from the RTS "What's On" catalog CSV file. The What's On catalog contains a mix of content types (movies, TV series episodes, documentaries, etc.), and our goal is to extract only actual movies for building a recommendation system.

## Source Data

**Input File**: `data/raw/original_raw_whatson.csv`
- Format: Tab-separated CSV
- Total Records: ~17,336 entries
- Language: French (Swiss RTS catalog)

### Original Schema

The source CSV contains 29 columns with French names:

| Original Column | Description |
|----------------|-------------|
| Collection | Collection or series name |
| Titre | Title |
| Titre Original | Original title (often English) |
| Durée | Duration (HH:MM:SS format) |
| Description courte | Short description |
| Code contenu | Content code (71=Films, 72=Téléfilms, 761=Animation) |
| Régions de Production | Production regions |
| Année de Production | Production year |
| Contrôle Parental | Parental control rating |
| Départment | Department |
| Réalisation | Director |
| Acteurs/Actrices | Actors |
| ... | (and 17 more broadcasting/rights columns) |

## Schema Transformation

All column names are renamed to snake_case English equivalents for consistency:

**Key Mappings** (see `src/cts_recommender/features/whatson_csv_schema.py`):
- `Collection` → `collection`
- `Titre` → `title`
- `Titre Original` → `original_title`
- `Durée` → `duration`
- `Code contenu` → `class_key`
- `Année de Production` → `production_year`
- `Réalisation` → `director`

## The Filtering Challenge

The What's On catalog is **messy** and contains:
- ✅ Feature films
- ✅ TV movies (téléfilms)
- ❌ TV series episodes (often with episode indicators like "ép.1", "Episode 1")
- ❌ Documentary series
- ❌ Short clips and trailers (< 35 minutes)
- ❌ Recurring programs (Box office shows, etc.)

### Key Data Quality Issues

1. **Inconsistent Titles**: Sometimes the title is an episode number ("ép.1") and the actual show name is in `Collection`
2. **Generic Collections**: Collections like "Film" or "Téléfilm" contain thousands of entries
3. **Missing Data**: Many `original_title` fields are empty (NaN)
4. **Series in Film Collections**: TV series episodes are sometimes marked as "Films de cinéma"
5. **Duplicate Entries**: Same title appears multiple times (e.g., "Rosa" appears 12 times - it's a series)

## Filtering Rules

We developed a **11-rule system** to identify movies vs. TV series episodes:

### Rule 1: Episode Pattern Detection ❌
If `title` OR `original_title` contains episode patterns → **NOT a movie**

Episode patterns:
```python
- r'ép\.\s*\d+'          # ép.1, ép. 2
- r'épisode\s*\d+'       # épisode 1, épisode 2
- r'^ep\.\s*\d+'         # ep.1, ep. 2
- r'^E\d+'               # E1, E23
- r'S\d+E\d+'            # S01E01, S1E2
- r'^\d+$'               # Just numbers: "1", "2"
- r'^\d+\s*-\s*'         # "1 - Title", "2 - Something"
- r'partie\s*\d+'        # partie 1, partie 2
- r'\bpart\s*\d+'        # part 1, part 2
```

### Rule 2: Date Pattern Detection ❌
If `title` contains date patterns → **NOT a movie** (likely recurring shows)

Date patterns:
```python
- r'\d{4}-\d{2}-\d{2}'   # YYYY-MM-DD (e.g., "Box office - 2018-06-11")
- r'\d{2}-\d{2}-\d{2}'   # YY-MM-DD or DD-MM-YY
- r'\d{2}\.\d{2}\.\d{2}' # YY.MM.DD or DD.MM.YY
- r'\d{2}/\d{2}/\d{2}'   # YY/MM/DD or DD/MM/YY
```

### Rule 3: Title Frequency Check ❌
If a `title` appears **more than 3 times** in the dataset → **NOT a movie** (likely series)

Example: "Episode 1" appears 40 times, "Top models" appears 34 times

### Rule 4: Collection Frequency Check ❌
If `collection` appears **more than 2 times** AND doesn't match film keywords → **NOT a movie**

This filters out TV series collections like:
- "Camping Paradis" (100 entries)
- "Rosamunde Pilcher" (78 entries)
- "Alex Hugo" (35 entries)

While preserving film collections like:
- "Film" (8,003 entries - has film keyword)
- "Fiction" (3,522 entries - has film keyword)

### Rule 5: Duration Check ❌
If `duration` is present (not `00:00:00`) AND less than **35 minutes** → **NOT a movie**

Filters out:
- Short clips
- Trailers
- TV episode segments

Note: `00:00:00` duration is treated as "unknown" and allowed to pass through.

### Rule 6: Generic Collection Name ❌
If `collection == title` AND `collection` matches film keywords → **NOT a movie**

Example: Row where title="Film" and collection="Film" is too generic

### Rule 7: Same Collection/Title Duration Check ❌
If `collection == title` (and NOT a film keyword), check `duration` ≥ 35 min or `00:00:00` → else **NOT a movie**

### Rule 8: Title Matches Original Title ✅
If `title == original_title` (exact match) → **IS a movie**

This is a strong positive signal that the entry is properly catalogued.

### Rule 9: Title in Collection ❌
If `title` is a substring of `collection` (length ≥ 3) → **NOT a movie** (likely series)

Example: title="Hugo" in collection="Alex Hugo" → series episode

### Rule 10: Film Collection Keywords ✅
If `collection` matches known film patterns → **IS a movie**

**Known Film Collections** (whitelist):
```python
'Film', 'Téléfilm', 'Fiction', 'Fiction\Achats', 'Cinéma',
'Film d\'action', 'Emotions fortes', 'Comédie', 'Film Jeunesse',
'Film de minuit', 'Ecran TV', 'Les classiques du cinéma',
'Court métrage', 'Nocturne', 'Box office'
```

**Film Keywords** (pattern matching):
```python
'film', 'cinéma', 'cinema', 'fiction', 'téléfilm',
'comédie', 'action', 'classique', 'nocturne', 'écran'
```

### Rule 11: Conservative Default ❌
If none of the above rules match → **NOT a movie** (conservative approach)

## Title Selection Logic

Since titles are inconsistent, we select the best title using this priority:

### Priority 1: Original Title (if not episode)
Use `original_title` if:
- Present and not empty
- Not an episode pattern

### Priority 2: Title (if not episode)
Use `title` if:
- Not an episode pattern
- Not empty

### Priority 3: Collection (fallback)
Use `collection` as last resort (even if generic like "Film")

### Example Transformations

| Collection | Titre | Titre Original | → Best Title |
|-----------|-------|----------------|--------------|
| Addict | ép.1 | NaN | Addict |
| À l'est d'éden | A l'est d'eden | East of Eden | East of Eden |
| Film | Beethoven | Beethoven | Beethoven |
| 24 heures pour survivre | L'ombre du soir | Ex : un samedi pas comme les autres | Ex : un samedi pas comme les autres |

## Results

After applying all filtering rules:

- **Input**: 17,336 records
- **Output**: ~11,000-12,000 movies (varies based on rule tuning)
- **Filtered Out**: ~5,000-6,000 TV series episodes, shorts, and invalid entries

### Content Code Distribution (Final Dataset)
```
71 - Films de cinéma              ~8,000 movies
72 - Téléfilms                    ~3,000 TV movies
761 - Longs métrages d'animation    ~100 animated films
```

## Implementation

### Files

1. **Schema Definition**: `src/cts_recommender/features/whatson_csv_schema.py`
   - Column name mappings
   - Original column list validation

2. **Exploration Notebook**: `experiments/notebooks/whatson_data.ipynb`
   - Data analysis
   - Pattern discovery
   - Rule development and testing
   - Validation of filtering logic

3. **Production Pipeline**: `src/cts_recommender/preprocessing/whatson_extraction.py` (TODO)
   - Implement production-ready extraction
   - Apply all filtering rules
   - Export cleaned movie catalog

### Usage Example

```python
from cts_recommender.io import readers
from cts_recommender.settings import get_settings
from cts_recommender.features.whatson_csv_schema import WHATSON_RENAME_MAP

cfg = get_settings()

# Load and rename
whatson_df = readers.read_csv(cfg.raw_dir / "original_raw_whatson.csv", sep='\t')
whatson_df = whatson_df.rename(columns=WHATSON_RENAME_MAP)

# Apply filtering (see notebook for full implementation)
movies_df = filter_movies(whatson_df)

# movies_df now contains only actual movies
```

## Validation

The notebook includes validation cells that verify each rule:

```
Rule 1 - Episode patterns: 367 filtered (0 kept)
Rule 2 - Date patterns: 22 filtered (0 kept)
Rule 3 - Title duplicates: ~800 filtered (0 kept)
Rule 4 - Collection duplicates: ~500 filtered (0 kept)
Rule 5 - Short duration: ~300 filtered (0 kept)
...
```

## Future Improvements

1. **Manual Review**: Flag edge cases for human review
2. **Collection Whitelist**: Maintain a curated list of known series collections
3. **NLP Classification**: Use description text to classify content type
4. **External Validation**: Cross-reference with TMDB/IMDb to verify movies
5. **Year Validation**: Filter out entries with invalid or missing production years

## Notes

- The filtering is **conservative**: when in doubt, we exclude rather than include
- Rule 8 (title == original_title) provides a positive signal before applying negative filters
- The 35-minute threshold is a heuristic (typical TV episode: 20-50 min, movies: 60+ min)
- `00:00:00` durations are treated as "unknown" rather than invalid
- Generic collections like "Film" are allowed only if they pass duration and other checks

---

**Last Updated**: 2025-10-03
**Notebook**: `experiments/notebooks/whatson_data.ipynb`
