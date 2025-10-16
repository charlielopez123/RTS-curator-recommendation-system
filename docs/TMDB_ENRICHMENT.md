# TMDB Enrichment Consistency

## Problem

TMDB fuzzy matching returns different movie IDs for the same title when run at different times. This breaks catalog matching.

**Example**: "Seul sur Mars" (The Martian)
- Programming enrichment → TMDB 286217 ✅ (correct)
- Catalog enrichment → TMDB 837912 ❌ (wrong movie)
- Result: Broadcast can't match to catalog

**Impact**: 27% of RTS broadcasts were unmatched, losing 340 IL training samples.

## Solution

### Current: Title-Based Fallback
Historical programming uses two-stage matching:
1. TMDB ID matching (primary)
2. Title matching (fallback for TMDB inconsistencies)

**Result**: 96.1% match rate (1193/1242 RTS broadcasts)

### Recommended: Re-enrich Catalog
TMDB API now returns correct results. Re-run to fix at source:

```bash
make extract-whatson
```

**Expected**: Near 100% match rate

## Validation

Test TMDB matching before re-enriching:
```bash
pytest tests/unit/test_tmdb_matching.py -v
```

## When to Re-enrich
- High mismatch rates detected
- Before new IL training cycles
- Periodically (quarterly)

## Current Stats
- RTS broadcasts: 1242
- Matched: 1193 (96.1%)
- Training samples: 1193
- Target: >95% match rate
