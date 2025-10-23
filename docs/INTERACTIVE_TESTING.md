# Interactive CTS Testing

This guide explains how to use the interactive testing tool for the Contextual Thompson Sampler (CTS) recommendation system.

## Overview

The interactive testing tool allows you to:
- Test CTS recommendations in different contexts (time, day, channel)
- View detailed signal breakdowns for each recommendation
- Provide feedback on recommendations (accept/reject)
- Indicate which signals were pertinent to your choice
- Save the updated CTS model after testing

## Quick Start

### Using Make (Recommended)

```bash
make interactive-test
```

This uses the default model and data paths from your configuration.

### Using the CLI Directly

```bash
uv run cts-reco-interactive-test
```

Or with custom paths:

```bash
uv run cts-reco-interactive-test \
  --cts-model data/models/cts_model.npz \
  --curator-model data/models/curator_logistic_model.joblib \
  --audience-model data/models/audience_ratings_model.joblib \
  --catalog data/processed/whatson/whatson_catalog.parquet \
  --historical-programming data/processed/programming/historical_programming.parquet \
  --num-days 7 \
  --num-recommendations 5
```

## How to Use

### 1. View Recommendations

For each context, you'll see:
- **Date and context** (hour, day of week, channel, season)
- **Top 5 movie recommendations** with:
  - Movie title and release year
  - Overall CTS score
  - **Predicted Audience Rating** (%)
  - **Rights End Date** (when TV rights expire)
  - Signal breakdown showing:
    - `[0] audience` - Expected audience rating
    - `[1] competition` - Competitive advantage
    - `[2] diversity` - Content diversity
    - `[3] novelty` - How recently shown
    - `[4] rights` - Rights urgency
    - `[5] curator_prob` - Curator acceptance probability

### 2. Provide Feedback

You have four options:

**Select a recommendation (1-5)**
```
Your choice: 1
```

After selecting, you can indicate which signals were pertinent:
```
Signal indices: 0 2 4
```
This tells the system that audience, diversity, and rights were important for your choice.

**Regenerate (r)**
```
Your choice: r
```
Rejects current slate and shows new recommendations (excluding rejected movies).

**Skip (s)**
```
Your choice: s
```
Skip this context and move to the next date/time/channel.

**Quit (q)**
```
Your choice: q
```
Exit the testing session.

### 3. Review Results

At the end, you'll see:
- Total recommendations made
- Signals that were most frequently marked as pertinent
- Summary statistics

## Advanced Options

### Save Updated Model

To save the CTS model after it's been updated with your feedback:

```bash
uv run cts-reco-interactive-test \
  --save-updated-model data/models/cts_model_updated.npz
```

### Customize Testing Parameters

```bash
uv run cts-reco-interactive-test \
  --num-days 14 \              # Test over 14 days instead of 7
  --num-recommendations 10 \   # Show top-10 instead of top-5
  --log-level DEBUG            # More verbose logging
```

## Signal Definitions

- **[0] audience**: Expected audience rating (normalized)
- **[1] competition**: Competitive advantage against other channels
- **[2] diversity**: Content diversity relative to recent programming
- **[3] novelty**: Time since last showing (higher = not shown recently)
- **[4] rights**: Urgency based on rights expiration
- **[5] curator_prob**: Probability that human curator would select this movie

## Tips

1. **Signal Feedback**: Providing signal feedback helps the CTS learn which factors matter most in different contexts. Press Enter to skip if you're not sure.

2. **Regeneration**: Use 'r' when you don't like any of the current recommendations. The system will show new options excluding rejected ones.

3. **Memory**: Once you select a movie, it's added to memory and won't appear in future contexts during this session.

4. **Context Variety**: The system randomly samples different hours and channels to test CTS across diverse contexts.

## Example Session

```
================================================================================
DATE: 2025-10-16
CONTEXT: 21:00 on Wednesday | RTS 1 | fall
================================================================================

[1] The Shawshank Redemption (1994)
    ID: 278
    Overall Score: 0.8542
    Predicted Audience: 18.50% | Rights End: 2026-12-31

    Signal Contributions (weighted by CTS):
      [0] audience       : 0.912 × 0.352 = 0.3210
      [1] competition    : 0.834 × 0.128 = 0.1068
      [2] diversity      : 0.756 × 0.215 = 0.1625
      [3] novelty        : 0.891 × 0.089 = 0.0793
      [4] rights         : 0.445 × 0.156 = 0.0694
      [5] curator_prob   : 0.923 × 0.060 = 0.0554

[2] Inception (2010)
    ...

Options:
  - Enter recommendation number [1-5] to select
  - Enter 'r' to regenerate without rejected movies
  - Enter 's' to skip this context
  - Enter 'q' to quit and end testing

Your choice: 1

You selected: The Shawshank Redemption

Which value signals were pertinent for your selection?
Enter signal indices separated by spaces (e.g., '0 2 4' for audience, diversity, rights)
Signal indices: 0=audience, 1=competition, 2=diversity, 3=novelty, 4=rights, 5=curator_prob
Or press Enter to indicate reward only (no signal feedback)

Signal indices: 0 5

✓ Recommendation accepted and CTS updated (added to memory)
```

## Troubleshooting

**"No available movies"**: This means all movies for that date have already been selected or are filtered out. Skip to the next context.

**Models not found**: Ensure you've run the complete pipeline first:
```bash
make run-all
```

**Import errors**: Reinstall the package:
```bash
uv sync
```
