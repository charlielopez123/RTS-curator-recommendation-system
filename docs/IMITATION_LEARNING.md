# Imitation Learning for TV Programming

This document describes the offline imitation learning (IL) system used to train the TV programming recommendation model by learning from historical curator decisions.

## Overview

The imitation learning system treats historical programming schedules as "expert demonstrations" and learns to replicate curator decision-making by combining:

1. **Curator Selection Signal** - The fact that a curator chose a specific movie in a specific context
2. **Multi-Objective Reward Components** - Quality metrics that explain why the decision was good

## Architecture

### Core Components

```
src/cts_recommender/imitation_learning/
├── IL_training.py              # Main training data processor
├── IL_constants.py             # Reward weights and hyperparameters
└── times_shown_tracker.py      # Dynamic times_shown computation
```

### Key Classes

#### `HistoricalDataProcessor`

Processes historical programming data for offline training.

**Responsibilities:**
- Iterates chronologically through historical broadcast decisions
- Creates training samples from curator choices
- Computes pseudo-rewards for each decision
- Maintains temporal consistency (no data leakage)

**Key Parameters:**
- `environment`: TVProgrammingEnvironment instance
- `historical_data`: Historical programming DataFrame
- `gamma`: Weighting factor for curator selection vs. reward signals (default: 0.6)

#### `TimesShownTracker`

Dynamically tracks broadcast counts during IL training iteration.

**Purpose:**
Ensures temporal correctness by only counting broadcasts that occurred *before* the current decision point.

**Data Sources:**
1. Catalog broadcast date columns (first_broadcast_date, rebroadcast_date_1-4)
2. Historical programming records (interest channels only: RTS1, RTS2)

**Key Methods:**
- `get_times_shown(catalog_id, reference_date)` - Count broadcasts before reference_date
- `get_last_broadcast_date(catalog_id, reference_date)` - Get most recent broadcast before reference_date

## Training Data Generation

### Process Flow

```python
# 1. Initialize tracker
tracker = TimesShownTracker(catalog_df, historical_df, interest_channels)

# 2. Iterate chronologically through historical decisions
for row in historical_df.iterrows():
    air_date = row['date']
    movie_id = row['catalog_id']

    # 3. Get times_shown AS OF air_date (no future leakage)
    times_shown = tracker.get_times_shown(movie_id, air_date)

    # 4. Compute rewards using temporally-correct counts
    rewards = env.reward.compute_rewards_for_historical_row(row, tracker)

    # 5. Create training sample
    sample = create_positive_sample(row, rewards)
```

### Sample Structure

Each training sample contains:

```python
{
    'context_features': [...],      # Temporal context (hour, day, season, etc.)
    'movie_features': [...],        # Movie metadata (genres, ratings, etc.)
    'movie_id': 'catalog_id',       # Movie identifier
    'selected': 1,                  # Positive sample (curator chose this)
    'reward': 0.85,                 # Combined reward signal
    'date': pd.Timestamp,           # Broadcast date
    'context_cache_key': 'key',     # For feature caching
    'current_memory': [...]         # Recent programming history
}
```

## Reward Components

Training samples are labeled with a **hybrid reward signal**:

```python
reward = gamma * curator_selection + (1 - gamma) * pseudo_reward
```

### Curator Selection Signal (gamma = 0.6)

**Value:** 1.0 (binary - curator chose this movie)

**Rationale:** The fact that a curator selected this movie in this context is itself a strong learning signal, representing implicit domain expertise.

### Pseudo-Reward (1 - gamma = 0.4)

Weighted combination of multiple objectives:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Audience** | 0.40 | Predicted/actual audience ratings |
| **Competition** | 0.15 | Competitive advantage timing |
| **Diversity** | 0.15 | Programming diversity (genre, language, etc.) |
| **Novelty** | 0.15 | Freshness (avoids repetition) |
| **Rights Urgency** | 0.15 | Urgency to use expiring broadcast rights |

**Weights defined in:** `src/cts_recommender/imitation_learning/IL_constants.py`

## Times Shown Tracking

### Why Dynamic Tracking?

During IL training, we must ensure **temporal correctness**:
- When processing a broadcast from 2023-03-15, `times_shown` should only count broadcasts *before* that date
- Static catalog values reflect the *final* state, not the state at each historical decision point

### Implementation

The `TimesShownTracker` computes `times_shown` on-demand:

```python
def get_times_shown(catalog_id: str, reference_date: pd.Timestamp) -> int:
    count = 0

    # 1. Count from catalog broadcast dates < reference_date
    for col in ['first_broadcast_date', 'rebroadcast_date_1', ...]:
        if catalog_df.loc[catalog_id, col] < reference_date:
            count += 1

    # 2. Count from historical records < reference_date (interest channels only)
    count += historical_df[
        (historical_df['catalog_id'] == catalog_id) &
        (historical_df['date'] < reference_date) &
        (historical_df['channel'].isin(['RTS1', 'RTS2']))
    ].shape[0]

    return count
```

### Key Properties

✅ **No double counting** - Each broadcast counted exactly once
✅ **Temporally correct** - Only counts past broadcasts
✅ **Interest channels only** - Counts RTS1/RTS2, not competitors
✅ **Read-only** - Queries data directly, no state mutations

## Usage Example

```python
from cts_recommender.imitation_learning.IL_training import HistoricalDataProcessor
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment

# 1. Load data
catalog_df = load_whatson_catalog()
historical_df = load_historical_programming()

# 2. Initialize environment
env = TVProgrammingEnvironment(
    catalog_df=catalog_df,
    historical_df=historical_df,
    # ... other params
)

# 3. Create processor
processor = HistoricalDataProcessor(
    environment=env,
    historical_data=historical_df,
    gamma=0.6  # 60% curator selection, 40% pseudo-reward
)

# 4. Generate training data
training_data = processor.prepare_training_data(
    negative_sampling_ratio=5,
    time_split_date='2024-01-01'  # Train/validation split
)
```

## Temporal Validation

To prevent data leakage during evaluation:

```python
# Split by time for temporal validation
training_data = processor.prepare_training_data(
    time_split_date='2023-12-01'
)

# Training set: broadcasts before 2023-12-01
# Validation set: broadcasts after 2023-12-01
```

This ensures the model is evaluated on *future* decisions it hasn't seen during training, mimicking real-world deployment.

## Integration with Reward Calculator

The reward system has two modes:

### 1. IL Training Mode (Historical Data)

```python
# Uses TimesShownTracker for dynamic computation
rewards = reward_calculator.compute_rewards_for_historical_row(
    historical_row=row,
    times_shown_tracker=tracker  # Temporally-correct counts
)
```

### 2. Online/Simulation Mode

```python
# Uses static catalog values
rewards = reward_calculator.compute_rewards_for_historical_row(
    historical_row=row,
    times_shown_tracker=None  # Falls back to catalog['times_shown']
)
```

This separation ensures:
- **Training:** Temporally correct, prevents data leakage
- **Production:** Fast inference using precomputed catalog values

## Best Practices

### 1. Data Preparation

✅ **Sort historical data by date** before processing
✅ **Filter to interest channels** (RTS1, RTS2) for training samples
✅ **Include all channels** in historical_df for novelty/competition calculations

### 2. Temporal Consistency

✅ **Always use tracker** during IL training
✅ **Never use future information** when computing features
✅ **Reset environment memory** before each training run

### 3. Negative Sampling

Consider adding negative samples (movies not chosen):

```python
# For each positive sample (curator chose movie A)
# Sample N movies that were NOT chosen in that context
training_data = processor.prepare_training_data(
    negative_sampling_ratio=5  # 5 negatives per positive
)
```

### 4. Hyperparameter Tuning

Key hyperparameters to tune:

| Parameter | Range | Description |
|-----------|-------|-------------|
| `gamma` | 0.5-0.8 | Balance between curator signal and rewards |
| `negative_sampling_ratio` | 3-10 | Number of negative samples per positive |
| Reward component weights | 0.0-1.0 | Individual reward importance |

## Limitations and Future Work

### Current Limitations

1. **No negative sampling implemented yet** - Currently only uses positive samples (curator choices)
2. **Fixed reward weights** - Weights are manually set, not learned
3. **No cross-channel learning** - Only learns from RTS1/RTS2 decisions

### Future Enhancements

1. **Implement negative sampling** - Sample non-chosen movies for contrastive learning
2. **Multi-task learning** - Jointly predict multiple objectives (audience, diversity, etc.)
3. **Inverse reinforcement learning** - Learn reward weights from curator behavior
4. **Temporal validation metrics** - Add time-based evaluation protocols

## Related Documentation

- [Reward Components](./REWARD_COMPONENTS.md) - Detailed reward calculations
- [WhatsOn Catalog](./WHATSON_CATALOG.md) - Catalog data structure
- [Historical Programming Pipeline](./HISTORICAL_PROGRAMMING_PIPELINE.md) - Data pipeline
- [Competition Scraping](./COMPETITION_SCRAPING.md) - Competitor data integration

## References

- **Imitation Learning:** Learning from expert demonstrations
- **Behavioral Cloning:** Supervised learning from expert actions
- **Inverse Reinforcement Learning:** Learning reward functions from behavior
- **Contextual Bandits:** Online learning with context-dependent rewards
