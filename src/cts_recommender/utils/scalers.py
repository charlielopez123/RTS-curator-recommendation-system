"""
Sklearn Pipeline Utilities for Feature Scaling

This module provides robust scaling transformations for ML features used in
runtime environments (e.g., TVProgrammingEnvironment for RL/IL training).

## Overview

These scalers are designed to normalize movie and programming features that often have:
- Missing values (TMDB metadata not found)
- Infinite values (from calculations or data errors)
- Negative values (errors in positive-only data like revenue)
- Heavy-tailed distributions (revenue, popularity vary over many orders of magnitude)
- Outliers (blockbuster movies vs. indie films)

## Design Philosophy

### Why This Pipeline?

The `make_safe_positive_pipeline()` creates a 3-step transformation:

1. **Median Imputation** (SimpleImputer)
   - Replaces missing/NaN/inf values with the median of the training data
   - Median is robust to outliers (unlike mean)
   - Ensures no NaN values flow into downstream transformers

2. **Optional Log Compression** (safe_log1p)
   - Applies log1p (log(1 + x)) to compress heavy-tailed distributions
   - Makes features more "normal-like" for better ML performance
   - Only applied to features with exponential distributions
   - Examples: revenue (0 to billions), popularity (0 to thousands), movie_age (0 to 100+)

3. **Quantile Transformation** (QuantileTransformer)
   - Maps values to uniform [0, 1] distribution based on rank
   - Extremely robust to outliers (uses percentiles, not min/max)
   - Ensures all features have the same scale for fair comparison in models
   - Preserves order: higher revenue → higher normalized value

### When to Use log_compress=True vs False?

**Use log_compress=True for heavy-tailed features:**
- `revenue`: Ranges from $0 to $2.8 billion (Avatar)
- `popularity`: TMDB popularity scores (0 to 1000+)
- `movie_age`: Years since release (0 to 100+)
- `rt_m`: Audience ratings (highly skewed distributions)

**Use log_compress=False for bounded/symmetric features:**
- `vote_average`: TMDB ratings (0-10 scale, roughly normal)
- `duration_min`: Movie runtime (70-180 mins, bounded range)

### Why Not StandardScaler or MinMaxScaler?

- **StandardScaler** (z-score normalization):
  - Assumes normal distribution (violated by heavy-tailed features)
  - Sensitive to outliers (mean and std are affected by extremes)
  - No handling of missing/inf values

- **MinMaxScaler** (0-1 linear scaling):
  - Extremely sensitive to outliers (one huge value compresses everything)
  - No handling of missing/inf values
  - Example: A $2B movie would compress all <$100M movies to near-zero

- **QuantileTransformer** (our choice):
  - Distribution-agnostic (works with any shape)
  - Outlier-proof (uses percentiles)
  - Handles skewed data gracefully

## Usage in TVProgrammingEnvironment

The environment fits scalers once on the full catalog during initialization:

```python
self.scaler_dict = {
    "revenue": make_safe_positive_pipeline(log_compress=True).fit(
        self.catalog_df[["revenue"]].replace([np.inf, -np.inf], np.nan)
    ),
    # ... more scalers
}
```

Then transforms individual movie features at runtime:

```python
norm_revenue = self.scaler_dict['revenue'].transform(
    pd.DataFrame({'revenue': [film_row['revenue']]})
)[0][0]
```

### Why use DataFrames instead of [[value]]?

When you fit a transformer with a DataFrame (with column names), sklearn stores those feature names.
To avoid warnings, you should also transform with DataFrames:

- `pd.DataFrame({'revenue': [100]})` → DataFrame with column name ✅ (no warnings)
- `[[100]]` → nested list without column name ⚠️ (triggers sklearn warning)

Using DataFrames for both fit and transform ensures consistency and avoids sklearn's
"X does not have valid feature names" warning.

### Why [0][0] after transform()?

The transformer returns a 2D array: `array([[0.75]])` (shape 1, 1)
- `[0]` extracts the first row → `array([0.75])`
- `[0]` extracts the first element → `0.75` (scalar)

This gives us the normalized scalar value to use in feature vectors.

## Testing Notes

These scalers are deterministic (random_state=0) for reproducibility.
Always fit on training data, then transform both train and test data.

## References

- sklearn QuantileTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html
- Log transformations for skewed data: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
"""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer
from sklearn.impute import SimpleImputer


def safe_log1p(X):
    """
    Safely apply log1p transformation to positive-valued data.

    Handles edge cases:
    - Replaces infinities with NaN (to be handled by upstream imputer)
    - Clips negative values to 0 (assumes data should be naturally positive)
    - Applies log1p for variance stabilization

    Parameters:
    -----------
    X : array-like
        Input data (expected to be positive-valued)

    Returns:
    --------
    np.ndarray
        Log-transformed data

    Examples:
    ---------
    >>> safe_log1p(np.array([[1, 10, 100]]))
    array([[0.69314718, 2.39789527, 4.61512052]])

    >>> safe_log1p(np.array([[np.inf, -5, 0]]))
    array([[nan, 0., 0.]])
    """
    X = np.asarray(X, dtype=float)
    # Replace infinities with NaN so imputer can handle them upstream if needed
    X[~np.isfinite(X)] = np.nan
    # Clip negatives to zero (since data should be naturally positive)
    X = np.clip(X, 0, None)
    # log1p is now safe
    return np.log1p(X)


def make_safe_positive_pipeline(log_compress: bool = True):
    """
    Create a robust scaling pipeline for positive-valued features.

    Pipeline steps:
    1. Impute missing/inf values with median
    2. (Optional) Apply safe log compression for heavy-tailed distributions
    3. Quantile transform to uniform [0,1] distribution (outlier-robust)

    Parameters:
    -----------
    log_compress : bool, default=True
        Whether to apply log1p transformation before quantile scaling.
        Use True for heavy-tailed features (revenue, popularity, movie_age).
        Use False for bounded features (vote_average, duration).

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for transform()

    Examples:
    ---------
    >>> # For heavy-tailed features (revenue, popularity)
    >>> scaler = make_safe_positive_pipeline(log_compress=True)
    >>> scaler.fit(df[['revenue']])
    >>> normalized = scaler.transform(df[['revenue']])

    >>> # For bounded features (vote_average, duration)
    >>> scaler = make_safe_positive_pipeline(log_compress=False)
    >>> scaler.fit(df[['vote_average']])
    >>> normalized = scaler.transform(df[['vote_average']])

    Notes:
    ------
    - QuantileTransformer with uniform distribution is robust to outliers
    - Safe for data with infinities, missing values, or negative values
    - Maintains consistent random_state=0 for reproducibility
    """
    steps = []
    # 1. Impute missing / replaced inf values
    steps.append(("impute", SimpleImputer(strategy="median")))
    # 2. Optional safe log compress for heavy-tailed positives
    if log_compress:
        steps.append(("safe_log1p", FunctionTransformer(safe_log1p, validate=False)))
    # 3. Rank-based mapping to uniform [0,1], robust to outliers
    steps.append(("quantile", QuantileTransformer(output_distribution="uniform", random_state=0, copy=True)))
    return make_pipeline(*[step for _, step in steps])
