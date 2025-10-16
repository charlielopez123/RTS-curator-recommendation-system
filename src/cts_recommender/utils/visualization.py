"""
Visualization utilities for reward signal distributions and IL training analysis.

Provides reusable plotting functions for analyzing training samples and reward components.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple


def clean_01(x: np.ndarray) -> np.ndarray:
    """
    Clean and clip data to [0, 1] range, removing non-finite values.

    Args:
        x: Input array

    Returns:
        Cleaned array clipped to [0, 1]
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.clip(x, 0, 1)


def freedman_diaconis_bins(x: np.ndarray) -> int:
    """
    Calculate optimal number of bins using Freedman-Diaconis rule.

    The Freedman-Diaconis rule is robust to outliers and works well
    for a wide range of distributions.

    Args:
        x: Data array

    Returns:
        Optimal number of bins
    """
    n = x.size
    if n < 2:
        return 1

    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25

    if iqr == 0:
        return min(30, max(1, int(np.sqrt(n))))

    h = 2 * iqr * n**(-1/3)
    if h <= 0:
        return min(30, max(1, int(np.sqrt(n))))

    return max(1, int(np.ceil((x.max() - x.min()) / h)))


def plot_reward_distributions(
    signals: Dict[str, List[float]],
    quantiles: Optional[List[Tuple[int, str, str]]] = None,
    figsize: Tuple[int, int] = (13, 7),
    title: str = "Reward Signal Distributions"
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a multi-panel visualization of reward signal distributions.

    Generates histograms (log-scale density) + ECDF plots for each signal,
    with quantile markers and a legend panel.

    Args:
        signals: Dictionary mapping signal names to value lists
                e.g., {"Audience": [0.1, 0.2, ...], "Competition": [...]}
        quantiles: List of (percentile, color, label) tuples for quantile markers
                  Default: [(5, "tab:green", "5th percentile"),
                           (50, "tab:purple", "Median"),
                           (95, "tab:orange", "95th percentile")]
        figsize: Figure size (width, height)
        title: Overall figure title

    Returns:
        Tuple of (figure, axes array)

    Example:
        >>> signals = {
        ...     "Audience": reward_audience,
        ...     "Competition": reward_competition,
        ...     "Diversity": reward_diversity,
        ...     "Novelty": reward_novelty,
        ...     "Rights": reward_rights,
        ... }
        >>> fig, axes = plot_reward_distributions(signals)
        >>> plt.show()
    """
    if quantiles is None:
        quantiles = [
            (5,  "tab:green",  "5th percentile"),
            (50, "tab:purple", "Median"),
            (95, "tab:orange", "95th percentile"),
        ]

    # Create figure with 6 subplots (5 signals + 1 legend)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    # Plot each signal
    for i, (name, values) in enumerate(signals.items()):
        ax = axes[i]
        x = clean_01(values)

        if x.size == 0:
            ax.set_title(f"{name} (empty)")
            ax.axis("off")
            continue

        # Histogram with log scale for density
        bins = freedman_diaconis_bins(x)
        ax.hist(x, bins=bins, density=True, alpha=0.7)
        ax.set_yscale("log")

        # ECDF on twin axis (red)
        xs = np.sort(x)
        ys = np.arange(1, xs.size + 1) / xs.size
        ax2 = ax.twinx()
        ax2.plot(xs, ys, linewidth=1.6, color="red")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("ECDF")

        # Quantile lines with color coding
        for q, color, _ in quantiles:
            qv = np.percentile(x, q)
            ax.axvline(qv, linestyle="--", linewidth=1.2, color=color)

        # Axes formatting
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_xlabel("score")
        ax.set_ylabel("density")
        ax.grid(True, linewidth=0.5, alpha=0.5)

        # Plain decimal x ticks
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))

        # ECDF y-axis ticks
        ax2.yaxis.set_major_locator(MultipleLocator(0.2))
        ax2.yaxis.set_minor_locator(MultipleLocator(0.1))

    # Legend in the 6th subplot
    legend_ax = axes[-1]
    legend_ax.axis("off")

    hist_proxy = Patch(alpha=0.7, label="Histogram (density)")
    ecdf_proxy = Line2D([0], [0], color="red", lw=1.6, label="ECDF")
    q_handles = [
        Line2D([0], [0], color=c, lw=1.2, ls="--", label=lab)
        for _, c, lab in quantiles
    ]

    legend_ax.legend(
        handles=[hist_proxy, ecdf_proxy, *q_handles],
        loc="center",
        frameon=False,
        title="Legend"
    )

    # Extra spacing between subplots
    fig.subplots_adjust(
        left=0.06, right=0.98, top=0.92, bottom=0.08,
        wspace=0.5, hspace=0.45
    )
    plt.suptitle(title, fontsize=14, y=0.98)

    return fig, axes


def extract_reward_signals(training_samples: List[Dict]) -> Dict[str, List[float]]:
    """
    Extract reward signal arrays from training samples.

    Args:
        training_samples: List of training sample dictionaries with 'value_signals' key

    Returns:
        Dictionary mapping signal names to value lists

    Example:
        >>> training_samples = joblib.load("training_samples.joblib")
        >>> signals = extract_reward_signals(training_samples)
        >>> print(signals.keys())
        dict_keys(['audience', 'competition', 'diversity', 'novelty', 'rights'])
    """
    reward_signals = {
        'audience': [],
        'competition': [],
        'diversity': [],
        'novelty': [],
        'rights': []
    }

    for sample in training_samples:
        value_signals = sample['value_signals']
        for key in reward_signals.keys():
            reward_signals[key].append(value_signals[key])

    return reward_signals


def print_reward_statistics(signals: Dict[str, List[float]]) -> None:
    """
    Print summary statistics for reward signals.

    Args:
        signals: Dictionary mapping signal names to value lists
    """
    print(f"Extracted {len(next(iter(signals.values())))} reward signals\n")

    for name, values in signals.items():
        values_arr = np.array(values)
        print(f"{name.capitalize():12s} - "
              f"range: [{values_arr.min():.3f}, {values_arr.max():.3f}], "
              f"mean: {values_arr.mean():.3f}, "
              f"median: {np.median(values_arr):.3f}")
