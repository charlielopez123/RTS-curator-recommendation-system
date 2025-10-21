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

from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.environments.TV_environment import TVProgrammingEnvironment


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
    ecdf_proxy = Line2D([0], [0], color="red", lw=1.6, label="ECDF (Empirical Cumulative Distribution Function)")
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


def _generate_random_contexts(
    n_samples: int,
    date_range: Tuple[str, str] = ('2024-01-01', '2024-12-31'),
    hour_range: Tuple[int, int] = (18, 23)
):
    """
    Generate random programming contexts for visualization.

    Args:
        n_samples: Number of random contexts to generate
        date_range: Tuple of (start_date, end_date) as 'YYYY-MM-DD' strings
        hour_range: Tuple of (min_hour, max_hour) for prime time

    Yields:
        Context instances
    """
    import pandas as pd
    from cts_recommender.environments.schemas import Context, Season, Channel
    from cts_recommender.utils.dates import get_season

    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])
    date_range_days = (end_date - start_date).days

    channels = [Channel.RTS1, Channel.RTS2]

    for _ in range(n_samples):
        # Random date
        random_days = np.random.randint(0, date_range_days + 1)
        random_date = start_date + pd.Timedelta(days=int(random_days))

        # Random hour and channel
        random_hour = np.random.randint(hour_range[0], hour_range[1] + 1)
        random_channel = np.random.choice(channels)

        # Get season using utility
        season_str = get_season(random_date.date())
        season = Season(season_str)

        yield Context(
            hour=random_hour,
            day_of_week=random_date.dayofweek,
            month=random_date.month,
            season=season,
            channel=random_channel
        )


def visualize_cts_signal_weight_distributions(
    cts_model: ContextualThompsonSampler,
    env: TVProgrammingEnvironment,
    signal_names: Optional[List[str]] = None,
    n_samples: int = 1000,
    date_range: Tuple[str, str] = ('2024-01-01', '2024-12-31'),
    hour_range: Tuple[int, int] = (9, 23),
    figsize: Tuple[int, int] = (15, 10)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize the distribution of CTS signal weights across artificial contexts.

    This shows how the Contextual Thompson Sampler weights different value signals
    (audience, competition, diversity, novelty, rights, curator acceptance) based on
    context features. Creates artificial contexts by sampling random dates/times to
    explore the full distribution of learned weights.

    Args:
        cts_model: Trained ContextualThompsonSampler instance
        env: TVProgrammingEnvironment instance (for converting contexts to features)
        signal_names: List of signal names for labeling. Default:
                     ['Audience', 'Competition', 'Diversity', 'Novelty', 'Rights', 'Curator']
        n_samples: Number of artificial contexts to generate (default: 1000)
        date_range: Tuple of (start_date, end_date) as 'YYYY-MM-DD' strings
        hour_range: Tuple of (min_hour, max_hour) for prime time slots
        figsize: Figure size (width, height)

    Returns:
        Tuple of (figure, axes array)

    Example:
        >>> from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
        >>> from cts_recommender.utils.visualization import visualize_cts_signal_weight_distributions
        >>>
        >>> # After training CTS model
        >>> fig, axes = visualize_cts_signal_weight_distributions(
        ...     cts_model=cts,
        ...     env=env,
        ...     n_samples=1000
        ... )
        >>> plt.show()
    """
    if signal_names is None:
        signal_names = ['Audience', 'Competition', 'Diversity', 'Novelty', 'Rights', 'Curator']

    # Generate random contexts and compute weights
    all_weights = []
    for context in _generate_random_contexts(n_samples, date_range, hour_range):
        # Get context features from environment
        context_features, _ = env.get_context_features(context)

        # Sample weights from CTS model
        weights = cts_model.thompson_sample_w(context_features)
        all_weights.append(weights)

    all_weights = np.array(all_weights)  # Shape: (n_samples, num_signals)

    # Create visualization with subplots for each signal
    num_signals = len(signal_names)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for i, name in enumerate(signal_names):
        ax = axes[i]
        weights = all_weights[:, i]

        # Histogram
        ax.hist(weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

        # Add statistics
        mean_val = weights.mean()
        median_val = np.median(weights)
        std_val = weights.std()
        min_val = weights.min()
        max_val = weights.max()

        # Vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.3f}')

        # Labels and title
        ax.set_xlabel('Weight Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name} Signal Weight\n(μ={mean_val:.3f}, σ={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}])',
                     fontsize=11)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    # Remove the 6th subplot if there are only 6 signals
    if num_signals <= 5:
        axes[5].axis('off')

    # Adjust layout to prevent title overlap - leave space at top for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


def plot_cts_training_history(
    history: Dict,
    figsize: Tuple[int, int] = (16, 12)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualize CTS warm-start training dynamics.

    Creates a multi-panel plot showing how parameters, gradients, curvatures,
    and noise evolve during warm-start training.

    Args:
        history: Training history dict returned by cts.warm_start(monitor_every=N)
        figsize: Figure size (width, height)

    Returns:
        Tuple of (figure, axes array)

    Example:
        >>> history = cts.warm_start(contexts, signals, rewards,
        ...                          epochs=5, monitor_every=500, verbose=True)
        >>> fig, axes = plot_cts_training_history(history)
        >>> plt.show()
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()

    steps = history['update_steps']
    epochs = history['epochs']

    # 1. Parameter norms
    ax = axes[0]
    ax.plot(steps, history['U_norm'], label='||U|| (weights)', linewidth=2)
    ax.plot(steps, history['b_norm'], label='||b|| (bias)', linewidth=2, linestyle='--')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('L2 Norm')
    ax.set_title('Parameter Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Parameter max values
    ax = axes[1]
    ax.plot(steps, history['U_max'], label='max|U|', linewidth=2)
    ax.plot(steps, history['b_max'], label='max|b|', linewidth=2, linestyle='--')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Max Absolute Value')
    ax.set_title('Parameter Max Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Loss
    ax = axes[2]
    ax.plot(steps, history['loss'], linewidth=2, color='red')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # 4. Gradient norms
    ax = axes[3]
    ax.plot(steps, history['grad_U_norm'], label='||grad_U||', linewidth=2)
    ax.plot(steps, history['grad_b_norm'], label='||grad_b||', linewidth=2, linestyle='--')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Gradient L2 Norm')
    ax.set_title('Gradient Magnitudes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Gradient max values
    ax = axes[4]
    ax.plot(steps, history['grad_U_max'], label='max|grad_U|', linewidth=2)
    ax.plot(steps, history['grad_b_max'], label='max|grad_b|', linewidth=2, linestyle='--')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Max Gradient')
    ax.set_title('Gradient Max Values')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Curvature (h_U) statistics
    ax = axes[5]
    ax.plot(steps, history['h_U_mean'], label='h_U mean', linewidth=2)
    ax.plot(steps, history['h_U_median'], label='h_U median', linewidth=2, linestyle='--')
    ax.axhline(y=history['h_U_mean'][0], color='gray', linestyle=':',
               label=f'h0={history["h_U_mean"][0]:.3f}')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Curvature h_U')
    ax.set_title('Curvature Estimates (h_U)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 7. Curvature range
    ax = axes[6]
    ax.plot(steps, history['h_U_max'], label='h_U max', linewidth=2)
    ax.plot(steps, history['h_U_min'], label='h_U min', linewidth=2, linestyle='--')
    ax.fill_between(steps, history['h_U_min'], history['h_U_max'], alpha=0.2)
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Curvature h_U')
    ax.set_title('Curvature Range (h_U)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 8. Noise standard deviation
    ax = axes[7]
    ax.plot(steps, history['noise_std_U'], label='noise_std(U)', linewidth=2)
    ax.plot(steps, history['noise_std_b'], label='noise_std(b)', linewidth=2, linestyle='--')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Noise Std Dev')
    ax.set_title('Thompson Sampling Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Signal-to-Noise Ratio
    ax = axes[8]
    # Compute SNR as parameter magnitude / noise magnitude
    snr_U = np.array(history['U_norm']) / (np.array(history['noise_std_U']) * np.sqrt(len(history['U_norm'])))
    ax.plot(steps, snr_U, linewidth=2, color='purple')
    ax.axhline(y=3, color='green', linestyle='--', linewidth=2, label='SNR=3 (target)')
    ax.set_xlabel('Update Steps')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('SNR: Parameter Signal vs Noise')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('CTS Warm-Start Training Diagnostics', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])

    return fig, axes


def print_cts_training_summary(history: Dict) -> None:
    """
    Print summary statistics from CTS training history.

    Args:
        history: Training history dict returned by cts.warm_start(monitor_every=N)

    Example:
        >>> history = cts.warm_start(contexts, signals, rewards,
        ...                          epochs=5, monitor_every=500)
        >>> print_cts_training_summary(history)
    """
    print("=" * 70)
    print("CTS WARM-START TRAINING SUMMARY")
    print("=" * 70)

    print(f"\nTraining Progress:")
    print(f"  Total updates: {history['update_steps'][-1]:,}")
    print(f"  Epochs completed: {history['epochs'][-1]:.2f}")
    print(f"  Monitoring points: {len(history['update_steps'])}")

    print(f"\nParameter Evolution:")
    print(f"  ||U|| : {history['U_norm'][0]:.6f} → {history['U_norm'][-1]:.6f} "
          f"(change: {history['U_norm'][-1] - history['U_norm'][0]:+.6f})")
    print(f"  ||b|| : {history['b_norm'][0]:.6f} → {history['b_norm'][-1]:.6f} "
          f"(change: {history['b_norm'][-1] - history['b_norm'][0]:+.6f})")
    print(f"  max|U|: {history['U_max'][0]:.6f} → {history['U_max'][-1]:.6f}")
    print(f"  max|b|: {history['b_max'][0]:.6f} → {history['b_max'][-1]:.6f}")

    print(f"\nCurvature (h_U) Evolution:")
    print(f"  Initial h_U (h0): {history['h_U_mean'][0]:.6f}")
    print(f"  Final h_U mean:   {history['h_U_mean'][-1]:.6f} "
          f"({'↑' if history['h_U_mean'][-1] > history['h_U_mean'][0] else '↓'} "
          f"{abs(history['h_U_mean'][-1] / history['h_U_mean'][0] - 1) * 100:.1f}%)")
    print(f"  Final h_U median: {history['h_U_median'][-1]:.6f}")
    print(f"  Final h_U range:  [{history['h_U_min'][-1]:.6f}, {history['h_U_max'][-1]:.6f}]")

    print(f"\nGradient Statistics (final epoch):")
    print(f"  ||grad_U||: {history['grad_U_norm'][-1]:.6f}")
    print(f"  ||grad_b||: {history['grad_b_norm'][-1]:.6f}")
    print(f"  max|grad_U|: {history['grad_U_max'][-1]:.6f}")
    print(f"  max|grad_b|: {history['grad_b_max'][-1]:.6f}")

    print(f"\nNoise Levels:")
    print(f"  Initial noise_std(U): {history['noise_std_U'][0]:.6f}")
    print(f"  Final noise_std(U):   {history['noise_std_U'][-1]:.6f}")

    # Compute final SNR
    final_snr = history['U_norm'][-1] / (history['noise_std_U'][-1] * np.sqrt(len(history['U_norm'])))
    print(f"\nSignal-to-Noise Ratio:")
    print(f"  Final SNR: {final_snr:.2f} {'✅' if final_snr >= 3 else '⚠️'} "
          f"(target: ≥3.0)")

    print(f"\nTraining Loss:")
    print(f"  Initial: {history['loss'][0]:.6f}")
    print(f"  Final:   {history['loss'][-1]:.6f}")
    print(f"  Reduction: {(1 - history['loss'][-1] / history['loss'][0]) * 100:.1f}%")

    # Check for h_U decay issue
    if history['h_U_mean'][-1] < history['h_U_mean'][0] * 0.5:
        print(f"\n⚠️  WARNING: h_U decreased significantly during training!")
        print(f"   This indicates gradients are smaller than h0.")
        print(f"   Consider: (1) decreasing h0, (2) using faster ema_decay, or (3) increasing lr")

    print("=" * 70)
