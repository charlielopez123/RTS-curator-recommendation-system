"""
Contextual Thompson Sampler Warm-Start Training Pipeline

Orchestrates the warm-start training of the CTS model using historical programming data
processed through the imitation learning pipeline. This pipeline:

1. Loads training data from IL pipeline (context features, value signals, rewards)
2. Initializes CTS model with hyperparameters from docs/CTS_HYPERPARAMETERS.md
3. Sets target weight initialization based on curator objectives (gamma=0.3)
4. Performs warm-start training on historical data
5. Saves trained model for online deployment

The warm-start process pre-trains the CTS model on historical curator decisions,
initializing the parameters before online learning. This provides a strong starting
point that reflects past programming patterns and curator objectives.

Key Configuration:
- Uses gamma=0.3 for weight initialization (30% curator signal, 70% value signals)
- Warm-start hyperparameters: lr=0.01, expl_scale=1e-4, ema_decay=0.9999
- Online hyperparameters: lr=0.05, expl_scale=0.001, ema_decay=0.999
- Training runs for 1 epoch through historical data
- Model learns to weight signals (audience, competition, diversity, novelty, rights, curator_acceptance)
  based on context (time, channel, season)

Output:
- Trained CTS model saved as .npz (parameters) and .rng.pkl (random state)
- Can be loaded for online recommendation or further training
"""

from pathlib import Path
import numpy as np
import logging
import joblib
from typing import Optional

from cts_recommender.models.contextual_thompson_sampler import ContextualThompsonSampler
from cts_recommender.imitation_learning.IL_constants import PSEUDO_REWARD_WEIGHTS
from cts_recommender.settings import get_settings

cfg = get_settings()
logger = logging.getLogger(__name__)


def _extract_signal_matrix(training_samples: list) -> np.ndarray:
    """
    Extract signal matrix from training samples.

    Returns array of shape (n_samples, 6) with columns:
    [audience, competition, diversity, novelty, rights, curator_acceptance]
    """
    signal_matrix = []
    for sample in training_samples:
        vs = sample['value_signals']
        row = [
            vs['audience'],
            vs['competition'],
            vs['diversity'],
            vs['novelty'],
            vs['rights'],
            float(sample['selected'])
        ]
        signal_matrix.append(row)
    return np.array(signal_matrix)


def run_CTS_warmstart_training_pipeline(
    training_data_file: Optional[Path] = None,
    model_output_file: Optional[Path] = None,
    gamma: float = 0.3,
    lr_warmstart: float = 0.01,
    lr_online: float = 0.05,
    expl_scale_warmstart: float = 1e-4,
    expl_scale_online: float = 0.001,
    ema_decay_warmstart: float = 0.9999,
    ema_decay_online: float = 0.999,
    h0: float = 0.1,
    tau: float = 2.0,
    alpha: float = 0.3,
    weight_decay: float = 1e-4,
    match_loss_weight: float = 0.5,
    epochs: int = 1,
    monitor_every: Optional[int] = 500,
    verbose: bool = True
) -> tuple[ContextualThompsonSampler, Path]:
    """
    Train Contextual Thompson Sampler with warm-start on historical data.

    This pipeline initializes and trains the CTS model using historical programming
    decisions from the imitation learning pipeline. The model learns context-dependent
    signal weights for recommendation.

    Parameters:
    -----------
    training_data_file : Path, optional
        Path to IL training data (.joblib)
        Default: data/processed/IL/training_data.joblib

    model_output_file : Path, optional
        Path to save trained CTS model (.npz + .rng.pkl)
        Default: data/models/cts_model.npz

    gamma : float, default=0.3
        Curator signal weight for initialization (30% curator, 70% value signals)
        Changed from gamma=0.6 used in IL data generation to better align with
        curator decision-making priorities

    lr_warmstart : float, default=0.01
        Learning rate for warm-start training (lower for stability)

    lr_online : float, default=0.05
        Learning rate for online learning after warm-start (higher for adaptation)

    expl_scale_warmstart : float, default=1e-4
        Exploration noise scale during warm-start (10x lower than online)

    expl_scale_online : float, default=0.001
        Exploration noise scale for online Thompson sampling

    ema_decay_warmstart : float, default=0.9999
        EMA decay for Hessian during warm-start (even slower for stability)

    ema_decay_online : float, default=0.999
        EMA decay for Hessian during online learning

    h0 : float, default=0.1
        Initial Hessian diagonal value (controls starting exploration magnitude)

    tau : float, default=2.0
        Softmax temperature (higher = more uniform weights)

    alpha : float, default=0.3
        Uniform mixing weight (prevents signal weight collapse, min 5% per signal)

    weight_decay : float, default=1e-4
        L2 regularization strength

    match_loss_weight : float, default=0.5
        Weight for signal matching loss (when curator signals provided)

    epochs : int, default=1
        Number of passes through training data

    monitor_every : int, optional
        Record diagnostics every N updates (None = no monitoring)
        Default: 500

    verbose : bool, default=True
        Print training progress

    Returns:
    --------
    tuple[ContextualThompsonSampler, Path]
        Trained CTS model and output file path

    Example:
    --------
    >>> cts, model_path = run_CTS_warmstart_training_pipeline(
    ...     gamma=0.3,
    ...     lr_warmstart=0.01,
    ...     epochs=1,
    ...     verbose=True
    ... )
    >>> # Model is saved and ready for online deployment
    >>> loaded_cts = ContextualThompsonSampler.load(model_path)
    """
    # Set default paths
    if training_data_file is None:
        training_data_file = cfg.processed_dir / "IL" / "training_data.joblib"
    if model_output_file is None:
        model_output_file = cfg.models_dir / "cts_model.npz"

    logger.info("=== Contextual Thompson Sampler Warm-Start Training Pipeline ===")
    logger.info(f"Training data: {training_data_file}")
    logger.info(f"Model output: {model_output_file}")
    logger.info(f"\nHyperparameters:")
    logger.info(f"  Gamma (curator weight): {gamma}")
    logger.info(f"  Learning rate (warm-start): {lr_warmstart}")
    logger.info(f"  Learning rate (online): {lr_online}")
    logger.info(f"  Exploration scale (warm-start): {expl_scale_warmstart}")
    logger.info(f"  Exploration scale (online): {expl_scale_online}")
    logger.info(f"  EMA decay (warm-start): {ema_decay_warmstart}")
    logger.info(f"  EMA decay (online): {ema_decay_online}")
    logger.info(f"  Initial Hessian (h0): {h0}")
    logger.info(f"  Temperature (tau): {tau}")
    logger.info(f"  Uniform mixing (alpha): {alpha}")
    logger.info(f"  Weight decay: {weight_decay}")
    logger.info(f"  Match loss weight: {match_loss_weight}")
    logger.info(f"  Epochs: {epochs}")

    # Load training samples (contains everything we need)
    training_samples_file = training_data_file.parent / "training_samples.joblib"
    logger.info("\nLoading training samples...")
    logger.info(f"From: {training_samples_file}")
    training_samples = joblib.load(training_samples_file)
    logger.info(f"Loaded {len(training_samples)} training samples")

    # Extract arrays from training samples
    logger.info("\nExtracting features from training samples...")
    context_features = np.array([s['context_features'] for s in training_samples])
    reward_targets = np.array([s['reward'] for s in training_samples])
    train_signals = _extract_signal_matrix(training_samples)

    # Extract dimensions
    num_samples, context_dim = context_features.shape
    num_signals = train_signals.shape[1]

    logger.info(f"\nData dimensions:")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Context dimension: {context_dim}")
    logger.info(f"  Number of signals: {num_signals}")
    logger.info(f"  Signal ordering: [audience, competition, diversity, novelty, rights, curator_acceptance]")

    # Initialize CTS model with online hyperparameters
    logger.info("\nInitializing Contextual Thompson Sampler...")
    cts = ContextualThompsonSampler(
        num_signals=num_signals,
        context_dim=context_dim,
        lr=lr_online,                    # Will be overridden during warm-start
        expl_scale=expl_scale_online,    # Will be overridden during warm-start
        ema_decay=ema_decay_online,      # Will be overridden during warm-start
        h0=h0,
        tau=tau,
        alpha=alpha,
        weight_decay=weight_decay,
        match_loss_weight=match_loss_weight,
        random_state=cfg.random_seed
    )
    logger.info(f"CTS initialized with {num_signals} signals and {context_dim} context features")

    # Initialize weights based on curator objectives
    logger.info("\nInitializing target weights...")
    logger.info(f"Curator signal weight (gamma): {gamma}")

    # Calculate target weights
    pseudo_weights = np.array([
        PSEUDO_REWARD_WEIGHTS['audience'],      # 0.4
        PSEUDO_REWARD_WEIGHTS['competition'],   # 0.15
        PSEUDO_REWARD_WEIGHTS['diversity'],     # 0.1
        PSEUDO_REWARD_WEIGHTS['novelty'],       # 0.2
        PSEUDO_REWARD_WEIGHTS['rights']         # 0.15
    ])

    target_weights = np.zeros(num_signals)
    target_weights[:5] = (1 - gamma) * pseudo_weights  # Value signals: 70%
    target_weights[5] = gamma                          # Curator signal: 30%

    logger.info("Target weight distribution:")
    signal_names = ['Audience', 'Competition', 'Diversity', 'Novelty', 'Rights', 'Curator']
    for i, (name, weight) in enumerate(zip(signal_names, target_weights)):
        logger.info(f"  {name}: {weight:.4f} ({100*weight:.1f}%)")

    cts.initialize_with_target_weights(target_weights)
    logger.info("Target weights initialized")

    # Warm-start training
    logger.info("\n" + "="*70)
    logger.info("Starting warm-start training...")
    logger.info("="*70)

    history = cts.warm_start(
        contexts=context_features,
        signals=train_signals,
        rewards=reward_targets,
        epochs=epochs,
        lr=lr_warmstart,
        expl_scale=expl_scale_warmstart,
        ema_decay=ema_decay_warmstart,
        weight_decay=weight_decay,
        monitor_every=monitor_every,
        verbose=verbose
    )

    logger.info("="*70)
    logger.info("Warm-start training completed")
    logger.info("="*70)

    # Log final model statistics
    logger.info("\nFinal model statistics:")
    logger.info(f"  Parameter norm |U|: {np.linalg.norm(cts.U):.4f}")
    logger.info(f"  Bias norm |b|: {np.linalg.norm(cts.b):.4f}")
    logger.info(f"  Max |U|: {np.abs(cts.U).max():.4f}")
    logger.info(f"  Max |b|: {np.abs(cts.b).max():.4f}")
    logger.info(f"  Hessian U median: {np.median(cts.h_U):.6f}")
    logger.info(f"  Hessian b median: {np.median(cts.h_b):.6f}")
    logger.info(f"  Exploration noise std U: {np.sqrt(cts.expl_scale / np.median(cts.h_U)):.6f}")
    logger.info(f"  Exploration noise std b: {np.sqrt(cts.expl_scale / np.median(cts.h_b)):.6f}")

    # Ensure output directory exists
    model_output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save trained model
    logger.info(f"\nSaving trained model to {model_output_file}...")
    cts.save(model_output_file)
    logger.info(f"Model saved successfully")
    logger.info(f"  Parameters: {model_output_file}")
    logger.info(f"  RNG state: {model_output_file.with_suffix('.rng.pkl')}")

    # Save training history if monitoring was enabled
    if history is not None:
        history_file = model_output_file.parent / f"{model_output_file.stem}_history.joblib"
        logger.info(f"\nSaving training history to {history_file}...")
        joblib.dump(history, history_file)
        logger.info("Training history saved")

    return cts, model_output_file
