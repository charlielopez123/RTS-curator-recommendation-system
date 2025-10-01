"""
Utilities for ensuring reproducible results across the project.
"""

import os
import random
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure reproducible results.

    Args:
        seed: Random seed value to use across all libraries
    """
    logger.info(f"Setting global random seed to {seed}")

    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)

    # scikit-learn uses NumPy's random state by default, but we can be explicit
    try:
        from sklearn.utils import check_random_state
        check_random_state(seed)
    except ImportError:
        pass

    logger.info("Global random seed set successfully")


def get_reproducible_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """
    Get a reproducible random state for use in functions that accept random_state.

    Args:
        seed: Seed for the random state. If None, uses default seed.

    Returns:
        numpy.random.RandomState: Seeded random state
    """
    if seed is None:
        seed = 42
    return np.random.RandomState(seed)


class ReproducibleContext:
    """
    Context manager for temporarily setting a specific seed within a block.

    Example:
        with ReproducibleContext(123):
            # All random operations here use seed 123
            train_test_split(X, y, random_state=123)
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.original_states = {}

    def __enter__(self):
        # Save current states
        self.original_states['random'] = random.getstate()
        self.original_states['numpy'] = np.random.get_state()

        # Set new seed
        set_global_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original states
        random.setstate(self.original_states['random'])
        np.random.set_state(self.original_states['numpy'])