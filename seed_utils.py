"""
Random Seed Utility Module
==========================

This file provides helper functions to ensure reproducibility across
Reinforcement Learning experiments. Since Gymnasium environments, numpy
operations, and PyTorch TensorBoard logging may all introduce randomness,
explicit seeding is required to make results consistent and comparable.

# Main responsibilities:
- set global random seeds for numpy and Python's random module,
- optionally apply seeding to Gymnasium environments,
- support multi-seed experiment workflows,
- provide a straightforward interface for standardized seeding across all
  hyperparameter tests (gamma, alpha, epsilon sweeps).

# Dependencies:
- numpy (for random seed)
- random (Python RNG)

Author: Calin Dragos George
Created: 26 October 2025
"""

import numpy as np
import random


def set_global_seed(seed):
    """
    set the random seed for both NumPy and Python's random module.
    this ensures exploration behavior and state transitions remain
    the same across runs when using the same seed.
    """
    np.random.seed(seed)
    random.seed(seed)


def set_env_seed(env, seed):
    """
    apply seed to the Gymnasium environment.
    this controls randomness in:
    - initial states at env.reset()
    - any internal stochastic behavior
    """
    env.reset(seed=seed)