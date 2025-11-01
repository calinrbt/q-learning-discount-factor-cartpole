"""
Experiment Configuration Module
===============================
This file defines configurable hyperparameters and settings used across 
reinforcement learning experiments. Instead of hardcoding alpha, gamma,
epsilon, number of episodes, or seed values inside the training scripts,
this module centralizes them to ensure reproducibility, maintainability,
and clean separation between experiment logic and RL implementation.

# Main responsibilities:
- provide a single source of truth for hyperparameter settings,
- define lists or ranges for sweep experiments (e.g., multiple gamma values),
- store global experiment settings such as environment ID, number of seeds,
  maximum episodes, and epsilon decay configuration
- enable programmatic loading of different experiment profiles for tutorials,
  research tests, and comparative studies.

# Why this matters:
- hyperparameter tuning is essential for Q-Learning performance,
- keeping them external improves flexibility and supports automated experiments.

# Dependencies:
- None (basic Python only)

Author: Calin Dragos George
Created: 26 October 2025
"""


# discount factor (gamma) {0.100, 0.900, 0.950, 0.990, 0.995, 0.998, 1.100}
# -----------------------
# controls how much the agent values future rewards compared to immediate rewards.
gamma = 0.998


# learning rate (alpha)
# ---------------------
# controls how fast the Q-values are updated. Too high -> unstable learning,
# too low -> very slow progress. For Q-learning on CartPole, 0.1 is a common starting point.
alpha = 0.1


# exploration rate (epsilon)
# --------------------------
# probability of selecting a random action during training (exploration).
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995


# number of training episodes
# ---------------------------
# each episode is one full run from reset() to termination.
# 5000 is useful for debugging and verifying logging/TensorBoard,
episodes = 5000


# environment id (gymnasium)
# --------------------------
env_id = "CartPole-v1"


# random seed
# -----------
# Ensures reproducible training results during initial testing.
# Multi-seed support will be added later for rigorous scientific evaluation.
seed = 0