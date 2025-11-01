"""
Main file
======================

This file serves as the primary entry point for running Reinforcement Learning
experiments with the modular architecture defined in this project. It loads
the experiment configuration, and initializes all required components (environment,
agent, logging utilities).

# Main responsibilities:
- load experiment configuration (gamma, alpha, epsilon, episodes, seeds),
- launch training or demo mode depending on CLI arguments,
- ensure no duplicated hyperparameter references outside config.py,
- provide a clean entry interface for experimentation.

Author: Calin Dragos George
Created: 26 October 2025
"""

import sys
import gymnasium as gym

from config import gamma, alpha, epsilon_start, epsilon_end, epsilon_decay, episodes, env_id, seed
from run_experiment import run_training, run_demo
from q_learning_agent import QLearningAgent
from cartpole_discretizer import CartPoleDiscretizer


def print_config():
    """Display current hyperparameters before training."""
    print("\n=== Experiment Configuration ===")
    print(f"Environment: {env_id}")
    print(f"gamma (discount): {gamma}")
    print(f"alpha (learning rate): {alpha}")
    print(f"epsilon_start: {epsilon_start} | epsilon_end: {epsilon_end} | epsilon_decay: {epsilon_decay}")
    print(f"episodes: {episodes}")
    print(f"seed: {seed}")
    print("================================\n")


def main():
    """
    Entry point:
    - Default mode: Training
    - '--demo' launches a playback of trained Q-table
      for the gamma value currently set in config.py
    """

    if "--demo" in sys.argv:
        print("\nLaunching DEMO using trained Q-table...\n")

        # create environment for demo
        env = gym.make(env_id, render_mode="human")

        # setup discretizer
        discretizer = CartPoleDiscretizer(
            cart_pos_bins=[-2.4, 0.0, +2.4],
            cart_vel_bins=[-1.0, 0.0, +1.0],
            pole_ang_bins=[-0.2095, -0.05, +0.05, +0.2095],
            pole_vel_bins=[-1.0, 0.0, +1.0]
        )

        # create dummy agent w/ correct Q-table shape
        q_shape = discretizer.get_state_shape()
        agent = QLearningAgent(
            action_space=env.action_space,
            q_table_shape=q_shape,
            alpha=alpha,
            gamma=gamma,
            epsilon=0.0
        )

        # load Q-table for gamma currently set in config.py
        model_path = f"models/q_table_gamma_{gamma:.3f}.npy"
        agent.load(model_path)

        run_demo(agent, env, discretizer)

    else:
        print_config()
        run_training()


if __name__ == "__main__":
    main()
