"""
Experiment Execution Module
===========================

This file coordinates the full reinforcement learning training loop using a
selected RL agent - Q-Learning Agent and a Gymnasium environment. It
handles episode iteration, environment interaction, logging, and experiment
lifecycle management. The goal is to separate the RL logic (inside the agent)
from the procedural workflow required to run and track experiments.

# Main responsibilities:
- initialize environment, agent, logger, and experiment configuration,
- execute multiple training episodes, collecting interaction data,
- handle stepping through Gymnasium environment: reset → step(action) → terminal,
- log performance metrics (e.g., episode rewards) using TensorBoard,
- save trained Q-Learning model for later use,
- provide hooks for multi-seed and hyperparameter sweep experiments.

# Dependencies:
- Gymnasium for environment interface,
- RL Agent class (provided externally),
- TensorBoard logger module,
- configuration module for hyperparameters,
- seed utilities for reproducibility.

Author: Calin Dragos George
Created: 26 October 2025
"""

import os
import gymnasium as gym
import numpy as np

from config import gamma, alpha, epsilon_start, epsilon_end, epsilon_decay, episodes, env_id, seed
from q_learning_agent import QLearningAgent
from cartpole_discretizer import CartPoleDiscretizer
from tensorboard_logger import TensorBoardLogger
from seed_utils import set_global_seed, set_env_seed


def run_training():
    """
    main training execution function.
    - creates environment and agent
    - runs the episode loop
    - logs reward values to TensorBoard
    - saves Q-table model using gamma from config.py
    """

    # set reproducible random behavior
    set_global_seed(seed)

    # create environment
    env = gym.make(env_id)

    # apply seed to environment
    set_env_seed(env, seed)

    # discretizer setup
    discretizer = CartPoleDiscretizer(
        cart_pos_bins=[-2.4, 0.0, +2.4],
        cart_vel_bins=[-1.0, 0.0, +1.0],
        pole_ang_bins=[-0.2095, -0.05, +0.05, +0.2095],
        pole_vel_bins=[-1.0, 0.0, +1.0]
    )

    # q-table shape determined from state bins
    q_table_shape = discretizer.get_state_shape()

    # RL agent
    agent = QLearningAgent(
        action_space=env.action_space,
        q_table_shape=q_table_shape,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start
    )

    # logger setup
    log_dir = f"logs/gamma_{gamma:.3f}/"
    logger = TensorBoardLogger(log_dir=log_dir)

    # log experiment metadata
    logger.log_parameter("gamma", gamma)
    logger.log_parameter("alpha", alpha)
    logger.log_parameter("epsilon", epsilon_start)
    logger.log_parameter("seed", seed)

    print("\nTraining started...\n")

    # training loop
    for ep in range(episodes):
        state, _ = env.reset()
        state = discretizer.discretize(state)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretizer.discretize(next_state)
            done = terminated or truncated
            episode_reward += reward

            agent.update(state, action, reward, next_state, done)
            state = next_state

        logger.log_reward(ep, episode_reward)
        print(f"Episode {ep} | Reward: {episode_reward}")
        # epsilon decay after each episode
        agent.epsilon = max(epsilon_end, agent.epsilon * epsilon_decay)

    print(f"\nTraining finished! Log directory: {log_dir}\n")
    print("To view logs, run:")
    print("tensorboard --logdir logs")
    print("Then open http://localhost:6006 in your browser\n")

    logger.close()

    # save Q-table model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/q_table_gamma_{gamma:.3f}.npy"
    agent.save(model_path)

    print(f"Model saved to: {model_path}")

    return agent, env, discretizer


def run_demo(agent, env, discretizer, episodes=30):
    """
    run demo using pure exploitation (no exploration)
    to visually inspect learned policy behavior
    after training has completed.
    """

    print("\n=== DEMO MODE: Exploit Only (No Random Actions) ===\n")

    for ep in range(episodes):
        state, _ = env.reset()
        state = discretizer.discretize(state)
        done = False
        episode_reward = 0

        while not done:
            action = np.argmax(agent.q_table[state])  # exploit best action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretizer.discretize(next_state)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            env.render()

        print(f"[DEMO] Episode {ep} | Reward: {episode_reward}")

    env.close()
