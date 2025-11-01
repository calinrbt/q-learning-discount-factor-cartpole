"""
TensorBoard Logging Utility Module
==================================

This file centralizes all TensorBoard logging operations for reinforcement
learning experiments using Gymnasium environments. It provides a clean and
reusable interface to record performance metrics (e.g., episode rewards,
episode lengths, hyperparameter settings) across different Q-Learning runs.

# Main responsibilities:
- create and manage TensorBoard SummaryWriter instances,
- log numerical metrics per episode (reward, stability signals, etc.),
- log experiment metadata such as gamma, alpha, epsilon, seed,
- ensure consistent logging directory structure across experiments,
- support comparisons in TensorBoard UI for hyperparameter sweeps.

# Dependencies:
- torch.utils.tensorboard (SummaryWriter)
- os / pathlib for directory management

Author: Calin Dragos George
Created: 26 October 2025
"""
import os
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    manages a tensorboard summarywriter instance.
    helps keep the training loop clean and modular.
    """

    def __init__(self, log_dir):
        # make sure the log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # create a tensorboard writer pointing to that directory
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_reward(self, episode, reward):
        """
        log the episode reward to tensorboard.
        this allows you to visualize learning progress over time.
        """
        self.writer.add_scalar("Episode Reward", reward, episode)

    def log_parameter(self, name, value):
        """
        optional: log hyperparameter values (e.g., gamma, alpha, epsilon)
        so you can later identify which run produced which curve.
        """
        self.writer.add_text(name, str(value))

    def close(self):
        """
        close the writer when training ends.
        this ensures all logs are properly saved to disk.
        """
        self.writer.close()