"""
CartPole State Discretizer Module
=================================

This file provides utilities for converting the continuous state space
of the CartPole-v1 environment into a discrete representation suitable
for tabular Q-Learning. Since CartPole observations consist of floating-point 
values that vary continuously, discretization is required in order to index 
Q-values efficiently in a Q-table.

# Main responsibilities:
- convert continuous environment observations into discrete state indices,
- define configurable discretization bins for each state dimension,
- ensure state indices always remain in valid bounds via clamping,
- provide a clean interface to be reused across multiple hyperparameter
  experiments and comparison studies (alpha, gamma, epsilon).

# Dependencies:
- numpy (for digitization and clipping)
- gymnasium environment (provides observation structure: [cart_pos, cart_vel, pole_ang, pole_vel])

# Notes:
- CartPole-v1 provides a 4-dimensional continuous observation space,
- proper discretization can significantly affect learning stability and performance,
- this module can be extended to support different bin configurations.

Author: Calin Dragos George
Created: 26 October 2025
"""

import numpy as np

class CartPoleDiscretizer:
    """
    converts the continuous cartpole state into discrete state bins.
    this is necessary for q-learning because q-tables work only with integers.
    """

    def __init__(self, cart_pos_bins, cart_vel_bins, pole_ang_bins, pole_vel_bins):
        # save all discretization bin configurations
        self.cart_pos_bins = cart_pos_bins
        self.cart_vel_bins = cart_vel_bins
        self.pole_ang_bins = pole_ang_bins
        self.pole_vel_bins = pole_vel_bins

        # store number of bins for each dimension (used for q-table shape)
        self.n_cart_pos = len(cart_pos_bins) + 1
        self.n_cart_vel = len(cart_vel_bins) + 1
        self.n_pole_ang = len(pole_ang_bins) + 1
        self.n_pole_vel = len(pole_vel_bins) + 1

    def get_state_shape(self):
        """
        return the shape of the discrete state space.
        used for initializing the q-table.
        """
        return (
            self.n_cart_pos,
            self.n_cart_vel,
            self.n_pole_ang,
            self.n_pole_vel
        )

    def discretize(self, observation):
        """
        convert continuous observation values into discrete state indexes.
        each value is digitized and safely clamped within valid bounds.
        """
        cart_pos, cart_vel, pole_ang, pole_vel = observation

        cart_pos_idx = np.digitize(cart_pos, self.cart_pos_bins)
        cart_vel_idx = np.digitize(cart_vel, self.cart_vel_bins)
        pole_ang_idx = np.digitize(pole_ang, self.pole_ang_bins)
        pole_vel_idx = np.digitize(pole_vel, self.pole_vel_bins)

        # clamp indexes to stay within valid q-table bounds
        cart_pos_idx = np.clip(cart_pos_idx, 0, self.n_cart_pos - 1)
        cart_vel_idx = np.clip(cart_vel_idx, 0, self.n_cart_vel - 1)
        pole_ang_idx = np.clip(pole_ang_idx, 0, self.n_pole_ang - 1)
        pole_vel_idx = np.clip(pole_vel_idx, 0, self.n_pole_vel - 1)

        return (
            cart_pos_idx,
            cart_vel_idx,
            pole_ang_idx,
            pole_vel_idx
        )