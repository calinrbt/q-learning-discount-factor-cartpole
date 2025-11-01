"""
Q-Learning Agent Module
=======================

This file implements a Q-Learning agent that can interact with
any discrete-action reinforcement learning environment. In my case the OpenAI Gymnasium.
The agent maintains a Q-table and updates it using the Temporal-Difference learning rule.

# Main responsibilities:
- initialize and store Q-values for (state, action),
- select actions using ε-greedy exploration strategy,
- update the Q values after each step,
- provide a modular interface that can be reused with multiple experiments,
- and choose different hyperparameters (alpha, gamma, epsilon).

# Dependencies:
- numpy
- gymnasium.ActionSpace (provided externally)
- Can be extended or reused for other environments with discrete state spaces

Author: Calin Dragos George
Created: 26.October.2025
"""
import numpy as np


class QLearningAgent:
    """
    simple q-learning agent that uses a q-table to learn from experience.
    it can work with any environment that has discrete state representations
    and a discrete action space.
    """

    def __init__(self, action_space, q_table_shape, alpha, gamma, epsilon):
        # save the action space (how many actions the agent can choose)
        self.action_space = action_space

        # create a q-table filled with zeros (initial knowledge = nothing)
        self.q_table = np.zeros(q_table_shape + (action_space.n,))

        # hyperparameters for learning:
        self.alpha = alpha     
        self.gamma = gamma      
        self.epsilon = epsilon

    def select_action(self, state):
        """
        choose an action using the ε-greedy strategy.
        - sometimes we explore (pick a random action)
        - sometimes we exploit (pick the best known action)
        this helps the agent learn new things while improving what it knows.
        """
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # explore randomly
        else:
            return np.argmax(self.q_table[state])  # exploit what we learned

    def update(self, state, action, reward, next_state, done):
        """
        update the q-table values using the q-learning formula.
        this is where learning happens.
        - we take the old q value
        - we make a correction based on the reward we received
        - qnd how good the next state could be in the future
        """
        old_value = self.q_table[state + (action,)]
        next_max = np.max(self.q_table[next_state])

        if done:
            target = reward  # if the game ended, no future rewards available
        else:
            target = reward + self.gamma * next_max  # bellman target formula

        # q-learning update rule
        new_value = old_value + self.alpha * (target - old_value)
        self.q_table[state + (action,)] = new_value

    def reset(self):
        """
        optional reset functiown.
        if we ever want to restart training from scratch,
        this can clear the q-table easily.
        """
        self.q_table.fill(0.0)
        
    def save(self, filepath):
        """
        save the q table values
        """
        np.save(filepath, self.q_table)
        print(f"Q-table saved to: {filepath}")

    def load(self, filepath):
        """
        load the q table values
        """
        self.q_table = np.load(filepath)
        print(f"Loaded Q-table from: {filepath}")
