#!/usr/bin/env python3

"""
This script describes the Partially Observable Minimax-Q Algorithm
"""

import numpy as np
from random import uniform


def main():
    # Define variables to be passed to Class
    alpha = 1.0
    gamma = 0.99
    explore = 0.2
    pomimaq = POMIMAQ(alpha, gamma, explore)


class POMIMAQ:
    def __init__(self, alpha=1.0, gamma=0.99, explore=0.2):
        # Number of possible sums [2-31]  ## (Subtract sum by 2 to get index)
        num_sums = 30
        # Number of possible shown cards [1-10]  ## (Subtract show by 1 to get index)
        num_shows = 10
        # Flag on whether the player has an ace
        usable_ace = 2
        # Number of possible actions [Hit, Stand]
        num_actions = 2

        # Define value function with state space dimensions
        self.V = np.ones((num_sums, num_shows, usable_ace))
        # Set all values where player has busted to zero
        self.V[20:29, :, :] = 0

        # Define Q-function with state space and both players' action space dimensions
        self.Q = np.ones((num_sums, num_shows, usable_ace, num_actions, num_actions))
        # Set all Q-values where player has busted to zero
        self.Q[20:29, :, :, :, :] = 0

        # Define policy with state space and single player's action space dimensions
        self.PI = np.ones((num_sums, num_shows, usable_ace, num_actions))
        # Divide by number of actions to get uniform probability for each action at each state
        self.PI = self.PI / num_actions

        # Learning rate
        self.alpha = alpha
        # Discount factor
        self.gamma = gamma
        # Probability to explore an action that is not the policy's
        self.explore = explore

    def choose_action(self, state):
        # Get a random decimal number between 0 and 1 for exploration and action choices
        explore_sample = uniform(0, 1)
        action_sample = uniform(0, 1)

        # If the sampled decimal is between 0 and the probability to explore
        if explore_sample < self.explore:
            # Sample an action randomly
            action = "Hit" if action_sample > 0.5 else "Stand"

        # If the sampled decimal is not between 0 and the probability to explore
        elif explore_sample >= self.explore:
            # Get the action proabilities from the policy
            action_probs = self.PI[state[0]][state[1]][state[2]]

            # Sample an action with the action probabilities
            action = "Hit" if action_sample > action_probs[0] else "Stand"

        return action

    def learn(self, s, s_, a, o_a, r):
        # Update Q-function with reward
        self.Q[s[0]][s[1]][s[2]][a][o_a] = (1 - self.alpha) * self.Q[s[0]][s[1]][s[2]][a][o_a] + self.alpha * (
            r + self.gamma * self.V[s_[0]][s_[1]][s_[2]]
        )

        # Still needs to be finished


if __name__ == "__main__":
    main()
