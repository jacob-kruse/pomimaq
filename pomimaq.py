#!/usr/bin/env python3

"""
This script describes the Partially Observable Minimax-Q Algorithm
"""

import numpy as np
from math import log10
from random import uniform
from scipy.optimize import linprog


def main():
    # Define variables to be passed to Class
    alpha = 1.0
    decay = 0.01 ** (1 / (10**6))
    gamma = 0.99
    explore = 0.2

    # Initialize POMIMAQ Class
    pomimaq = POMIMAQ(alpha, decay, gamma, explore)

    # Testing for linear programming
    pomimaq.Q[1, 9, 0] = [[0.2, 0.9], [0.8, 0.7]]
    pomimaq.learn([1, 9, 0], [11, 9, 0], 1, 0, 0)
    print(pomimaq.Q[1, 9, 0])


class POMIMAQ:
    def __init__(self, alpha=1.0, decay=(0.01 ** (1 / (10**6))), gamma=0.99, explore=0.2):
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
        # Learning decay rate
        self.decay = decay
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
        # Update Q-function for the state and actions using the observed reward
        self.Q[s[0], s[1], s[2], a, o_a] = (1 - self.alpha) * self.Q[s[0], s[1], s[2], a, o_a] + self.alpha * (
            r + self.gamma * self.V[s_[0], s_[1], s_[2]]
        )

        # Use linear programming to find the min-max action
        # Define c = [v, PI(Stand), PI(Hit)] where v is minimum from pseudo code
        c = np.zeros(3)
        c[0] = -1.0

        # Define left-side of inequality, A_ub = [num_actions, num_actions + 1]
        A_ub = np.ones((2, 3))
        # Skipping the first column, assign the transpose of the Q-function to the matrix
        A_ub[:, 1:] = -self.Q[s[0], s[1], s[2]].T

        # Define right-side of inequality, b_ub = [num_actions]
        b_ub = np.zeros(2)

        # Define left-side of equality, column vector of size (num_actions + 1)
        A_eq = np.ones((1, 3))
        # Define first value of column vector as 0
        A_eq[0, 0] = 0

        # Define right-side of equality as a 1x1 vector
        b_eq = [1]

        # Define bounds, actions need a valid probability 0 < PI(*) < 1, [PI(Stand), PI(Hit), v]
        bounds = [(None, None), (0.0, 1.0), (0.0, 1.0)]

        # Solve using linear programming function
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        # Printing for testing (JK)
        print(res)

        # Update the policy at the current state with the optimal action probabilities
        self.PI[s[0], s[1], s[2]] = res.x[:1]

        # Update the value function at the current state with the minimum value
        self.V[s_[0], s_[1], s_[2]] = res.x[0]

        # Decay the learning rate
        self.alpha = self.alpha * self.decay


if __name__ == "__main__":
    main()
