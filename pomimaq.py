#!/usr/bin/env python3

"""
This script describes the Partially Observable Minimax-Q Algorithm
"""

import numpy as np
from random import uniform
from scipy.optimize import linprog


def main():
    # Define variables to be passed to Class
    alpha = 1.0
    decay = 0.01 ** (1 / (10**6))  # Chosen so that alpha is 0.01 after a million cycles
    gamma = 0.99
    explore = 0.2
    learning = True

    # Initialize POMIMAQ Class
    pomimaq = POMIMAQ(alpha, decay, gamma, explore, learning)


class POMIMAQ:
    def __init__(self, alpha=1.0, decay=(0.01 ** (1 / (10**6))), gamma=0.99, explore=0.2, learning=True):
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
        # Flag that determines if we are in the learning phase
        self.learning = learning
        # Tracks number of steps for learning
        self.steps = 0

                # ---------------- Belief Model ----------------
        # win_counts[s, a]: number of times action 'a' led to a winning terminal outcome in state 's'
        # play_counts[s, a]: total number of times action 'a' was taken in state 's'
        # belief[s, a] = win_counts / play_counts = estimated probability of winning if action 'a' is chosen in state 's'
        self.win_counts = np.zeros((num_sums, num_shows, usable_ace, num_actions), dtype=np.float32)
        # Start at 1 to avoid division by zero and allow belief to converge from a neutral prior
        self.play_counts = np.ones((num_sums, num_shows, usable_ace, num_actions), dtype=np.float32)
        self.belief = self.win_counts / self.play_counts

        # Weight for belief-based shaping in the Q target.
        # Set to 0.0 to disable shaping, increase gradually (e.g., 0.1 ~ 0.5).
        self.belief_weight = 0.2

    def update_belief(self, state, a, reward):
        """
        Update the empirical belief P(win | state, action).
        This function assumes reward > 0 indicates a win.
        """
        state = self.convert_state(state)
        i, j, k = state
        a = int(a)

        win = 1.0 if reward > 0 else 0.0
        self.win_counts[i, j, k, a] += win
        self.play_counts[i, j, k, a] += 1.0
        self.belief[i, j, k, a] = self.win_counts[i, j, k, a] / self.play_counts[i, j, k, a]

    # ---------------- Belief Query Function ----------------
    def get_belief(self, state, a):
        """
        Return the current estimated probability of winning
        when taking action 'a' in the given blackjack state.
        """
        state = self.convert_state(state)
        i, j, k = state
        return float(self.belief[i, j, k, int(a)])


    def choose_action(self, state):
        # Convert the passed state
        state = self.convert_state(state)

        # Get a random decimal number between 0 and 1 for exploration and action choices
        explore_sample = uniform(0, 1)
        action_sample = uniform(0, 1)

        # If we're in the learning phase and the sample is between 0 and the probability to explore
        if self.learning and explore_sample < self.explore:
            # Sample an action randomly (Stand = 0, Hit = 1)
            action = 1 if action_sample > 0.5 else 0

        # In all other cases
        else:
            # Get the action proabilities from the policy
            action_probs = self.PI[state[0], state[1], state[2]]

            # Sample an action with the action probabilities (Stand = 0, Hit = 1)
            action = 1 if action_sample > action_probs[0] else 0

        return action

    def learn(self, s, s_, a, o_a, r, done =False):
        # Convert the passed states to indexable equivalents
        s = self.convert_state(s)
        s_ = self.convert_state(s_)

        # Update Q-function for the state and actions using the observed reward
        self.Q[s[0], s[1], s[2], a, o_a] = (1 - self.alpha) * self.Q[s[0], s[1], s[2], a, o_a] + self.alpha * (
            r + self.gamma * self.V[s_[0], s_[1], s_[2]] + self.belief_weight * (self.belief[s[0], s[1], s[2], a] - 0.5)
        )

        # Use linear programming to find the min-max action
        # Define c = [v, PI(Stand), PI(Hit)] where v is minimum from pseudo code
        c = np.zeros(3)
        c[0] = -1.0

        # Define left-side of inequality, A_ub = [num_actions, num_opp_actions + 1]
        A_ub = np.ones((2, 3))
        # Skipping the first column, assign the transpose of the Q-function to the matrix
        A_ub[:, 1:] = -self.Q[s[0], s[1], s[2]].T

        # Define right-side of inequality, b_ub = [num_actions]
        b_ub = np.zeros(2)

        # Define left-side of equality, column vector of size (num_opp_actions + 1)
        A_eq = np.ones((1, 3))
        # Define first value of column vector as 0
        A_eq[0, 0] = 0

        # Define right-side of equality as a 1x1 vector
        b_eq = [1]

        # Define bounds, actions need a valid probability 0 < PI(*) < 1, [PI(Stand), PI(Hit), v]
        bounds = [(None, None), (0.0, 1.0), (0.0, 1.0)]

        # Solve using linear programming function
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

        # If the linear programming succeeded
        if res.success:
            # Update the policy at the current state with the optimal action probabilities
            self.PI[s[0], s[1], s[2]] = res.x[1:]

            # Update the value function at the current state with the minimum value
            self.V[s[0], s[1], s[2]] = res.x[0]

        # If the linear programming failed
        elif not res.success:
            # Print the error message
            print("Linear Programming Failed: %s" % res.message)

        if done:
            self.update_belief(s, a, r)
        # Decay the learning rate
        self.alpha = self.alpha * self.decay

        # Increment step counter
        self.steps += 1

    def convert_state(self, state):
        # Convert True or False for "usable_ace" to 1 or 0
        usable_ace = 0 if not state[2] else 1

        # Subtract 2 from the player sum and 1 from dealers card to get indexes
        converted_state = [(state[0] - 2), (state[1] - 1), usable_ace]

        return converted_state

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        n_player_sum: int = 32,
        n_opponent_card: int = 11,
        n_usable_ace: int = 2,
        n_actions: int = 2,
    ):
        """
        - self_sum:            0..31
        - opponent_card:       0..10  (store the raw value from obs[1])
        - usable_ace:          0 or 1
        - action:              0 = stick, 1 = hit
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.n_player_sum = n_player_sum
        self.n_opponent_card = n_opponent_card
        self.n_usable_ace = n_usable_ace
        self.n_actions = n_actions

        # Initialize Q-table with zeros
        self.Q = np.zeros(
            (n_player_sum, n_opponent_card, n_usable_ace, n_actions),
            dtype=np.float32,
        )

    def _obs_to_index(self, obs):
        """
        Map observation (self_sum, opp_card, usable_ace) to Q-table indices.

        obs[0]: self_sum          (0..31)
        obs[1]: opp_card          (0..10)
        obs[2]: usable_ace        (0/1 or bool)
        """
        self_sum, opp_card, usable = obs

        self_sum = int(np.clip(self_sum, 0, self.n_player_sum - 1))
        opp_card = int(np.clip(opp_card, 0, self.n_opponent_card - 1))
        usable = int(bool(usable))

        return self_sum, opp_card, usable

    def select_action(self, obs):
        """
        Epsilon-greedy action selection.
        With probability epsilon: choose a random action (exploration).
        Otherwise: choose argmax_a Q(s, a) (exploitation).
        """
        i_sum, i_card, i_ace = self._obs_to_index(obs)

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[i_sum, i_card, i_ace, :]
            return int(np.argmax(q_values))

    def update(self, obs, action, reward, next_obs, done: bool):
        """
        Standard Q-learning update:
        target = r + gamma * max_a' Q(s', a')    if not done
        target = r                               if done
        """
        i_sum, i_card, i_ace = self._obs_to_index(obs)
        a = int(action)

        q_sa = self.Q[i_sum, i_card, i_ace, a]

        if done:
            target = reward
        else:
            n_sum, n_card, n_ace = self._obs_to_index(next_obs)
            next_q = self.Q[n_sum, n_card, n_ace, :]
            target = reward + self.gamma * np.max(next_q)

        self.Q[i_sum, i_card, i_ace, a] = q_sa + self.lr * (target - q_sa)

    def decay_epsilon(self):
        """
        Decay epsilon after each step or episode.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_epsilon(self, value: float):
        """
        Manually set epsilon if needed (e.g., for evaluation).
        """
        self.epsilon = float(value)



if __name__ == "__main__":
    main()
