#!/usr/bin/env python3

"""
This script is for testing
"""

import numpy as np
from pomimaq import POMIMAQ


def main():
    # Define variables to be passed to Class
    alpha = 1.0
    decay = 0.01 ** (1 / (10**6))
    gamma = 0.99
    explore = 0.2
    learning = True

    # Initialize POMIMAQ Class
    pomimaq = POMIMAQ(alpha, decay, gamma, explore, learning)

    # Testing for linear programming
    # Define Q function at state with testing values
    # [Player=3, Opponent=10, UsableAce=False]
    pomimaq.Q[1, 9, 0] = [[0.2, 0.9], [0.8, 0.7]]

    # Call the learn() function for the state above
    pomimaq.learn([1, 9, 0], [11, 9, 0], 1, 0, 0)

    # Save the current version of the class to a ".npy" file
    np.save("Minmax", pomimaq)

    # Load the class from the ".npy" file
    mimaq = np.load("Minmax.npy", allow_pickle=True).item()

    # Print result of learn() function after loading
    print(f"Q Function:\n {mimaq.Q[1, 9, 0]}")
    print(f"Optimal Policy Update: {mimaq.PI[1, 9, 0]}")
    print(f"Minimax Value: {mimaq.V[1, 9, 0]}")


if __name__ == "__main__":
    main()
