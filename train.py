#!/usr/bin/env python3

"""
This script provides the training process for different algorithms
"""

import os
import my_envs
import warnings
import numpy as np
import gymnasium as gym
from pomimaq import POMIMAQ


def main():
    # If the "MinMax_Q.npy" file does not exist
    if not os.path.exists("MinMax_Q.npy"):
        # Define variables to be passed to POMIMAQ Class
        alpha = 1.0
        decay = 0.01 ** (1 / (10**6))
        gamma = 0.99
        explore = 0.2
        learning = True

        # Initialize a new POMIMAQ Class
        mimaq = POMIMAQ(alpha, decay, gamma, explore, learning)

    else:
        # Load the class from the ".npy" file
        mimaq = np.load("MinMax_Q.npy", allow_pickle=True).item()

    # Suppress meaningless warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

    # Initialize and reset the gym environment for blackjack
    env = gym.make("blackjack-v1")
    obs, info = env.reset()

    # Define variables for win tracking
    total_wins, total_losses, total_draws, wins, losses, draws = 0, 0, 0, 0, 0, 0
    # Define previous step variable for logic in periodic print statements
    prev_step = -1
    # Define opponent action to be -1 (No-op) initially in case player goes first
    opp_action = 1

    # Try the following
    try:
        # Loop while the steps of minmax-q learning algorithm is less than a million
        while mimaq.steps <= 1000000:
            # If it's the players turn
            if info["current_turn"] == "player":
                # Call the minmax-q choose_action() function with the current observation
                action = mimaq.choose_action(obs[0])

            # If it's the dealers turn
            elif info["current_turn"] == "dealer":
                # # If the dealer's sum is less than 17, dealer will hit
                # if obs[1][0]:
                #     action = 1

                # # If the dealer's sum is 17 or greater, dealer will stand
                # else:
                #     action = 0

                # Randomly sample an action
                action = env.action_space.sample()

                # Assign the action to opp_action
                opp_action = action

            # Assign the current observation to prev_obs for storage
            prev_obs = obs[0]

            # Execute the chosen action for the current player in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # If the current player is dealer, backwards because we called step()
            if info["current_turn"] == "dealer":
                # Call the learn() function for the minmax-q algorithm to process transition
                mimaq.learn(prev_obs, obs[0], action, opp_action, reward)

            # If the current game has ended
            if terminated or truncated:
                # Reset the environment
                obs, info = env.reset()
                # sleep(2.0)

                # Increment the wins/losses/draws based on outcome
                if reward == 1.0:
                    wins += 1
                    total_wins += 1
                elif reward == -1.0:
                    losses += 1
                    total_losses += 1
                else:
                    draws += 1
                    total_draws += 1

            # If learning steps is divisible by 1,000
            if mimaq.steps % 1000 == 0 and prev_step != mimaq.steps:
                # Periodic print statements
                print(f"Step {mimaq.steps}")
                print(f"Wins: {wins}  Losses: {losses}  Differential: {wins-losses}")
                print(f"Total Wins: {total_wins}  Total Losses: {total_losses}  Total Differential: {total_wins-total_losses}\n")

                # Reset the periodic win tracking variables
                wins, losses, draws = 0, 0, 0
                # Store the previous step to avoid multiple printing
                prev_step = mimaq.steps

            # If the steps is divisible by 10,000
            if mimaq.steps % 10000 == 0:
                # Save the minmax-q algorithm periodically
                np.save("MinMax_Q", mimaq)

    # Handle Keyboard Interrupt
    except KeyboardInterrupt:
        # Print the number of learning steps for the minmax-q algorithm
        print(f"\nSteps: {mimaq.steps}")

        # Save the current state of the minmax-q algorithm
        np.save("MinMax_Q", mimaq)


if __name__ == "__main__":
    main()
