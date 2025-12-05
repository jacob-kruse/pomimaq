#!/usr/bin/env python3

"""
This script provides a way to get gameplay metrics after algorithm(s) are trained
"""

import os
import my_envs
import warnings
import numpy as np
import gymnasium as gym


def main():
    # Define number of games to simulate
    total_games = 100000

    # Define trained agents simulation and win/loss comparison
    agent1 = "MMQ_vs_Default"
    # agent1 = "Q_vs_Default"
    # agent1 = "MMQ_vs_MMQ_1"
    # agent1 = "Q_vs_Q_1"
    agent2 = None
    # agent2 = "MMQ_vs_MMQ_2"
    # agent2 = "Q_vs_Q_2"

    # If "agent1" file is defined and exists, load the previously saved agent Class
    if agent1 and os.path.exists(f"agents/{agent1}.npy"):
        player1 = np.load(f"agents/{agent1}.npy", allow_pickle=True).item()

    # If "agent1" is not defined or if it doesn't exist, return an error
    else:
        print("Define correct Agent 1 file")
        return

    # If an "agent2" file is defined and does not exist, return an error
    if agent2 and not os.path.exists(f"agents/{agent2}.npy"):
        print("Agent 2 file does not exist")
        return

    # If "agent2" file is defined and exists, load the previously saved agent Class
    elif agent2 and os.path.exists(f"agents/{agent2}.npy"):
        player2 = np.load(f"agents/{agent2}.npy", allow_pickle=True).item()

    # If no "agent2" file is defined, define player 2 as the default policy
    else:
        player2 = None

    # Define tracking variables (Only need to keep track of player 1, just negate for other)
    games, wins, losses, draws = 0, 0, 0, 0
    prev_game = -1

    # Suppress meaningless warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

    # Initialize and reset the gym environment for blackjack
    env = gym.make("blackjack-v1")
    obs, info = env.reset()

    # Print the current assessment process
    print(f"\033[1m{agent1}\033[0m")

    # Try the following
    try:
        # Loop while the steps of minmax-q learning algorithm is less than a million
        while games < total_games:
            # If it's the players turn
            if info["current_turn"] == "player":
                # If there is an agent for player1
                if player1:
                    # Call the choose_action() function with the current observation
                    action = player1.choose_action(obs[0])

                # Play default policy if no player specified
                else:
                    # If the player's sum is less than 17, hit
                    if obs[0][0] < 17:
                        action = 1

                    # If the player's sum is 17 or greater, stand
                    else:
                        action = 0

            # If it's the dealers turn
            elif info["current_turn"] == "dealer":
                # If there is an agent for player2
                if player2:
                    # Call the choose_action() function with the current observation
                    action = player2.choose_action(obs[1])

                # Play default policy if no player specified
                else:
                    # If the player's sum is less than 17, hit
                    if obs[1][0] < 17:
                        action = 1

                    # If the player's sum is 17 or greater, stand
                    else:
                        action = 0

            # Execute the chosen action for the current player in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # If the current game has ended
            if terminated or truncated:
                # Reset the environment
                obs, info = env.reset()

                # Increment games and wins/losses/draws based on outcome
                games += 1
                if reward == 1.0:
                    wins += 1
                elif reward == -1.0:
                    losses += 1
                else:
                    draws += 1

            # If games is divisible by 10,000
            if games % 10000 == 0 and prev_game != games:
                # Print games played so far periodically
                print(f"Games: {games}")

                # Store the previous game to avoid multiple printing
                prev_game = games

        # Print the number of games simulated and Wins and Losses
        print(f"\n\033[1m{agent1}\033[0m")
        print(f"Games: {games}")
        print(f"Wins: {wins}  Losses: {losses}  Draws: {draws}  Differential: {wins-losses}")

    # Handle Keyboard Interrupt
    except KeyboardInterrupt:
        # Print the number of games simulated and Wins and Losses
        print(f"\n\033[1m{agent1}\033[0m")
        print(f"Games: {games}")
        print(f"Wins: {wins}  Losses: {losses}  Draws: {draws}  Differential: {wins-losses}")


if __name__ == "__main__":
    main()
