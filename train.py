#!/usr/bin/env python3

"""
This script provides the training process for different algorithms
"""

import os
import my_envs
import warnings
import numpy as np
import gymnasium as gym
from pomimaq import POMIMAQ, QLearningAgent


def main():
    # Define number of learning steps to complete training
    total_steps = 1000000

    # Define agents: [file name, agent type] ;  None = Default Policy
    agent1 = ["MMQ_vs_MMQ2_1", "MMQ"]
    agent2 = ["MMQ_vs_MMQ2_2", "MMQ2"]

    # If an "agent1" file is defined and does not exist, initialize a new agent Class
    if agent1 and not os.path.exists(f"agents/{agent1[0]}.npy"):
        if agent1[1] == "MMQ":
            player1 = POMIMAQ()
        elif agent1[1] == "Q":
            player1 = QLearningAgent()
        else:
            print("Wrong type for Agent 1")
            return

    # If an "agent1" file is defined and exists, load the previously saved agent Class
    elif agent1 and os.path.exists(f"agents/{agent1[0]}.npy"):
        player1 = np.load(f"agents/{agent1[0]}.npy", allow_pickle=True).item()

    # If "agent1" is not defined, return an error
    else:
        print("Define Agent for Player 1")
        return

    # If an "agent2" file is defined and does not exist, initialize a new agent Class
    if agent2 and not os.path.exists(f"agents/{agent2[0]}.npy"):
        if agent2[1] == "MMQ":
            player2 = POMIMAQ()
        elif agent2[1] == "Q":
            player2 = QLearningAgent()
        else:
            print("Wrong type for Agent 2")
            return

    # If an "agent2" file is defined and exists, load the previously saved agent Class
    elif agent2 and os.path.exists(f"agents/{agent2[0]}.npy"):
        player2 = np.load(f"agents/{agent2[0]}.npy", allow_pickle=True).item()

    # If no "agent2" file is defined, define player 2 as the default policy
    else:
        player2 = None

    # Suppress meaningless warnings
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

    # Initialize and reset the gym environment for blackjack
    env = gym.make("blackjack-v1")
    obs, info = env.reset()

    # Define "done" as False initially
    done = False
    # Define variables for win tracking
    wins, losses, draws = 0, 0, 0
    # Define opponent actions to be Hit initially (Initially Hit since game will continue)
    opp_action_1, opp_action_2 = 1, 1
    # Define previous step variable for logic in periodic print statements
    prev_step = -1

    # Print the current training process
    print(f"\033[1m{agent1[0]}\033[0m")

    # Try the following
    try:
        # Loop while the steps of player1 is less than the total learning steps defined
        while player1.steps <= total_steps:
            # Get the the player who is on the current turn
            turn = info["current_turn"]

            # If it's the players turn
            if turn == "player":
                # Call the choose_action() function with the current observation
                action = player1.choose_action(obs[0])

                # Assign the action to opp_action for player 2
                opp_action_2 = action

            # If it's the dealers turn
            elif turn == "dealer":
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

                # Assign the action to opp_action for player 1
                opp_action_1 = action

            # Assign the current observation to prev_obs for storage
            prev_obs = obs

            # Execute the chosen action for the current player in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            # If the game has terminated, set "done" to True to do modified update
            if terminated or truncated:
                done = True

            # If the current player is "dealer", backwards because we called step()
            if turn == "player":
                # Call the learn() function for player1 to process transition
                player1.learn(prev_obs[0], obs[0], action, opp_action_1, reward, done)

            # If the current player is "player" and agent2 is defined
            elif turn == "dealer" and player2:
                # Call the learn() function for player2 to process transition
                player2.learn(prev_obs[1], obs[1], action, opp_action_2, -reward, done)

            # If the current game has ended
            if terminated or truncated:
                # Reset the environment and other variables
                obs, info = env.reset()
                done = False
                opp_action_1, opp_action_2 = 1, 1

                # Increment the wins/losses/draws based on outcome
                if reward == 1.0:
                    wins += 1
                elif reward == -1.0:
                    losses += 1
                else:
                    draws += 1

            # If learning steps is divisible by 10,000
            if player1.steps % 10000 == 0 and prev_step != player1.steps:
                # Periodic print statements
                print(f"Step {player1.steps}")
                print(f"Wins: {wins}  Losses: {losses}  Differential: {wins-losses}")

                # Save the player1 class periodically
                np.save(f"agents/{agent1[0]}", player1)

                # If there is a player2, save the class
                if player2:
                    np.save(f"agents/{agent2[0]}", player2)

                # Store the previous step to avoid multiple printing and saving
                prev_step = player1.steps

        # After the loop has finished, set the "learning" flag for player1 to False and save it
        player1.learning = False
        np.save(f"agents/{agent1[0]}", player1)

        # If there is an agent for player2, set the "learning" flag for player2 to False and save
        if player2:
            player2.learning = False
            np.save(f"agents/{agent2[0]}", player2)

    # Handle Keyboard Interrupt
    except KeyboardInterrupt:
        # Print the training process and number of learning steps for the player1 agent
        print(f"\n\033[1m{agent1[0]}\033[0m")
        print(f"Steps: {player1.steps}")

        # Save the current state of player1
        np.save(f"agents/{agent1[0]}", player1)

        # If an agent is defined for player2, save the current state
        if player2:
            np.save(f"agents/{agent2[0]}", player1)


if __name__ == "__main__":
    main()
