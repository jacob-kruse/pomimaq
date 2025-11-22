#!/usr/bin/env python3

"""
This script starts the gym environment for Blackjack and simulates actions
"""

import my_envs
import gymnasium as gym


env = gym.make("blackjack-v1")

obs, info = env.reset()

player_stick = False
dealer_stick = False

print(
    f"Starting Agent: {info['current_turn']}\n"
    "\033[4mInitial Observations\033[0m\n"
    f"Player Sum: {obs[0][0]}  Dealer Showing Card: {obs[0][1]}\n"
    f"Dealer Sum: {obs[1][0]}  Player Showing Card: {obs[1][1]}"
)

for _ in range(15):
    # player's turn
    if info["current_turn"] == "player" and not player_stick:
        action = env.action_space.sample()
        str_action = "Hit" if action else "Stand"
        print(f"\nPlayer's Action: {str_action}")
        if action == 0:
            player_stick = True
    # dealer's turn
    elif info["current_turn"] == "dealer" and not dealer_stick:
        action = env.action_space.sample()
        str_action = "Hit" if action else "Stand"
        print(f"\nDealer's Action: {str_action}")
        if action == 0:
            dealer_stick = True
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        "\033[4mObservations\033[0m\n"
        f"Player Sum: {obs[0][0]} ; Dealer Showing Card: {obs[0][1]}\n"
        f"Dealer Sum: {obs[1][0]} ; Player Showing Card: {obs[1][1]}"
    )

    if terminated or truncated:
        break
