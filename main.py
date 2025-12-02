#!/usr/bin/env python3

"""
This script starts the gym environment for Blackjack and simulates actions
"""

import my_envs
import warnings
from time import sleep
import gymnasium as gym

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

env = gym.make("blackjack-v1", render_mode="human")

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
    env.render()
    print(
        "\033[4mObservations\033[0m\n"
        f"Player Sum: {obs[0][0]} ; Dealer Showing Card: {obs[0][1]}\n"
        f"Dealer Sum: {obs[1][0]} ; Player Showing Card: {obs[1][1]}"
    )

    sleep(6.0)
    if terminated or truncated:
        input('\nPress "Enter" to close window')
        break
