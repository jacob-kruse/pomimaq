import gymnasium as gym
import my_envs

env = gym.make("blackjack-v1")

obs, info = env.reset()

player_stick = False
dealer_stick = False
print(obs)
for _ in range(15):
    # player's turn
    if info["current_turn"] == "player" and not player_stick:
        print('player')
        action = 1
        if action == 0:
            player_stick = True
    # dealer's turn
    elif info["current_turn"] == "dealer" and not dealer_stick:
        print('dealer')
        action = env.action_space.sample()
        if action == 0:
            dealer_stick = True
    obs, reward, terminated, truncated, info = env.step(action)
    print(action, obs, reward)

    if terminated or truncated:
        print("Episode ended.")
        break