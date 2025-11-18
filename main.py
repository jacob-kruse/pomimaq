import gymnasium as gym
import my_envs

env = gym.make("blackjack-v1")

obs, info = env.reset()

for _ in range(15):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward)

    if terminated or truncated:
        print("Episode ended.")
        break