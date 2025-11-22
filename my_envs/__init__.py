from gymnasium.envs.registration import register

register(
    id="blackjack-v1",
    entry_point="my_envs.self_blackjack:BlackjackEnv",
)
