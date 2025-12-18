"""Registration to the gymnasium"""

import gymnasium as gym

gym.register(
    id="neuron_poker-v0",
    entry_point="gym_env.env:HoldemTable",
    kwargs={},
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
