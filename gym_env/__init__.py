"""Registration to the gymnasium"""

import gymnasium as gym

gym.register(
    id="Holdem_NoLimit-v0",
    entry_point="gym_env.env:HoldemTable",
    kwargs={},
    max_episode_steps=1000,
    reward_threshold=1000.0,
)

# Also register with the neuron_poker name for backwards compatibility
gym.register(
    id="neuron_poker-v0",
    entry_point="gym_env.env:HoldemTable",
    kwargs={},
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
