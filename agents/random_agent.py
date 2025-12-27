import numpy as np


class RandomAgent:
    """Simple random agent for training"""

    def __init__(self, name="RandomAgent"):
        self.name = name

    def action(self, action_space, observation, info):
        legal_actions = (
            [a.value for a in action_space] if action_space else list(range(13))
        )
        if legal_actions:
            action = np.random.choice(legal_actions)
        else:
            action = 0
        return action, {"agent_type": "random"}
