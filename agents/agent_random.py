"""Random player"""

import random

from gym_env.enums import Action


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name="Random"):
        """Initialization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ""
        self.temp_stack = []
        self.name = name
        self.autoplay = True

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        # Choose randomly from the legal actions provided by the environment
        if action_space:  # Check if there are any legal moves
            # Convert action_space to list if it's not already
            legal_actions = list(action_space)
            return random.choice(legal_actions)
        else:
            # No legal moves, return FOLD as default
            return Action.FOLD
