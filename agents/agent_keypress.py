"""manual keypress agent"""

from gym_env.enums import Action


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name="Keypress"):
        """Initialization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ""
        self.temp_stack = []
        self.name = name
        self.autoplay = True

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = (observation, info)  # not using the observation for manual decision

        # Create a mapping from action value to Action enum
        action_mapping = {action.value: action for action in Action}

        # Filter to only include legal actions
        legal_actions = [
            action for action in action_space if action in action_mapping.values()
        ]

        if not legal_actions:
            # No legal moves, return FOLD as default
            return Action.FOLD

        # Display legal actions
        print(f"\n=== {self.name}'s Turn ===")
        print("Legal actions:")
        for i, action in enumerate(legal_actions):
            print(f"  {action.value}: {action.name}")

        # Get user input
        while True:
            try:
                choice = int(input("Enter action number: "))
                selected_action = action_mapping.get(choice)

                if selected_action in legal_actions:
                    return selected_action
                else:
                    print(
                        f"Action {choice} is not legal. Choose from: {[a.value for a in legal_actions]}"
                    )
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                return Action.FOLD  # Default to fold on Ctrl+C
