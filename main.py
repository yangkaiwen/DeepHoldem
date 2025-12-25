"""
neuron poker

Usage:
  main.py play random [options]
  main.py play keypress [options]
  main.py play consider_equity [options]
  main.py play equity_improvement --improvement_rounds=<> [options]
  main.py play ac_train [options]
  main.py play dqn_train [options]
  main.py play dqn_play [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].
  --load_model=<>           Path to load model from

"""

import logging

import numpy as np
import pandas as pd
from docopt import docopt

from gym_env.env import HoldemTable
from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


# pylint: disable=import-outside-toplevel


def command_line_parser():
    """Entry function"""
    args = docopt(__doc__)
    if args["--log"]:
        logfile = args["--log"]
    else:
        print("Using default log file")
        logfile = "default"
    model_name = args["--name"] if args["--name"] else "dqn1"
    screenloglevel = (
        logging.INFO
        if not args["--screenloglevel"]
        else getattr(logging, args["--screenloglevel"].upper())
    )
    init_logger(screenlevel=screenloglevel, filename=logfile)
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    if args["play"]:
        num_episodes = 1 if not args["--episodes"] else int(args["--episodes"])
        runner = GameRunner(
            num_episodes=num_episodes,
            stack=int(args["--stack"]),
        )

        if args["random"]:
            runner.random_agents()

        elif args["keypress"]:
            runner.key_press_agents()

        elif args["ac_train"]:
            load_path = args["--load_model"]
            runner.ac_train(load_path)

    else:
        raise RuntimeError("Argument not yet implemented")


class GameRunner:
    """Orchestration of playing games"""

    def __init__(self, num_episodes, stack=500):
        """Initialize"""
        self.winner_in_episodes = []
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

    def ac_train(self, load_path=None):
        """
        Run training with Actor-Critic agents.
        Multiple agents share the same network and learn from all experiences.
        """
        from agents.ac_agent import PokerACAgent, RandomAgent

        # Create ONE central agent instance
        central_agent = PokerACAgent(name="CentralAC", device="auto")

        if load_path:
            central_agent.load(load_path)

        central_agent.train_mode()

        for episode in range(self.num_episodes):
            # Randomize number of players (2-10)
            num_players = np.random.randint(2, 11)

            # Create environment
            self.env = HoldemTable(initial_stacks=self.stack)

            # Add players - mix of AC agents (controlled by central_agent) and Random agents
            # E.g., 80% AC agents, 20% Random
            ac_player_count = 0
            for i in range(num_players):
                if np.random.random() < 0.8:  # 80% chance of AC agent
                    # We pass the SAME central_agent instance
                    # The agent is stateless regarding the hand, so this is fine
                    # It uses info['player_data']['position'] to separate buffers
                    self.env.add_player(central_agent)
                    ac_player_count += 1
                else:
                    self.env.add_player(RandomAgent(name=f"Random_{i}"))

            if ac_player_count == 0:
                continue  # Skip if no AC agents

            self.log.info(
                f"Episode {episode+1}/{self.num_episodes}: {num_players} players ({ac_player_count} AC)"
            )

            # Generate random stacks (200BB to 2000BB)
            # Default BB is 2, so 400 to 4000
            bb = self.env.big_blind
            random_stacks = np.random.uniform(200 * bb, 2000 * bb, num_players)

            # Random dealer position
            dealer_pos = np.random.randint(0, num_players)

            # Run episode
            self.env.reset(options={"stacks": random_stacks, "dealer_pos": dealer_pos})
            self.env.run()

            # Collect experiences from ALL players
            all_experiences = self.env.get_player_experiences()

            # Update central agent using aggregated experiences
            # The update method now handles the dict of {seat_id: experiences}
            central_agent.update(all_experiences)

            # Save periodically
            if (episode + 1) % 100 == 0:
                central_agent.save(f"models/ac_agent_{episode+1}.pt")

    def key_press_agents(self):
        """Create an environment with key press agents"""
        from agents.agent_keypress import KeypressAgent

        num_of_plrs = 3
        self.env = HoldemTable(initial_stacks=self.stack)
        for _ in range(num_of_plrs):
            player = KeypressAgent()
            self.env.add_player(player)

        # Episode loop
        for episode in range(self.num_episodes):
            self.log.info(f"Starting episode {episode + 1}/{self.num_episodes}")

            # Store initial stacks before reset (which deducts blinds)
            initial_stacks = [self.stack] * num_of_plrs

            self.env.reset()
            self.env.run()

            # Get and output player experiences
            player_experiences = self.env.get_player_experiences()

            print("\n" + "=" * 80)
            print(f"PLAYER EXPERIENCES - EPISODE {episode + 1}")
            print("=" * 80)

            for player_id, experiences in player_experiences.items():
                player = self.env.players[player_id]
                print(
                    f"\nPlayer {player_id} ({player.name}) - {len(experiences)} experiences:"
                )
                print("-" * 80)

                for exp_idx, exp in enumerate(experiences):
                    print(f"\n  Experience {exp_idx + 1}:")
                    print(
                        f"    State shape: {exp['state'].shape if exp['state'] is not None else 'None'}"
                    )
                    print(f"    Action: {exp['action']}")
                    print(f"    Reward: {exp['reward']:.4f}")
                    print(
                        f"    Next state shape: {exp['next_state'].shape if exp['next_state'] is not None else 'None'}"
                    )
                    print(f"    Done: {exp['done']}")

            # Output terminal rewards
            terminal_rewards = self.env.get_terminal_rewards()
            print("\n" + "=" * 80)
            print("TERMINAL REWARDS:")
            print("-" * 80)
            for player_id, reward in enumerate(terminal_rewards):
                player = self.env.players[player_id]
                print(f"Player {player_id} ({player.name}): {reward:.4f}")
            print("=" * 80 + "\n")


if __name__ == "__main__":
    command_line_parser()
