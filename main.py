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
  main.py play game_evaluator [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log=<>                  log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player
  --load_model=<>           Path to load model from
  --model_path=<>           Path to model for game_evaluator (or 'keypress')
  --num_players=<>          Number of players (2-9)
  --opponent_probs=<>       Opponent probabilities (e.g. "0.5,0.2,0.3")
  --log_results=<>          Path to save evaluation results (CSV)
  --use_llm                 Use LLM agents as opponents

"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix for Windows DLL error: [WinError 1114]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import torch
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
        # print("Using default log file")
        logfile = "default"
    model_name = args["--name"] if args["--name"] else "dqn1"
    screenloglevel = (
        logging.WARNING
        if not args["--screenloglevel"]
        else getattr(logging, args["--screenloglevel"].upper())
    )
    init_logger(screenlevel=screenloglevel, filename=logfile)
    # print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    if args["play"]:
        num_episodes = 1 if not args["--episodes"] else int(args["--episodes"])
        stack_val = int(args["--stack"]) if args["--stack"] else 500
        runner = GameRunner(
            num_episodes=num_episodes,
            stack=stack_val,
        )

        if args["keypress"]:
            runner.key_press_agents()

        elif args["ac_train"]:
            load_path = args["--load_model"]
            use_llm = args["--use_llm"]
            runner.ac_train(load_path, use_llm)

        elif args["game_evaluator"]:
            from game_evaluator import GameEvaluator

            model_path = args["--model_path"]
            num_players = args["--num_players"]
            opponent_probs_str = args["--opponent_probs"]
            log_results = args["--log_results"]
            opponent_probs = (
                [float(x) for x in opponent_probs_str.split(",")]
                if opponent_probs_str
                else None
            )

            evaluator = GameEvaluator(
                model_path=model_path,
                num_episodes=num_episodes,
                num_players=num_players,
                initial_stack=args["--stack"],
                opponent_probabilities=opponent_probs,
                log_results_path=log_results,
            )
            evaluator.run()

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

    def ac_train(self, load_path=None, use_llm=False):
        """
        Run training with Actor-Critic agents.
        Multiple agents share the same network and learn from all experiences.
        """
        from agents.ac_agent import PokerACAgent
        from agents.random_agent import RandomAgent

        if use_llm:
            from agents.llm_agent import LLMAgent
            import random

            models = [
                "xiaomi/mimo-v2-flash:free",
                "mistralai/devstral-2512:free",
            ]

            # Create a pool of LLM agents
            # llm_agent_pool = [
            #     LLMAgent(name=f"LLM_{i}", model=models[(i - 1) % len(models)])
            #     for i in range(1, 10)
            # ]

        # Create ONE central agent instance
        # Initializing agent
        central_agent = PokerACAgent(name="CentralAC", device="auto")

        if load_path:
            central_agent.load(load_path)

        central_agent.train_mode()

        # Initialize performance log
        # with open("training_performance.csv", "w") as f:
        #     f.write("episode,roi\n")

        for episode in range(self.num_episodes):
            # Randomize number of players (2-10)
            num_players = np.random.randint(2, 11)

            # Create environment
            self.env = HoldemTable(initial_stacks=self.stack)

            # Add players
            self.env.add_player(central_agent)
            ac_player_count = 1

            if use_llm:
                # Create agents dynamically for each episode
                for i in range(num_players - 1):
                    model = random.choice(models)
                    agent = LLMAgent(name=f"LLM_{i}", model=model)
                    self.env.add_player(agent)
            else:
                ac_player_count = num_players
                for i in range(1, num_players):
                    self.env.add_player(central_agent)

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

            # Record performance
            # ROI = (Final Stack - Initial Stack) / Initial Stack
            # central_agent is at index 0
            final_stack = self.env.players[0].stack
            initial_stack = random_stacks[0]
            roi = (final_stack - initial_stack) / initial_stack

            # with open("training_performance.csv", "a") as f:
            #     f.write(f"{episode+1},{roi}\n")

            # Save periodically
            if (episode + 1) % 5000 == 0 or (episode + 1) == 1:
                model_path = f"models/ac_agent_{episode+1}.pt"
                central_agent.save(model_path)

                # Evaluation 1: vs Random (Every saved model)
                # print(f"\nRunning Evaluation vs Random for {model_path}...")
                import subprocess

                cmd_random = [
                    "python",
                    "main.py",
                    "play",
                    "game_evaluator",
                    f"--model_path={model_path}",
                    "--episodes=10000",
                    "--opponent_probs=1,0,0",
                    "--log_results=evaluation_vs_random.csv",
                    "--log=eval_random",
                ]
                subprocess.run(cmd_random)

            # Evaluation 2: vs Mini LLM (Every 10th saved model, i.e., every 10k episodes)
            if (episode + 1) % 50000 == 0:
                print(f"\nRunning Evaluation vs Mini LLM for {model_path}...")
                cmd_mini = [
                    "python",
                    "main.py",
                    "play",
                    "game_evaluator",
                    f"--model_path={model_path}",
                    "--episodes=100",
                    "--opponent_probs=0,1,0",
                    "--log_results=evaluation_vs_mini.csv",
                    "--log=eval_mini",
                ]
                subprocess.run(cmd_mini)

    def key_press_agents(self):
        """Create an environment with key press agents"""
        from agents.keypress_agent import KeypressAgent

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
    # python main.py play ac_train --episodes=500000
    # python main.py play game_evaluator --model_path=models/ac_agent_500000.pt --episodes=1000 --opponent_probs=1,0,0 --log_results=evaluation_vs_random.csv --log=eval_random
