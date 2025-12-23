"""
neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
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

"""

import logging

import gymnasium as gym
import numpy as np
import pandas as pd
from docopt import docopt

import gym_env
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
    _ = get_config()
    init_logger(screenlevel=screenloglevel, filename=logfile)
    print(f"Screenloglevel: {screenloglevel}")
    log = logging.getLogger("")
    log.info("Initializing program")

    if args["selfplay"]:
        num_episodes = 1 if not args["--episodes"] else int(args["--episodes"])
        runner = SelfPlay(
            render=args["--render"],
            num_episodes=num_episodes,
            use_cpp_montecarlo=args["--use_cpp_montecarlo"],
            funds_plot=args["--funds_plot"],
            stack=int(args["--stack"]),
        )

        if args["random"]:
            runner.random_agents()

        elif args["keypress"]:
            runner.key_press_agents()

        elif args["consider_equity"]:
            runner.equity_vs_random()

        elif args["equity_improvement"]:
            improvement_rounds = int(args["--improvement_rounds"])
            runner.equity_self_improvement(improvement_rounds)

        elif args["dqn_train"]:
            runner.dqn_train_keras_rl(model_name)

        elif args["dqn_play"]:
            runner.dqn_play_keras_rl(model_name)

    else:
        raise RuntimeError("Argument not yet implemented")


class SelfPlay:
    """Orchestration of playing against itself"""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500):
        """Initialize"""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with random number of players (2-6) with random stacks and dealer position"""
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"

        for episode in range(self.num_episodes):
            # Randomly select number of players (2-6)
            num_of_plrs = np.random.randint(2, 7)

            # Randomly assign stacks for each player (200-2000)
            player_stacks = np.random.uniform(200, 2000, num_of_plrs)

            # Randomly select dealer position (0 to num_players-1)
            dealer_pos = np.random.randint(0, num_of_plrs)

            self.log.info(
                f"Starting episode {episode + 1}/{self.num_episodes} with {num_of_plrs} players"
            )
            self.log.info(f"Player stacks: {player_stacks}")
            self.log.info(f"Dealer position: {dealer_pos}")

            # Create new environment for this episode
            self.env = gym.make(
                env_name,
                initial_stacks=self.stack,  # This will be overridden per player
                render=self.render,
                funds_plot=self.funds_plot,  # Pass funds_plot to environment
            )

            # Add players with their randomly assigned stacks
            for i in range(num_of_plrs):
                player = RandomPlayer()
                self.env.unwrapped.add_player(player)
                # Override the stack for this player
                self.env.unwrapped.players[i].stack = int(player_stacks[i])

            # Reset with dealer position override via options dict
            obs, info = self.env.reset(options={"dealer_pos": dealer_pos})

            done = False
            total_reward = 0

            while not done:
                # For random agents, the environment handles automatic play
                # We just need to step through until the hand is done
                action = None  # Will be handled by autoplay agents
                obs, reward, done, info = self.env.step(action)
                total_reward += reward

            # Track the winner
            if "winner" in info and info["winner"] is not None:
                self.winner_in_episodes.append(info["winner"])
                self.log.info(
                    f"Episode {episode + 1} winner: Player {info['winner']}, Reward: {total_reward}"
                )
            else:
                self.log.info(
                    f"Episode {episode + 1} completed, Reward: {total_reward}"
                )

        if self.winner_in_episodes:
            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = league_table.index[0]
            self.log.info("League Table (Hand Wins)")
            self.log.info("============")
            self.log.info(league_table)
            self.log.info(f"Best Player: {best_player}")

    def key_press_agents(self):
        """Create an environment with key press agents"""
        from agents.agent_keypress import Player as KeyPressAgent

        env_name = "Holdem_NoLimit-v0"
        num_of_plrs = 3
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        for _ in range(num_of_plrs):
            player = KeyPressAgent()
            self.env.unwrapped.add_player(player)

        # Episode loop
        for episode in range(self.num_episodes):
            self.log.info(f"Starting episode {episode + 1}/{self.num_episodes}")

            # Store initial stacks before reset (which deducts blinds)
            initial_stacks = [self.stack] * num_of_plrs

            self.env.unwrapped.reset()
            self.env.unwrapped.run()

            # Get and output player experiences
            # player_experiences = self.env.unwrapped.get_player_experiences()

            # print("\n" + "=" * 80)
            # print(f"PLAYER EXPERIENCES - EPISODE {episode + 1}")
            # print("=" * 80)

            # for player_id, experiences in player_experiences.items():
            #     player = self.env.unwrapped.players[player_id]
            #     print(
            #         f"\nPlayer {player_id} ({player.name}) - {len(experiences)} experiences:"
            #     )
            #     print("-" * 80)

            #     for exp_idx, exp in enumerate(experiences):
            #         print(f"\n  Experience {exp_idx + 1}:")
            #         print(
            #             f"    State shape: {exp['state'].shape if exp['state'] is not None else 'None'}"
            #         )
            #         print(f"    Action: {exp['action']}")
            #         print(f"    Reward: {exp['reward']:.4f}")
            #         print(
            #             f"    Next state shape: {exp['next_state'].shape if exp['next_state'] is not None else 'None'}"
            #         )
            #         print(f"    Done: {exp['done']}")

            # Output terminal rewards
            # terminal_rewards = self.env.unwrapped.get_terminal_rewards()
            # print("\n" + "=" * 80)
            # print("TERMINAL REWARDS:")
            # print("-" * 80)
            # for player_id, reward in enumerate(terminal_rewards):
            #     player = self.env.unwrapped.players[player_id]
            #     print(f"Player {player_id} ({player.name}): {reward:.4f}")
            # print("=" * 80 + "\n")

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/50", min_call_equity=0.5, min_bet_equity=-0.5)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/80", min_call_equity=0.8, min_bet_equity=-0.8)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/70/70", min_call_equity=0.7, min_bet_equity=-0.7)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=-0.3)
        )
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(RandomPlayer())

        for episode in range(self.num_episodes):
            self.log.info(f"Starting episode {episode + 1}/{self.num_episodes}")
            obs, info = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = None
                obs, reward, done, info = self.env.step(action)
                total_reward += reward

            if "winner" in info and info["winner"] is not None:
                self.winner_in_episodes.append(info["winner"])
                self.log.info(
                    f"Episode {episode + 1} winner: Player {info['winner']}, Reward: {total_reward}"
                )

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer

        calling = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        betting = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        for improvement_round in range(improvement_rounds):
            env_name = "neuron_poker-v0"
            self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
            for i in range(6):
                self.env.unwrapped.add_player(
                    EquityPlayer(
                        name=f"Equity/{calling[i]}/{betting[i]}",
                        min_call_equity=calling[i],
                        min_bet_equity=betting[i],
                    )
                )

            for _ in range(self.num_episodes):
                obs, info = self.env.reset()
                done = False
                while not done:
                    action = None
                    obs, reward, done, info = self.env.step(action)

                if "winner" in info and info["winner"] is not None:
                    self.winner_in_episodes.append(info["winner"])

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve:
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train_keras_rl(self, model_name):
        """Implementation of kreras-rl deep q learing."""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        env = gym.make(
            env_name,
            initial_stacks=self.stack,
            funds_plot=self.funds_plot,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
        )

        np.random.seed(123)
        env.seed(123)
        env.unwrapped.add_player(
            EquityPlayer(name="equity/50/70", min_call_equity=0.5, min_bet_equity=0.7)
        )
        env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=0.3)
        )
        env.unwrapped.add_player(RandomPlayer())
        env.unwrapped.add_player(RandomPlayer())
        env.unwrapped.add_player(RandomPlayer())
        env.unwrapped.add_player(
            PlayerShell(name="keras-rl", stack_size=self.stack)
        )  # shell is used for callback to keras rl

        env.reset()

        dqn = DQNPlayer()
        dqn.initiate_agent(env)
        dqn.train(env_name=model_name)

    def dqn_play_keras_rl(self, model_name):
        """Create 6 players, one of them a trained DQN"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_keras_rl_dqn import Player as DQNPlayer
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/50", min_call_equity=0.5, min_bet_equity=0.5)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/50/80", min_call_equity=0.8, min_bet_equity=0.8)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/70/70", min_call_equity=0.7, min_bet_equity=0.7)
        )
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=0.3)
        )
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(
            PlayerShell(name="keras-rl", stack_size=self.stack)
        )

        self.env.reset()

        dqn = DQNPlayer(load_model=model_name, env=self.env)
        dqn.play(nb_episodes=self.num_episodes, render=self.render)

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random"""
        from agents.agent_consider_equity import Player as EquityPlayer
        from agents.agent_custom_q1 import Player as Custom_Q1
        from agents.agent_random import Player as RandomPlayer

        env_name = "neuron_poker-v0"
        self.env = gym.make(env_name, initial_stacks=self.stack, render=self.render)
        # self.env.unwrapped.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        # self.env.unwrapped.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        # self.env.unwrapped.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.unwrapped.add_player(
            EquityPlayer(name="equity/20/30", min_call_equity=0.2, min_bet_equity=-0.3)
        )
        # self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(RandomPlayer())
        self.env.unwrapped.add_player(Custom_Q1(name="Deep_Q1"))

        for _ in range(self.num_episodes):
            obs, info = self.env.reset()
            done = False
            while not done:
                action = None
                obs, reward, done, info = self.env.step(action)

            if "winner" in info and info["winner"] is not None:
                self.winner_in_episodes.append(info["winner"])

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")


if __name__ == "__main__":
    command_line_parser()
