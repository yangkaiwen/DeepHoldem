"""Actor-Critic Self-Play Training for Texas Hold'em"""

import logging
import numpy as np
import gymnasium as gym
from agents.agent_actor_critic import Player as ACPlayer
from agents.agent_random import Player as RandomPlayer

log = logging.getLogger(__name__)


class ActorCriticTrainer:
    """Trainer for Actor-Critic agents in self-play"""

    def __init__(
        self,
        num_ac_agents=2,
        initial_stack=500,
        learning_rate=1e-4,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
    ):
        """
        Initialize trainer.

        Args:
            num_ac_agents: Number of AC agents to train
            initial_stack: Initial stack for each player
            learning_rate: Learning rate for networks
            hidden_dim: Hidden dimension for networks
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        self.num_ac_agents = num_ac_agents
        self.initial_stack = initial_stack
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Create AC agents
        self.ac_agents = [
            ACPlayer(
                name=f"AC_Agent_{i}",
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                learning_rate=learning_rate,
            )
            for i in range(num_ac_agents)
        ]

        self.env = None
        self.episode_stats = []

    def train_minibatch(
        self, num_episodes=10, batch_size=32, gamma=0.99, min_players=2, max_players=6
    ):
        """
        Train for a minibatch of episodes.

        Args:
            num_episodes: Number of episodes in this minibatch
            batch_size: Batch size for network updates
            gamma: Discount factor
            min_players: Minimum number of players per episode
            max_players: Maximum number of players per episode
        """
        episode_rewards = {agent.name: [] for agent in self.ac_agents}

        for episode in range(num_episodes):
            # Random number of players
            num_total_players = np.random.randint(min_players, max_players + 1)
            num_random_agents = max(1, num_total_players - self.num_ac_agents)

            # Random stacks
            stacks = np.random.uniform(200, 2000, num_total_players)
            dealer_pos = np.random.randint(0, num_total_players)

            log.info(
                f"Episode {episode + 1}/{num_episodes}: {self.num_ac_agents} AC + {num_random_agents} Random"
            )
            log.info(f"Stacks: {stacks}, Dealer: {dealer_pos}")

            # Create environment
            self.env = gym.make(
                "neuron_poker-v0",
                initial_stacks=self.initial_stack,
                render=False,
                funds_plot=False,
            )

            # Add AC agents
            ac_seats = {}
            for i, agent in enumerate(self.ac_agents):
                self.env.unwrapped.add_player(agent)
                ac_seats[agent.name] = i
                self.env.unwrapped.players[i].stack = int(stacks[i])
                agent.initial_stack = int(
                    stacks[i]
                )  # Track initial stack for reward calculation

            # Add random agents
            for i in range(num_random_agents):
                random_agent = RandomPlayer()
                self.env.unwrapped.add_player(random_agent)
                ac_idx = self.num_ac_agents + i
                self.env.unwrapped.players[ac_idx].stack = int(stacks[ac_idx])

            # Run episode
            obs, info = self.env.reset(options={"dealer_pos": dealer_pos})

            # Initialize trajectory collection for all AC agents
            for agent in self.ac_agents:
                agent.collect_trajectory()

            done = False
            step_count = 0
            agent_actions = {agent.name: 0 for agent in self.ac_agents}

            while not done:
                current_agent = self.env.unwrapped.current_player.agent_obj
                current_seat = self.env.unwrapped.acting_agent

                # Store observation before action
                obs_before = obs.copy()

                # Take step
                obs, reward, done, _, info = self.env.step(None)

                # Record transition for AC agents (intermediate step)
                if isinstance(current_agent, ACPlayer):
                    current_agent.record_transition(
                        obs_before,
                        (
                            self.env.unwrapped.current_player.actions[-1].value
                            if self.env.unwrapped.current_player.actions
                            else 0
                        ),
                        reward,
                        obs,
                        done,
                    )
                    agent_actions[current_agent.name] += 1

                step_count += 1
                if step_count > 1000:  # Safety check
                    log.warning("Episode exceeded 1000 steps, terminating")
                    break

            # Process trajectories for all AC agents with terminal rewards
            for agent in self.ac_agents:
                # Get final stack for reward calculation
                agent_stack = None
                for i, p in enumerate(self.env.unwrapped.players):
                    if p.agent_obj is agent:
                        agent_stack = p.stack
                        break

                # Terminal reward is the change in stack
                terminal_reward = (
                    agent_stack - agent.initial_stack if agent_stack else 0
                )

                # Process trajectory: compute returns and store transitions
                agent.process_trajectory(terminal_reward=terminal_reward, gamma=gamma)

                # Train on batch
                if len(agent.buffer) >= batch_size:
                    critic_loss, actor_loss = agent.train_on_batch(batch_size, gamma)
                    if critic_loss is not None:
                        log.info(
                            f"{agent.name} - Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}"
                        )
                        episode_rewards[agent.name].append(
                            {
                                "critic_loss": critic_loss,
                                "actor_loss": actor_loss,
                                "num_actions": agent_actions[agent.name],
                                "terminal_reward": terminal_reward,
                            }
                        )
                else:
                    log.debug(
                        f"{agent.name} - Buffer size {len(agent.buffer)} < batch size {batch_size}"
                    )

            self.env.close()

        return episode_rewards

    def evaluate(self, num_episodes=5, opponent_types=["random", "equity"]):
        """
        Evaluate trained agents against different opponents.

        Args:
            num_episodes: Number of evaluation episodes
            opponent_types: Types of opponents to play against
        """
        results = {
            agent.name: {"wins": 0, "total_reward": 0} for agent in self.ac_agents
        }

        for episode in range(num_episodes):
            for opponent_type in opponent_types:
                self.env = gym.make(
                    "neuron_poker-v0", initial_stacks=self.initial_stack, render=False
                )

                # Add AC agents
                for agent in self.ac_agents:
                    self.env.unwrapped.add_player(agent)

                # Add opponents
                if opponent_type == "random":
                    for _ in range(2):
                        self.env.unwrapped.add_player(RandomPlayer())
                elif opponent_type == "equity":
                    from agents.agent_consider_equity import Player as EquityPlayer

                    self.env.unwrapped.add_player(EquityPlayer(name="equity/50/50"))
                    self.env.unwrapped.add_player(RandomPlayer())

                obs, info = self.env.reset()
                done = False
                episode_rewards = {agent.name: 0 for agent in self.ac_agents}

                while not done:
                    obs, reward, done, _, info = self.env.step(None)

                    if self.env.unwrapped.current_player.agent_obj in self.ac_agents:
                        agent_name = self.env.unwrapped.current_player.agent_obj.name
                        episode_rewards[agent_name] += reward

                for agent in self.ac_agents:
                    results[agent.name]["total_reward"] += episode_rewards[agent.name]
                    if info.get("winner") == self.env.unwrapped.players.index(agent):
                        results[agent.name]["wins"] += 1

                self.env.close()

        return results


def main():
    """Example training script"""
    import sys

    logging.basicConfig(level=logging.INFO)

    # Initialize trainer
    trainer = ActorCriticTrainer(
        num_ac_agents=2,
        initial_stack=500,
        learning_rate=1e-4,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
    )

    # Training loop
    num_minibatches = 5
    episodes_per_minibatch = 10

    for minibatch in range(num_minibatches):
        log.info(f"\n{'='*60}")
        log.info(f"Minibatch {minibatch + 1}/{num_minibatches}")
        log.info(f"{'='*60}")

        rewards = trainer.train_minibatch(
            num_episodes=episodes_per_minibatch,
            batch_size=32,
            gamma=0.99,
            min_players=2,
            max_players=6,
        )

        log.info("Minibatch Results:")
        for agent_name, agent_rewards in rewards.items():
            if agent_rewards:
                avg_critic_loss = np.mean([r["critic_loss"] for r in agent_rewards])
                avg_actor_loss = np.mean([r["actor_loss"] for r in agent_rewards])
                log.info(
                    f"  {agent_name}: Critic Loss={avg_critic_loss:.4f}, Actor Loss={avg_actor_loss:.4f}"
                )

        # Evaluate periodically
        if (minibatch + 1) % 2 == 0:
            log.info("\nEvaluating...")
            eval_results = trainer.evaluate(num_episodes=3)
            for agent_name, results in eval_results.items():
                log.info(
                    f"  {agent_name}: Wins={results['wins']}, Avg Reward={results['total_reward']/3:.2f}"
                )

    log.info("\nTraining completed!")


if __name__ == "__main__":
    main()
