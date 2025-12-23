"""Training script for Actor-Critic poker agent"""

import logging
import numpy as np
import argparse
from gym_env.holdem_table import HoldemTable
from agent_actor_critic import Player
from gym_env.enums import Action

log = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_env(num_players=3, initial_stacks=100, render=False):
    """Create poker environment"""
    env = HoldemTable(
        initial_stacks=initial_stacks,
        small_blind=1,
        big_blind=2,
        render=render,
        funds_plot=False,
        use_cpp_montecarlo=False,
        raise_illegal_moves=False,
        calculate_equity=False,
    )
    return env


def create_agents(num_players, state_dim=124, action_dim=len(Action) - 2):
    """Create Actor-Critic agents for all players"""
    agents = []

    for i in range(num_players):
        agent = Player(
            name=f"AC_Agent_{i}",
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            learning_rate=1e-4,
            buffer_size=10000,
        )
        agents.append(agent)

    return agents


def train_episode(env, agents, episode_num):
    """Train for one episode (hand)"""
    log.info(f"Starting episode {episode_num}")

    # Reset environment
    obs, info = env.reset()

    # Run the hand
    env.run()

    # Get experiences from environment
    env_experiences = env.get_player_experiences()
    terminal_rewards = env.get_terminal_rewards()

    log.info(f"Episode {episode_num} completed. Winner: {env.winner_ix}")
    log.info(f"Terminal rewards: {terminal_rewards}")

    # Process experiences for each agent
    total_critic_loss = 0
    total_actor_loss = 0
    trained_agents = 0

    for i, agent in enumerate(agents):
        if i in env_experiences:
            # Process this agent's experiences
            agent.process_env_experiences(env_experiences, i)

            # Get latest training loss
            critic_loss, actor_loss = agent.train_on_batch(batch_size=32, gamma=0.99)

            if critic_loss is not None and actor_loss is not None:
                total_critic_loss += critic_loss
                total_actor_loss += actor_loss
                trained_agents += 1

    # Calculate average losses
    avg_critic_loss = total_critic_loss / trained_agents if trained_agents > 0 else 0
    avg_actor_loss = total_actor_loss / trained_agents if trained_agents > 0 else 0

    # Log agent performance
    for i, agent in enumerate(agents):
        log.info(f"Agent {i} buffer size: {len(agent.state_buffer)}")

    return avg_critic_loss, avg_actor_loss, env.winner_ix


def evaluate_agents(env, agents, num_eval_hands=100):
    """Evaluate agents' performance"""
    log.info("Evaluating agents...")

    wins = [0] * len(agents)
    total_profits = [0] * len(agents)

    for hand_num in range(num_eval_hands):
        # Reset environment
        obs, info = env.reset()

        # Run hand
        env.run()

        # Record results
        winner = env.winner_ix
        if winner is not None:
            wins[winner] += 1

        # Calculate profits
        for i, player in enumerate(env.players):
            profit = player.stack - env.initial_stacks
            total_profits[i] += profit

    # Calculate win rates
    win_rates = [w / num_eval_hands for w in wins]
    avg_profits = [p / num_eval_hands for p in total_profits]

    log.info(f"Win rates: {win_rates}")
    log.info(f"Average profits per hand: {avg_profits}")

    return win_rates, avg_profits


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Actor-Critic poker agents")
    parser.add_argument("--num_players", type=int, default=3, help="Number of players")
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--initial_stacks", type=int, default=100, help="Initial stack size"
    )
    parser.add_argument(
        "--eval_every", type=int, default=100, help="Evaluate every N episodes"
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument(
        "--eval_only", action="store_true", help="Only evaluate, no training"
    )

    args = parser.parse_args()

    setup_logging()

    # Create environment
    env = create_env(
        num_players=args.num_players,
        initial_stacks=args.initial_stacks,
        render=args.render,
    )

    # Create agents
    agents = create_agents(args.num_players)

    # Add agents to environment
    for agent in agents:
        env.add_player(agent)

    if args.eval_only:
        # Evaluation only
        evaluate_agents(env, agents, num_eval_hands=100)
        return

    # Training loop
    for episode in range(args.num_episodes):
        # Train one episode
        critic_loss, actor_loss, winner = train_episode(env, agents, episode)

        # Log training progress
        if episode % 10 == 0:
            log.info(
                f"Episode {episode}: Critic Loss = {critic_loss:.4f}, Actor Loss = {actor_loss:.4f}, Winner = {winner}"
            )

        # Evaluate periodically
        if args.eval_every > 0 and episode % args.eval_every == 0 and episode > 0:
            log.info(f"Evaluating after episode {episode}...")
            win_rates, avg_profits = evaluate_agents(env, agents, num_eval_hands=50)

            # Log evaluation results
            for i, (win_rate, avg_profit) in enumerate(zip(win_rates, avg_profits)):
                log.info(
                    f"Agent {i}: Win Rate = {win_rate:.2%}, Avg Profit = {avg_profit:.2f}"
                )

    # Final evaluation
    log.info("Final evaluation...")
    win_rates, avg_profits = evaluate_agents(env, agents, num_eval_hands=100)

    # Save models (optional)
    # for i, agent in enumerate(agents):
    #     agent.save_model(f"agent_{i}_final.h5")


if __name__ == "__main__":
    main()
