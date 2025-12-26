import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import torch.optim as optim
from collections import deque, defaultdict
import os

from agents.network import DynamicPokerNetwork
from gym_env.enums import Action


class PokerACAgent:
    """
    Actor-Critic agent for Texas Hold'em poker with variable players and self-play.
    Designed to work with the HoldemTable environment.
    """

    def __init__(self, name="ACAgent", device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.name = name

        # Create network - matching environment's max of 10 players
        self.network = DynamicPokerNetwork(max_players=10, max_action_seq=100).to(
            self.device
        )

        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=3e-4)

        # Training buffers - keyed by seat_id
        self.episode_buffer = defaultdict(list)
        # Buffer for accumulating experiences across episodes for batch update
        self.training_buffer = defaultdict(list)
        self.min_batch_size = 2000  # Minimum steps before performing an update

        self.replay_buffer = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        # Statistics
        self.training_step = 0
        self.episodes_played = 0
        self.total_reward = 0

        print(f"PokerACAgent '{name}' initialized on {self.device}")
        print(
            f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}"
        )

    def action(self, action_space, observation, info):
        """
        Required interface method - returns action for current player.

        Args:
            action_space: Environment action space
            observation: Raw observation array from environment
            info: Additional info dict

        Returns:
            action: Selected action index
            action_info: Dictionary with action info
        """
        # Parse observation into network format
        obs_dict = self._parse_observation(observation)

        # Get legal actions from environment's legal_moves
        legal_actions = (
            [a.value for a in action_space] if action_space else list(range(13))
        )

        # Forward pass through network
        with torch.no_grad():
            policy_logits, value = self.network(obs_dict)

            # Create mask for illegal actions
            action_mask = torch.full((13,), float("-inf"), device=self.device)
            for act in legal_actions:
                if 0 <= act < 13:
                    action_mask[act] = 0.0

            # Apply mask
            masked_logits = policy_logits + action_mask.unsqueeze(0)

            # Sample action
            action_probs = F.softmax(masked_logits, dim=-1)

            # Handle potential numerical instability
            if torch.isnan(action_probs).any():
                action = 0  # Default to FOLD
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()

            # Ensure action is legal (fallback)
            if action not in legal_actions:
                action = legal_actions[0] if legal_actions else 0

            # Get log probability
            log_prob = action_dist.log_prob(torch.tensor([action]).to(self.device))

        # Store transition for training
        # Get seat from info to separate experiences by player
        seat = info.get("player_data", {}).get("position", 0)
        if isinstance(seat, str):  # Handle "Unknown" case
            seat = 0

        self.episode_buffer[seat].append(
            {
                "observation": obs_dict,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "legal_actions": legal_actions,
                "reward": 0.0,  # Will be filled later
                "done": False,
            }
        )

        # Return action and info
        return action, {
            "log_prob": log_prob.item(),
            "value": value.item(),
            "legal_actions": legal_actions,
            "action_name": Action(action).name,
        }

    def _parse_observation(self, observation: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Parse the observation from HoldemTable environment into network input.

        Observation format (based on env._get_environment()):
        [
            hole_card1, hole_card2,           # 2 values (-1 if no card)
            comm_card1-5,                     # 5 values (-1 if no card)

            global_features (7):              # 7 values
              - num_players
              - community_pot / (big_blind * 100)
              - stage (0-4)
              - num_active_players
              - num_callers
              - num_raisers
              - bb_pos

            player_features (6 per player, for 10 players): # 10*6 = 60 values
              for each player:
                - is_active (0/1)
                - did_raise (0/1)
                - money_invested / (big_blind * 100)
                - current_stack / (big_blind * 100)
                - is_current (0/1)
                - is_bb (0/1)

            PARD sequence (variable length):  # 4 values per action
              - player_seat
              - action_idx
              - reward / (big_blind * 100)
              - done (0/1)
        ]

        Total length: 2 + 5 + 7 + 60 = 74 fixed + variable PARD
        """
        # Convert to numpy array if needed
        if isinstance(observation, list):
            observation = np.array(observation, dtype=np.float32)
        elif torch.is_tensor(observation):
            observation = observation.cpu().numpy()

        obs = observation.flatten() if len(observation.shape) > 1 else observation
        idx = 0

        # 1. Hole cards (2)
        hole_cards = np.zeros(2, dtype=np.int32)
        for i in range(2):
            if idx < len(obs):
                hole_cards[i] = int(obs[idx])
            idx += 1

        # 2. Community cards (5)
        community_cards = np.zeros(5, dtype=np.int32)
        for i in range(5):
            if idx < len(obs):
                community_cards[i] = int(obs[idx])
            idx += 1

        # 3. Global features (7)
        global_features = np.zeros(7, dtype=np.float32)
        if idx + 7 <= len(obs):
            global_features = obs[idx : idx + 7].astype(np.float32)
        idx += 7

        # 4. Player features (10 players * 6 features = 60)
        player_features = np.zeros((10, 6), dtype=np.float32)
        player_mask = np.zeros(10, dtype=np.float32)

        for i in range(10):
            if idx + 6 <= len(obs):
                player_features[i] = obs[idx : idx + 6]
                # Use is_active (first feature) as mask
                player_mask[i] = player_features[i, 0]
                idx += 6

        # 5. PARD sequence (variable length)
        max_actions = 100  # Network's max
        action_sequence = np.zeros((max_actions, 4), dtype=np.float32)
        action_mask = np.zeros(max_actions, dtype=np.float32)

        remaining_len = len(obs) - idx
        action_count = min(remaining_len // 4, max_actions)

        for i in range(action_count):
            if idx + 4 <= len(obs):
                action_sequence[i] = obs[idx : idx + 4]
                action_mask[i] = 1.0
                idx += 4

        # Convert to tensors
        obs_dict = {
            "hole_cards": torch.from_numpy(hole_cards).unsqueeze(0).to(self.device),
            "community_cards": torch.from_numpy(community_cards)
            .unsqueeze(0)
            .to(self.device),
            "player_features": torch.from_numpy(player_features)
            .unsqueeze(0)
            .to(self.device),
            "player_mask": torch.from_numpy(player_mask).unsqueeze(0).to(self.device),
            "global_features": torch.from_numpy(global_features)
            .unsqueeze(0)
            .to(self.device),
            "action_sequence": torch.from_numpy(action_sequence)
            .unsqueeze(0)
            .to(self.device),
            "action_mask": torch.from_numpy(action_mask).unsqueeze(0).to(self.device),
        }

        return obs_dict

    def update(self, all_experiences: Dict[int, List[Dict]]):
        """
        Update policy using experiences from all players in the episode.
        Accumulates experiences and only runs optimization when batch size is reached.

        Args:
            all_experiences: Dict mapping seat_id to list of experience dicts
                             (from env.get_player_experiences())
        """
        if not all_experiences or not self.episode_buffer:
            return

        # Process each player's experiences
        for seat_id, experiences in all_experiences.items():
            if seat_id not in self.episode_buffer:
                continue

            agent_buffer = self.episode_buffer[seat_id]

            # Verify alignment
            if len(experiences) != len(agent_buffer):
                min_len = min(len(experiences), len(agent_buffer))
                experiences = experiences[:min_len]
                agent_buffer = agent_buffer[:min_len]

            if not experiences:
                continue

            # Calculate Returns (Monte Carlo)
            R = 0
            player_returns = []
            player_values = []

            rewards = [exp["reward"] for exp in experiences]
            values = [exp["value"].item() for exp in agent_buffer]

            for r in reversed(rewards):
                R = r + self.gamma * R
                player_returns.insert(0, R)

            # Calculate advantages
            player_advantages = [ret - val for ret, val in zip(player_returns, values)]

            # Collect data for batch
            for i in range(len(experiences)):
                agent_exp = agent_buffer[i]
                state = agent_exp["observation"]

                # Store state components
                for key in state.keys():
                    self.training_buffer["states_" + key].append(state[key])

                self.training_buffer["actions"].append(agent_exp["action"])
                self.training_buffer["returns"].append(player_returns[i])
                self.training_buffer["advantages"].append(player_advantages[i])

        # Clear episode buffer as we've extracted the data
        self.episode_buffer.clear()

        # Check if we have enough data for an update
        current_batch_size = len(self.training_buffer["actions"])
        if current_batch_size < self.min_batch_size:
            return

        # --- Perform Update ---

        # Stack tensors
        states = {}
        state_keys = [
            "hole_cards",
            "community_cards",
            "player_features",
            "player_mask",
            "global_features",
            "action_sequence",
            "action_mask",
        ]
        for key in state_keys:
            states[key] = torch.cat(self.training_buffer["states_" + key], dim=0)

        actions = torch.tensor(self.training_buffer["actions"], dtype=torch.long).to(
            self.device
        )
        returns = torch.tensor(self.training_buffer["returns"], dtype=torch.float32).to(
            self.device
        )
        advantages = torch.tensor(
            self.training_buffer["advantages"], dtype=torch.float32
        ).to(self.device)

        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO-style update
        for _ in range(3):  # Multiple epochs
            # Forward pass
            policy_logits, values = self.network(states)
            values = values.squeeze()

            # Get action probabilities
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)

            # New log probabilities
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # Policy loss
            policy_loss = -(new_log_probs * advantages).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = (
                policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            self.training_step += 1

        # Clear training buffer after update
        self.training_buffer.clear()

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "network_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "episodes_played": self.episodes_played,
                "total_reward": self.total_reward,
            },
            path,
        )
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"Warning: Checkpoint {path} not found, starting from scratch")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
        self.total_reward = checkpoint.get("total_reward", 0)
        print(f"Model loaded from {path}")

    def train_mode(self):
        """Set to training mode"""
        self.network.train()

    def eval_mode(self):
        """Set to evaluation mode"""
        self.network.eval()


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
