import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import torch.optim as optim
from collections import deque, defaultdict
import os
import csv
import time

from agents.network import ActorNetwork, CriticNetwork
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

        # Create networks - matching environment's max of 10 players
        self.actor = ActorNetwork(max_players=10, max_action_seq=100).to(self.device)
        self.critic = CriticNetwork(max_players=10, max_action_seq=100).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Training buffers - keyed by seat_id
        self.episode_buffer = defaultdict(list)
        # Buffer for accumulating experiences across episodes for batch update
        self.training_buffer = defaultdict(list)
        self.min_batch_size = 2000  # Minimum steps before performing an update

        # Hyperparameters
        self.gamma = 1
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 1.0
        self.max_grad_norm = 0.5

        # Statistics
        self.training_step = 0
        self.episodes_played = 0
        self.total_reward = 0

        # Logging
        self.log_dir = "log"
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training_losses.csv")

        # Initialize log file with headers if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "policy_loss", "value_loss", "entropy_loss"])

        # print(f"PokerACAgent '{name}' initialized on {self.device}")
        # print(f"Logging losses to {self.log_file}")
        # print(
        #     f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}"
        # )

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
            policy_logits = self.actor(obs_dict)
            value = self.critic(obs_dict)

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
              - community_pot / (fraction of initial stacks)
              - stage (0-4)
              - num_active_players
              - num_callers
              - num_raisers
              - bb_pos

            player_features (6 per player, for 10 players): # 10*6 = 60 values
              for each player:
                - is_active (0/1)
                - did_raise (0/1)
                - money_invested / (fraction of initial stacks)
                - current_stack / (fraction of initial stacks)
                - is_current (0/1)
                - is_bb (0/1)

            PARD sequence (variable length):  # 4 values per action
              - player_seat
              - action_idx
              - reward (fraction of initial stacks)
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

            # Calculate Returns and Advantages (GAE)
            player_returns = []
            player_advantages = []

            rewards = [exp["reward"] for exp in experiences]
            dones = [exp["done"] for exp in experiences]
            values = [exp["value"].item() for exp in agent_buffer]

            # Bootstrap value for the last step (0 if done, else we'd need next_value)
            # In poker, episodes usually end with done=True, so next_value=0 is safe.
            next_value = 0
            gae = 0

            for i in reversed(range(len(rewards))):
                # Delta = r + gamma * V_next * (1-done) - V_curr
                mask = 1.0 - float(dones[i])
                delta = rewards[i] + self.gamma * next_value * mask - values[i]

                # GAE = delta + gamma * lambda * GAE_next * (1-done)
                gae = delta + self.gamma * self.gae_lambda * mask * gae

                # Target Return = Advantage + Value
                # This is what the Critic should predict (V_target)
                player_returns.insert(0, gae + values[i])
                player_advantages.insert(0, gae)

                next_value = values[i]

            # Collect data for batch
            for i in range(len(experiences)):
                agent_exp = agent_buffer[i]
                state = agent_exp["observation"]

                # Store state components
                for key in state.keys():
                    self.training_buffer["states_" + key].append(state[key])

                self.training_buffer["actions"].append(agent_exp["action"])
                self.training_buffer["log_probs"].append(agent_exp["log_prob"].item())
                self.training_buffer["returns"].append(player_returns[i])
                self.training_buffer["advantages"].append(player_advantages[i])

        # Clear episode buffer as we've extracted the data
        self.episode_buffer.clear()

        # Check if we have enough data for an update
        current_batch_size = len(self.training_buffer["actions"])

        # Print buffer fill percentage
        fill_pct = (current_batch_size / self.min_batch_size) * 100
        print(
            f"Buffer: {current_batch_size}/{self.min_batch_size} ({fill_pct:.1f}%)",
            end="\r",
        )

        if current_batch_size < self.min_batch_size:
            return

        print()  # Move to next line for update message

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
        old_log_probs = torch.tensor(
            self.training_buffer["log_probs"], dtype=torch.float32
        ).to(self.device)
        returns = torch.tensor(self.training_buffer["returns"], dtype=torch.float32).to(
            self.device
        )
        advantages = torch.tensor(
            self.training_buffer["advantages"], dtype=torch.float32
        ).to(self.device)

        # Normalize advantages
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(3):  # Multiple epochs
            # Forward pass
            policy_logits = self.actor(states)
            values = self.critic(states)
            values = values.squeeze()

            # Get action probabilities
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)

            # New log probabilities
            new_log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy().mean()

            # PPO Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Surrogate Loss
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values, returns)

            # Loss components
            p_loss = policy_loss
            v_loss = self.value_coef * value_loss
            e_loss = -self.entropy_coef * entropy

            # Optimize Actor
            actor_loss = p_loss + e_loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # Optimize Critic
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

            self.training_step += 1

        print(
            f"  [UPDATE] Network updated at step {self.training_step} (Batch size: {current_batch_size})"
        )
        print(
            f"    Losses -> Policy: {p_loss.item():.4f} | Value: {v_loss.item():.4f} | Entropy: {e_loss.item():.4f}"
        )

        # Log to file
        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.training_step,
                        p_loss.item(),
                        v_loss.item(),
                        e_loss.item(),
                    ]
                )
        except Exception as e:
            print(f"Error writing to log file: {e}")

        # Clear training buffer after update
        self.training_buffer.clear()

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "training_step": self.training_step,
                "episodes_played": self.episodes_played,
                "total_reward": self.total_reward,
            },
            path,
        )
        # print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        if not os.path.exists(path):
            print(f"Warning: Checkpoint {path} not found, starting from scratch")
            return

        checkpoint = torch.load(path, map_location=self.device)

        if "network_state_dict" in checkpoint:
            print("Loading legacy checkpoint (shared network)...")
            net_state = checkpoint["network_state_dict"]

            # Map for Actor
            actor_state = {}
            for k, v in net_state.items():
                if k.startswith("actor"):
                    actor_state[k] = v
                elif k.startswith("critic"):
                    pass
                else:
                    actor_state["feature_extractor." + k] = v

            # Map for Critic
            critic_state = {}
            for k, v in net_state.items():
                if k.startswith("critic"):
                    critic_state[k] = v
                elif k.startswith("actor"):
                    pass
                else:
                    critic_state["feature_extractor." + k] = v

            try:
                self.actor.load_state_dict(actor_state)
                self.critic.load_state_dict(critic_state)
                print("Legacy checkpoint loaded successfully.")
            except Exception as e:
                print(f"Failed to load legacy checkpoint: {e}")
        else:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.actor_optimizer.load_state_dict(
                checkpoint["actor_optimizer_state_dict"]
            )
            self.critic_optimizer.load_state_dict(
                checkpoint["critic_optimizer_state_dict"]
            )

        self.training_step = checkpoint.get("training_step", 0)
        self.episodes_played = checkpoint.get("episodes_played", 0)
        self.total_reward = checkpoint.get("total_reward", 0)
        # print(f"Model loaded from {path}")

    def train_mode(self):
        """Set to training mode"""
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        """Set to evaluation mode"""
        self.actor.eval()
        self.critic.eval()
