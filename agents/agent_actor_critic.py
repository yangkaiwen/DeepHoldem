"""Actor-Critic Agent for Texas Hold'em"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

log = logging.getLogger(__name__)


class ActorCriticNetwork:
    """Actor-Critic network with transformer-based architecture"""

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        learning_rate=1e-4,
    ):
        """
        Initialize Actor-Critic network.

        Args:
            state_dim: Observation space dimension
            action_dim: Action space dimension (number of possible actions)
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            learning_rate: Learning rate for optimizers
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Build shared feature extraction
        self.feature_extractor = self._build_feature_extractor(
            state_dim, hidden_dim, num_heads, num_layers
        )

        # Actor network (policy)
        self.actor = self._build_actor(hidden_dim, action_dim)

        # Critic network (value)
        self.critic = self._build_critic(hidden_dim)

        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    def _build_feature_extractor(self, state_dim, hidden_dim, num_heads, num_layers):
        """Build feature extraction network"""
        inputs = keras.Input(shape=(state_dim,))
        x = keras.layers.Dense(hidden_dim, activation="relu")(inputs)

        for _ in range(num_layers):
            # Multi-head attention
            attn_output = keras.layers.MultiHeadAttention(num_heads=num_heads)(x, x)
            x = keras.layers.Add()([x, attn_output])
            x = keras.layers.LayerNormalization()(x)

            # Feed-forward
            ff = keras.layers.Dense(hidden_dim * 2, activation="relu")(x)
            ff = keras.layers.Dense(hidden_dim)(ff)
            x = keras.layers.Add()([x, ff])
            x = keras.layers.LayerNormalization()(x)

        return keras.Model(inputs=inputs, outputs=x)

    def _build_actor(self, hidden_dim, action_dim):
        """Build actor network (policy)"""
        inputs = keras.Input(shape=(hidden_dim,))
        x = keras.layers.Dense(hidden_dim, activation="relu")(inputs)
        x = keras.layers.Dense(hidden_dim // 2, activation="relu")(x)
        outputs = keras.layers.Dense(action_dim, activation="softmax")(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def _build_critic(self, hidden_dim):
        """Build critic network (value function)"""
        inputs = keras.Input(shape=(hidden_dim,))
        x = keras.layers.Dense(hidden_dim, activation="relu")(inputs)
        x = keras.layers.Dense(hidden_dim // 2, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)
        return keras.Model(inputs=inputs, outputs=outputs)

    def get_action_and_value(self, state):
        """Get action probabilities and value estimate from state"""
        features = self.feature_extractor(np.array([state]))
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs[0].numpy(), value[0][0].numpy()

    def get_action(self, state, legal_actions):
        """Sample action from policy, respecting legal actions"""
        action_probs, _ = self.get_action_and_value(state)

        # Mask illegal actions
        legal_mask = np.zeros(len(action_probs))
        for action in legal_actions:
            legal_mask[action.value] = 1

        masked_probs = action_probs * legal_mask
        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)

        # Sample action
        action_idx = np.random.choice(len(action_probs), p=masked_probs)
        return action_idx

    def update(self, state, action, reward, next_state, done, gamma=0.99):
        """Update actor and critic networks"""
        state = np.array([state])
        next_state = np.array([next_state])

        features = self.feature_extractor(state)
        next_features = self.feature_extractor(next_state)

        with tf.GradientTape() as tape:
            value = self.critic(features)
            next_value = self.critic(next_features)

            # Calculate TD target
            td_target = reward + gamma * next_value * (1 - done)
            td_error = td_target - value

            # Critic loss
            critic_loss = tf.reduce_mean(tf.square(td_error))

        # Update critic
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables)
        )

        # Actor update
        with tf.GradientTape() as tape:
            action_probs = self.actor(features)
            action_log_prob = tf.math.log(action_probs[0, action] + 1e-8)

            # Actor loss (policy gradient)
            actor_loss = -action_log_prob * tf.stop_gradient(td_error)[0, 0]

        # Update actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )

        return float(critic_loss), float(actor_loss)


class Player:
    """Actor-Critic Agent Player"""

    def __init__(
        self,
        name="AC_Agent",
        state_dim=124,
        action_dim=15,
        hidden_dim=256,
        num_heads=4,
        num_layers=2,
        learning_rate=1e-4,
    ):
        """
        Initialize Actor-Critic agent.

        Args:
            name: Player name
            state_dim: Observation space dimension
            action_dim: Number of actions
            hidden_dim: Hidden dimension for network
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            learning_rate: Learning rate
        """
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_stack = 0  # Set by trainer before episode

        # Network
        self.network = ActorCriticNetwork(
            state_dim, action_dim, hidden_dim, num_heads, num_layers, learning_rate
        )

        # Experience buffer
        self.state_buffer = deque(maxlen=1000)
        self.action_buffer = deque(maxlen=1000)
        self.reward_buffer = deque(maxlen=1000)
        self.next_state_buffer = deque(maxlen=1000)
        self.done_buffer = deque(maxlen=1000)

        # Trajectory collection
        self.current_trajectory = []

    @property
    def buffer(self):
        """Return buffer deque-like object for size checking"""
        return self.state_buffer

    def action(self, legal_moves, observation, info):
        """Choose action based on policy"""
        action_idx = self.network.get_action(observation, legal_moves)
        return action_idx

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in buffer"""
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.done_buffer.append(done)

    def collect_trajectory(self):
        """
        Start collecting trajectory for this episode.
        Call this at the start of each episode.
        """
        self.current_trajectory = []

    def record_transition(self, state, action, reward, next_state, done):
        """
        Record a single transition during the episode.

        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward (can be 0 for intermediate steps)
            next_state: Resulting state
            done: Whether episode ended
        """
        self.current_trajectory.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
            }
        )

    def process_trajectory(self, terminal_reward=0, gamma=0.99):
        """
        Process trajectory and store transitions with computed returns.

        This handles the case where an agent takes multiple actions during a hand
        and receives a single terminal reward at the end.

        The key insight: we work backwards from the terminal reward to compute
        the discounted return (G_t) for each step:

        G_t = r_t + gamma * G_{t+1}
        G_T = terminal_reward (at the end)

        Args:
            terminal_reward: Final reward received at end of episode
            gamma: Discount factor
        """
        if not hasattr(self, "current_trajectory") or not self.current_trajectory:
            return

        # Work backwards from terminal reward
        # For the last step, the return is the immediate reward + terminal reward
        trajectory = self.current_trajectory
        returns = [0] * len(trajectory)

        # Initialize return at the end
        G = terminal_reward

        # Work backwards through trajectory
        for t in range(len(trajectory) - 1, -1, -1):
            # G_t = r_t + gamma * G_{t+1}
            G = trajectory[t]["reward"] + gamma * G
            returns[t] = G

        # Store all transitions with computed returns
        for t, trans in enumerate(trajectory):
            # Use computed return as the reward
            self.store_transition(
                state=trans["state"],
                action=trans["action"],
                reward=returns[t],  # Use computed return, not immediate reward
                next_state=trans["next_state"],
                done=trans["done"],
            )

        # Clear trajectory
        self.current_trajectory = []

    def process_trajectory_bootstrap(self, terminal_value=0, gamma=0.99):
        """
        Alternative: use bootstrapping with final state value.

        Instead of using terminal reward, use the critic's estimate of the
        final state value. This is better when the hand doesn't always end
        with explicit terminal rewards.

        Args:
            terminal_value: Critic's estimate of final state value
            gamma: Discount factor
        """
        if not hasattr(self, "current_trajectory") or not self.current_trajectory:
            return

        trajectory = self.current_trajectory

        # Work backwards with bootstrapping
        G = terminal_value

        for t in range(len(trajectory) - 1, -1, -1):
            G = trajectory[t]["reward"] + gamma * G

            self.store_transition(
                state=trajectory[t]["state"],
                action=trajectory[t]["action"],
                reward=G,
                next_state=trajectory[t]["next_state"],
                done=trajectory[t]["done"],
            )

        self.current_trajectory = []

    def train_on_batch(self, batch_size=32, gamma=0.99):
        """Train network on batch of experiences"""
        if len(self.state_buffer) < batch_size:
            return None, None

        # Sample batch
        indices = random.sample(range(len(self.state_buffer)), batch_size)

        critic_losses = []
        actor_losses = []

        for idx in indices:
            state = self.state_buffer[idx]
            action = self.action_buffer[idx]
            reward = self.reward_buffer[idx]
            next_state = self.next_state_buffer[idx]
            done = self.done_buffer[idx]

            critic_loss, actor_loss = self.network.update(
                state, action, reward, next_state, done, gamma
            )
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

        return np.mean(critic_losses), np.mean(actor_losses)

    def __repr__(self):
        return f"ActorCriticAgent({self.name})"
