"""
Example: How to Use Trajectory Processing

This demonstrates the multi-step trajectory processing workflow:
1. Collect trajectories during an episode
2. Process them with terminal reward
3. Store for training
"""

from agents.agent_actor_critic import Player
import numpy as np

# Create an actor-critic agent
agent = Player(name="TestAgent", state_dim=124, action_dim=15)

# Set initial stack (in real training, trainer does this)
agent.initial_stack = 1000

# ============================================================================
# EXAMPLE EPISODE: 3 actions, terminal reward of +50
# ============================================================================

print("=" * 70)
print("TRAJECTORY PROCESSING EXAMPLE")
print("=" * 70)

# 1. Start episode
print("\n1. COLLECT_TRAJECTORY() - Start new episode")
agent.collect_trajectory()
print(f"   Created empty trajectory: {agent.current_trajectory}")

# 2. Record transitions during episode
print("\n2. RECORD_TRANSITION() - During episode steps")

# Simulate 3 actions with 0 immediate reward
transitions = [
    {
        "state": np.random.randn(124),
        "action": 0,
        "reward": 0,
        "next_state": np.random.randn(124),
        "done": False,
    },
    {
        "state": np.random.randn(124),
        "action": 2,
        "reward": 0,
        "next_state": np.random.randn(124),
        "done": False,
    },
    {
        "state": np.random.randn(124),
        "action": 1,
        "reward": 0,
        "next_state": np.random.randn(124),
        "done": False,
    },
]

for i, trans in enumerate(transitions):
    agent.record_transition(
        state=trans["state"],
        action=trans["action"],
        reward=trans["reward"],
        next_state=trans["next_state"],
        done=trans["done"],
    )
    print(f"   Step {i}: action={trans['action']}, immediate_reward={trans['reward']}")

print(f"   Trajectory length: {len(agent.current_trajectory)}")

# 3. Process trajectory with terminal reward
print("\n3. PROCESS_TRAJECTORY() - End of episode")
terminal_reward = 50  # Agent won 50 chips
gamma = 0.99

print(f"   Terminal Reward: {terminal_reward}")
print(f"   Discount Factor (gamma): {gamma}")
print(f"\n   Backward computation (G_t = r_t + gamma * G_t+1):")

# Show the backward computation
G = 0
for i in range(len(transitions) - 1, -1, -1):
    G = transitions[i]["reward"] + gamma * G
    print(
        f"   Step {i}: G_{i} = {transitions[i]['reward']} + {gamma} * {G/(gamma if i < len(transitions)-1 else 1):.3f} = {G:.3f}"
    )

print(f"\n   These returns are now stored in the replay buffer:")
print(
    f"   - Step 0: store_transition(state0, action=0, reward={transitions[0]['reward'] + gamma * (transitions[1]['reward'] + gamma * (transitions[2]['reward'] + gamma * terminal_reward)):.3f}, ...)"
)
print(
    f"   - Step 1: store_transition(state1, action=2, reward={transitions[1]['reward'] + gamma * (transitions[2]['reward'] + gamma * terminal_reward):.3f}, ...)"
)
print(
    f"   - Step 2: store_transition(state2, action=1, reward={transitions[2]['reward'] + gamma * terminal_reward:.3f}, ...)"
)

# Actually process the trajectory
agent.process_trajectory(terminal_reward=terminal_reward, gamma=gamma)

print(f"\n   Transitions stored in buffer: {len(agent.buffer)}")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print(
    """
1. BACKWARD COMPUTATION:
   - Work backwards from terminal reward
   - Each earlier transition gets discounted credit
   - G_0 < G_1 < G_2 (earlier actions less directly responsible)

2. CREDIT ASSIGNMENT:
   - Every action receives some credit from the terminal reward
   - Discounting means 'causally closer' actions get more credit
   - This is how multi-step credit assignment works

3. TRAINING EFFECT:
   - When critic updates on step 0 with G_0 ≈ 48.5:
     "Action 0 contributed to winning ~48.5 chips"
   - When critic updates on step 2 with G_2 ≈ 49.5:
     "Action 2 contributed to winning ~49.5 chips"

4. vs WITHOUT TRAJECTORY PROCESSING:
   - Would store: reward=0 for all steps, done=True only at end
   - Agent wouldn't learn the connection between actions and outcome
   - With trajectory: agent learns each action's contribution

5. HYPERPARAMETER TUNING:
   - Lower gamma (e.g., 0.9): Each action gets less credit from distant outcomes
   - Higher gamma (e.g., 0.999): Each action gets more credit from outcomes
   - Typical: gamma=0.99 (good balance)
"""
)

print("=" * 70)
