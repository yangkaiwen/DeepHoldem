import torch
import numpy as np
from gym_env.env import HoldemTable
from agents.ac_agent import PokerACAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from agents.keypress_agent import KeypressAgent

# Global agent for the worker process
worker_agent = None

def init_worker():
    """Initialize the worker process with a local agent instance."""
    global worker_agent
    # Force CPU for workers to avoid CUDA context overhead/errors in subprocesses
    worker_agent = PokerACAgent(name="WorkerAC", device="cpu")
    worker_agent.actor.eval()
    worker_agent.critic.eval()

def run_evaluation_episode(actor_state, critic_state, model_path, num_players, initial_stack, opponent_probabilities):
    """
    Run a single evaluation episode in the worker process.
    """
    global worker_agent
    
    # Initialize AC agent if needed
    if model_path != "keypress" and model_path != "random":
        if worker_agent is None:
            init_worker()
        # Load weights
        if actor_state and critic_state:
            worker_agent.actor.load_state_dict(actor_state)
            worker_agent.critic.load_state_dict(critic_state)
    
    # Determine main agent key
    if model_path == "keypress":
        main_agent_key = "keypress_agent"
    elif model_path == "random":
        main_agent_key = "random_agent_main"
    else:
        main_agent_key = "ac_agent"

    # Determine num_players
    n_players = num_players if num_players else np.random.randint(2, 10)

    # Determine initial stack
    stacks = []
    if initial_stack:
        stacks = [int(initial_stack)] * n_players
    else:
        # Random between 400 and 4000
        stacks = np.random.randint(400, 4001, size=n_players).tolist()

    # Create Env
    env = HoldemTable(initial_stacks=stacks[0])

    # Add Agents
    agent_types = []  # Track type for ROI mapping

    if model_path == "keypress":
        env.add_player(KeypressAgent(name="Keypress"))
        agent_types.append(main_agent_key)
    elif model_path == "random":
        env.add_player(RandomAgent(name="Random_Main"))
        agent_types.append(main_agent_key)
    else:
        env.add_player(worker_agent)
        agent_types.append(main_agent_key)

    # Add Opponents
    # Probabilities: [Random, Mini, Large]
    for i in range(1, n_players):
        rand = np.random.random()
        p_random, p_mini, p_large = opponent_probabilities

        # Normalize probabilities to sum to 1 if they don't
        total_p = sum(opponent_probabilities)
        if total_p > 0:
            p_random /= total_p
            p_mini /= total_p
            p_large /= total_p

        if rand < p_random:
            env.add_player(RandomAgent(name=f"Random_{i}"))
            agent_types.append("random_agent")
        elif rand < p_random + p_mini:
            # Mini
            env.add_player(
                LLMAgent(name=f"Mini_{i}", model="xiaomi/mimo-v2-flash:free")
            )
            agent_types.append("llm_agent_mini")
        else:
            # Large
            env.add_player(
                LLMAgent(
                    name=f"Large_{i}",
                    model="meta-llama/llama-3.3-70b-instruct:free",
                )
            )
            agent_types.append("llm_agent_large")

    # Run Episode
    dealer_pos = np.random.randint(0, n_players)
    env.reset(options={"stacks": stacks, "dealer_pos": dealer_pos})
    env.run()

    # Collect results
    results = []
    for i, player in enumerate(env.players):
        initial = env.hand_starting_stacks[i]
        final = player.stack
        investment = env.player_max_win[i]
        a_type = agent_types[i]
        
        results.append({
            "agent_type": a_type,
            "initial": initial,
            "final": final,
            "investment": investment
        })
        
    return results

def run_episode_task(actor_state, critic_state, stack, use_llm=False):
    """
    Run a single episode in the worker process.
    
    Args:
        actor_state: State dict for the actor network
        critic_state: State dict for the critic network
        stack: Initial stack size
        use_llm: Whether to use LLM agents (not fully implemented in parallel yet)
    
    Returns:
        env_experiences: Experiences from the environment
        agent_buffer: Experiences from the agent (with tensors)
        stats: Dict with episode statistics (roi, etc.)
    """
    global worker_agent
    if worker_agent is None:
        init_worker()
    
    # Load latest weights
    worker_agent.actor.load_state_dict(actor_state)
    worker_agent.critic.load_state_dict(critic_state)
    
    # Clear buffer for the new episode
    worker_agent.episode_buffer.clear()
    
    # Setup environment
    # Randomize number of players (2-10)
    num_players = np.random.randint(2, 11)
    env = HoldemTable(initial_stacks=stack)
    
    # Add players
    env.add_player(worker_agent)
    
    if use_llm:
        # Placeholder for LLM logic if needed
        # For now, fallback to self-play or random if LLM not safe to pickle
        pass
    else:
        # Self-play: fill table with the same agent
        for i in range(1, num_players):
            env.add_player(worker_agent)
            
    # Run episode
    bb = env.big_blind
    random_stacks = np.random.uniform(200 * bb, 2000 * bb, num_players)
    dealer_pos = np.random.randint(0, num_players)
    
    env.reset(options={"stacks": random_stacks, "dealer_pos": dealer_pos})
    env.run()
    
    env_experiences = env.get_player_experiences()
    agent_buffer = worker_agent.episode_buffer
    
    # Move agent buffer tensors to CPU and detach to ensure they are picklable and don't hold graph
    # (They should already be on CPU since device="cpu", but good to be safe)
    cpu_agent_buffer = {}
    for seat, experiences in agent_buffer.items():
        cpu_agent_buffer[seat] = []
        for exp in experiences:
            new_exp = {}
            for k, v in exp.items():
                if isinstance(v, torch.Tensor):
                    new_exp[k] = v.cpu().detach()
                elif isinstance(v, dict): # observation dict
                    new_exp[k] = {sk: sv.cpu().detach() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                else:
                    new_exp[k] = v
            cpu_agent_buffer[seat].append(new_exp)
            
    # Calculate stats (e.g. ROI for the first player)
    final_stack = env.players[0].stack
    initial_stack = random_stacks[0]
    roi = (final_stack - initial_stack) / initial_stack
    
    stats = {
        "roi": roi,
        "num_players": num_players
    }
            
    return env_experiences, cpu_agent_buffer, stats
