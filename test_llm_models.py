import os
import time
import re
from dotenv import load_dotenv
from agents.llm_agent import LLMAgent

# Load environment variables
load_dotenv()

models = [
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.1-405b-instruct:free",
    "xiaomi/mimo-v2-flash:free",
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "qwen/qwen-2.5-vl-7b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/devstral-2512:free",
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
]

failed_models = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "allenai/olmo-3-32b-think:free",
    "qwen/qwen3-4b:free",
    "z-ai/glm-4.5-air:free",
]

models = [m for m in models if m not in failed_models]

prompt = """Your are a Texas Hold'em playing agent. Based on the game history provided, choose your next action from the legal moves listed below by responding with ONLY the index of your chosen action.

[IMPORTANT]You are seated at position: 0
Your current cards are: ['2C', '5H']
Community cards: []
Dealer(Button) position: 0
Total number of players at the table: 3
INITIAL player stacks: [2042, 1194, 3836]

Card notation: Suits are C=Clubs, D=Diamonds, H=Hearts, S=Spades. T stands for 10 (e.g., 'TH' = 10 of Hearts).
Player seat starts from 0 to n-1 note that 0 is not necessarily the dealer. The play direction is from 0->1->2->...n-1->0 in clockwise direction just like a poker game.
The actions taken so far are recorded in the history below.


GAME STARTS!
 ===Round: Stage: PREFLOP===

Seat 1 (Keypress): Action.SMALL_BLIND - Remaining stack: 1193
Round pot: 1, Community pot: 0,player pot: 1
min_call: 1, last_raise_amount: 1

Seat 2 (Keypress): Action.BIG_BLIND - Remaining stack: 3834
Round pot: 3, Community pot: 0,player pot: 2
min_call: 2, last_raise_amount: 1

It is your turn to make a decision.Here are the legal moves:
2: CALL
0: FOLD
11: BET_MIN_RAISE
5: BET_1_2_POT
6: BET_2_3_POT
7: BET_3_4_POT
8: BET_POT
9: BET_3_2_POT
10: BET_2_POT
12: ALL_IN
Choose the index of your available legal action and respond with ONLY the index. (Example response: 2)"""

print(f"Testing {len(models)} models...\n")

# Initialize agent (model param here doesn't matter as we override it in _call_llm)
agent = LLMAgent(name="Tester")

for model in models:
    print(f"--- Testing {model} ---")
    try:
        start_time = time.time()
        # We use _call_llm directly to test the raw response
        # Note: _call_llm was modified to accept model param
        response = agent._call_llm(prompt, model=model)
        elapsed = time.time() - start_time

        print(f"Time: {elapsed:.2f}s")
        print(f"Response: {response}")

        # Check if response contains a valid integer
        match = re.search(r"\d+", str(response))
        if match:
            print(f"Parsed Action: {match.group()}")
        else:
            print("Parsed Action: None (Failed to parse integer)")

    except Exception as e:
        print(f"ERROR: {e}")

    print("\n")
