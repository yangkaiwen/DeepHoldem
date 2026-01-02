import numpy as np
import logging
import torch
import sys
from gym_env.env import HoldemTable
from agents.ac_agent import PokerACAgent
from agents.random_agent import RandomAgent
from agents.llm_agent import LLMAgent
from agents.keypress_agent import KeypressAgent


class GameEvaluator:
    def __init__(
        self,
        model_path,
        num_episodes,
        num_players=None,
        initial_stack=None,
        opponent_probabilities=None,
        log_results_path=None,
    ):
        self.model_path = model_path
        self.num_episodes = int(num_episodes)
        self.num_players = int(num_players) if num_players else None
        self.initial_stack = int(initial_stack) if initial_stack else None
        self.opponent_probabilities = opponent_probabilities or [
            0.5,
            0.2,
            0.3,
        ]  # Random, Mini, Large
        self.log_results_path = log_results_path
        self.log = logging.getLogger(__name__)

        # Initialize AC agent if needed
        if model_path != "keypress" and model_path != "random":
            self.ac_agent = PokerACAgent(name="ACAgent", device="auto")
            # Load model if path is provided and exists
            if model_path:
                try:
                    self.ac_agent.load(model_path)
                    self.ac_agent.eval_mode()
                except Exception as e:
                    print(f"Could not load model from {model_path}: {e}")
        else:
            self.ac_agent = None

    def run(self):
        # Determine main agent key
        if self.model_path == "keypress":
            main_agent_key = "keypress_agent"
        elif self.model_path == "random":
            main_agent_key = "random_agent_main"
        else:
            main_agent_key = "ac_agent"

        # Initialize history for metrics
        # We track lists to calculate Mean ROI and BB/100, which are robust to stack size variance
        agent_history = {
            key: {
                "rois": [],
                "profits_bb": [],
                "total_profit": 0.0,
                "total_invest": 0.0,
            }
            for key in [
                main_agent_key,
                "random_agent",
                "llm_agent_mini",
                "llm_agent_large",
            ]
        }

        results = []  # List of [ROI_Main, ROI_Random, ROI_Mini, ROI_Large] per episode

        for episode in range(self.num_episodes):
            # Determine num_players
            n_players = (
                self.num_players if self.num_players else np.random.randint(2, 10)
            )

            # Determine initial stack
            stacks = []
            if self.initial_stack:
                stacks = [int(self.initial_stack)] * n_players
            else:
                # Random between 400 and 4000
                stacks = np.random.randint(400, 4001, size=n_players).tolist()

            # Create Env
            env = HoldemTable(initial_stacks=stacks[0])

            # Add Agents
            # Seat 0: AC Agent or Keypress
            agent_types = []  # Track type for ROI mapping

            if self.model_path == "keypress":
                env.add_player(KeypressAgent(name="Keypress"))
                agent_types.append(main_agent_key)
            elif self.model_path == "random":
                env.add_player(RandomAgent(name="Random_Main"))
                agent_types.append(main_agent_key)
            else:
                env.add_player(self.ac_agent)
                agent_types.append(main_agent_key)

            # Add Opponents
            # Probabilities: [Random, Mini, Large]
            for i in range(1, n_players):
                rand = np.random.random()
                p_random, p_mini, p_large = self.opponent_probabilities

                # Normalize probabilities to sum to 1 if they don't
                total_p = sum(self.opponent_probabilities)
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
                        # LLMAgent(name=f"Mini_{i}", model="openai/gpt-4.1-mini")
                        LLMAgent(name=f"Mini_{i}", model="xiaomi/mimo-v2-flash:free")
                    )
                    agent_types.append("llm_agent_mini")
                else:
                    # Large
                    env.add_player(
                        # LLMAgent(name=f"Large_{i}", model="openai/gpt-4.1-mini")
                        LLMAgent(
                            name=f"Large_{i}",
                            model="meta-llama/llama-3.3-70b-instruct:free",
                        )
                    )
                    agent_types.append("llm_agent_large")

            # Run Episode
            # dealer_pos random
            dealer_pos = np.random.randint(0, n_players)
            env.reset(options={"stacks": stacks, "dealer_pos": dealer_pos})
            env.run()

            # Calculate stats and update history
            for i, player in enumerate(env.players):
                initial = env.hand_starting_stacks[i]
                final = player.stack
                investment = env.player_max_win[i]
                profit = final - initial

                a_type = agent_types[i]

                # 1. ROI for this specific hand
                roi = (profit / investment) if investment > 0 else 0.0
                agent_history[a_type]["rois"].append(roi)

                # 2. Profit in Big Blinds
                # env.big_blind is usually 2, but we access it dynamically
                bb = env.big_blind
                profit_bb = profit / bb
                agent_history[a_type]["profits_bb"].append(profit_bb)

                # 3. Accumulate totals
                agent_history[a_type]["total_profit"] += profit
                agent_history[a_type]["total_invest"] += investment

                # Debug print for main agent
                if a_type == main_agent_key:
                    won = "YES" if profit > 0 else "NO"
                    pct_stack = (profit / initial) * 100
                    other_players = n_players - 1
                    # print(
                    #     f"Episode {episode+1}: Won={won}, Stack%={pct_stack:.2f}%, Opponents={other_players}, Invest={investment}, Profit={profit}, ROI={roi:.4f}"
                    # )

            # Calculate current Mean ROI for display
            current_mean_rois = []
            current_bb_100s = []
            current_cum_nrois = []

            for key in [
                main_agent_key,
                "random_agent",
                "llm_agent_mini",
                "llm_agent_large",
            ]:
                rois = agent_history[key]["rois"]
                profits_bb = agent_history[key]["profits_bb"]
                total_profit = agent_history[key]["total_profit"]
                total_invest = agent_history[key]["total_invest"]

                mean_roi = np.mean(rois) if rois else 0.0
                bb_100 = (np.mean(profits_bb) * 100) if profits_bb else 0.0
                cum_nroi = (total_profit / total_invest) if total_invest > 0 else 0.0

                current_mean_rois.append(round(mean_roi, 4))
                current_bb_100s.append(round(bb_100, 2))
                current_cum_nrois.append(round(cum_nroi, 4))

            results.append(current_mean_rois)

            # Progress Bar
            progress = (episode + 1) / self.num_episodes
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)

            main_roi = current_mean_rois[0]

            sys.stdout.write(
                f"\r[{bar}] {progress*100:.1f}% | Ep: {episode+1}/{self.num_episodes} | Main ROI: {main_roi:.4f}"
            )
            sys.stdout.flush()

            # print(
            #     f"Episode {episode+1}/{self.num_episodes} Mean NROI: {current_mean_rois} | BB/100: {current_bb_100s} | Cum NROI: {current_cum_nrois}"
            # )

        # Final Metrics Calculation
        final_metrics = {}
        for key in [
            main_agent_key,
            "random_agent",
            "llm_agent_mini",
            "llm_agent_large",
        ]:
            rois = agent_history[key]["rois"]
            profits_bb = agent_history[key]["profits_bb"]
            total_profit = agent_history[key]["total_profit"]
            total_invest = agent_history[key]["total_invest"]

            if rois:
                mean_roi = np.mean(rois)
                std_roi = np.std(rois)
                # BB/100 = Average BB profit per hand * 100
                bb_100 = np.mean(profits_bb) * 100

                # Cumulative NROI
                cum_nroi = (total_profit / total_invest) if total_invest > 0 else 0.0

                final_metrics[key] = (
                    f"Mean ROI: {mean_roi:.4f} (±{std_roi:.4f}) | BB/100: {bb_100:.2f} | Cum NROI: {cum_nroi:.4f}"
                )
            else:
                final_metrics[key] = "N/A"

        print("\n" + "=" * 80)
        print(f"Final Performance over {self.num_episodes} episodes:")
        print(f"Main Agent ({main_agent_key}): {final_metrics[main_agent_key]}")
        print(f"Random: {final_metrics['random_agent']}")
        print(f"Mini LLM: {final_metrics['llm_agent_mini']}")
        print(f"Large LLM: {final_metrics['llm_agent_large']}")
        print("=" * 80 + "\n")

        if self.log_results_path:
            import csv
            import os
            from datetime import datetime

            file_exists = os.path.isfile(self.log_results_path)
            with open(self.log_results_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        [
                            "Timestamp",
                            "Model",
                            "Episodes",
                            "Opponent_Probs",
                            "Main_Mean_ROI",
                            "Main_Std_ROI",
                            "Main_BB_100",
                            "Main_Cum_NROI",
                        ]
                    )

                # Extract main agent metrics
                rois = agent_history[main_agent_key]["rois"]
                profits_bb = agent_history[main_agent_key]["profits_bb"]
                total_profit = agent_history[main_agent_key]["total_profit"]
                total_invest = agent_history[main_agent_key]["total_invest"]

                if rois:
                    mean_roi = np.mean(rois)
                    std_roi = np.std(rois)
                    bb_100 = np.mean(profits_bb) * 100
                    cum_nroi = (
                        (total_profit / total_invest) if total_invest > 0 else 0.0
                    )
                else:
                    mean_roi = std_roi = bb_100 = cum_nroi = 0.0

                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.model_path,
                        self.num_episodes,
                        str(self.opponent_probabilities),
                        f"{mean_roi:.4f}",
                        f"{std_roi:.4f}",
                        f"{bb_100:.2f}",
                        f"{cum_nroi:.4f}",
                    ]
                )
            self.log.info(f"Results saved to {self.log_results_path}")

        return final_metrics
