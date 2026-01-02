import numpy as np
import logging
import torch
import sys
import csv
from datetime import datetime
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
        import concurrent.futures
        from parallel_worker import run_evaluation_episode

        # Determine main agent key
        if self.model_path == "keypress":
            main_agent_key = "keypress_agent"
        elif self.model_path == "random":
            main_agent_key = "random_agent_main"
        else:
            main_agent_key = "ac_agent"

        # Initialize history for metrics
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

        # Prepare weights if using AC agent
        actor_state = None
        critic_state = None
        if self.ac_agent:
            actor_state = {k: v.cpu() for k, v in self.ac_agent.actor.state_dict().items()}
            critic_state = {k: v.cpu() for k, v in self.ac_agent.critic.state_dict().items()}

        # Parallel Execution Setup
        num_workers = 20
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
        futures = []

        self.log.info(f"Starting evaluation with {num_workers} parallel workers...")

        # Submit all tasks
        for _ in range(self.num_episodes):
            futures.append(
                executor.submit(
                    run_evaluation_episode,
                    actor_state,
                    critic_state,
                    self.model_path,
                    self.num_players,
                    self.initial_stack,
                    self.opponent_probabilities
                )
            )

        completed_episodes = 0
        
        try:
            for future in concurrent.futures.as_completed(futures):
                completed_episodes += 1
                try:
                    results = future.result()
                    
                    # Process results
                    for res in results:
                        a_type = res["agent_type"]
                        initial = res["initial"]
                        final = res["final"]
                        investment = res["investment"]
                        profit = final - initial

                        # 1. ROI for this specific hand
                        roi = (profit / investment) if investment > 0 else 0.0
                        agent_history[a_type]["rois"].append(roi)

                        # 2. Profit in Big Blinds
                        # Assuming BB is 2 for now as it's standard in this env, 
                        # or we could pass it back from worker if needed.
                        bb = 2 
                        profit_bb = profit / bb
                        agent_history[a_type]["profits_bb"].append(profit_bb)

                        # 3. Accumulate totals
                        agent_history[a_type]["total_profit"] += profit
                        agent_history[a_type]["total_invest"] += investment

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

                    # Log progress
                    if completed_episodes % 10 == 0 or completed_episodes == self.num_episodes:
                        print(
                            f"Episode {completed_episodes}/{self.num_episodes} | "
                            f"Mean ROI: {current_mean_rois} | "
                            f"BB/100: {current_bb_100s} | "
                            f"Cum NROI: {current_cum_nrois}",
                            end="\r",
                        )

                except Exception as e:
                    self.log.error(f"Evaluation worker failed: {e}")
                    import traceback
                    traceback.print_exc()

        except KeyboardInterrupt:
            self.log.info("Evaluation interrupted. Shutting down workers...")
            executor.shutdown(wait=False)
            raise

        executor.shutdown(wait=True)
        print() # Newline after progress bar



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
