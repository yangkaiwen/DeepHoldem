import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete

from gym_env.cycle import PlayerCycle
from gym_env.enums import Action, Stage
from tools.hand_evaluator import get_winner
from tools.helper import flatten

# pylint: disable=import-outside-toplevel

log = logging.getLogger(__name__)

winner_in_episodes = []
MONTEACRLO_RUNS = 1000  # relevant for equity calculation if switched on


class StateHistoryRecorder:
    """Simple state history recorder that maintains a text ledger with transformer embeddings."""

    def __init__(self):
        """
        Initialize the state history recorder.

        Args:
            model_name: Name of the transformer model to use
        """
        self.ledger = ""  # word representation of the game history and decision to make

    def record(self, message):
        """
        Record a message by appending it to the ledger.

        Args:
            message (str): Message to record
        """
        if self.ledger:
            self.ledger += " " + str(message)
        else:
            self.ledger = str(message)

    def reset(self):
        """Reset the ledger (empty it)."""
        self.ledger = ""
        self.encode = []


class StageData:
    """Preflop, flop, turn and river"""

    def __init__(self, num_players):
        """data"""
        self.calls = [False] * num_players  # ix[0] = dealer
        self.raises = [False] * num_players  # ix[0] = dealer
        self.min_call_at_action = [0] * num_players  # ix[0] = dealer
        self.contribution = [0] * num_players  # ix[0] = dealer
        self.stack_at_action = [0] * num_players  # ix[0] = dealer
        self.community_pot_at_action = [0] * num_players  # ix[0] = dealer


class HoldemTable(Env):
    """Pokergame environment"""

    def __init__(
        self,
        initial_stacks=100,
        small_blind=1,
        big_blind=2,
    ):
        """
        The table needs to be initialized once at the beginning

        Args:
            num_of_players (int): number of players that need to be added
            initial_stacks (real): initial stacks per player
            small_blind (real)
            big_blind (real)
            funds_plot (bool): show plot of funds history at end of each episode

        """
        self.num_of_players = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.bb_pos = None  # Track big blind player position
        self.players = []
        self.table_cards = None
        self.dealer_pos = None
        self.player_status = []  # one hot encoded
        self.current_player = None
        self.player_cycle = None  # cycle iterator
        self.stage = None
        self.last_player_pot = None
        self.viewer = None
        self.player_max_win = None  # used for side pots
        self.round_number_in_street = 0
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.played_in_round = None
        self.min_call = None
        self.last_raise_amount = 0  # Track raise amount for minimum raise rule
        self.community_data = None
        self.player_data = None
        self.stage_data = None
        self.deck = None
        self.action = None
        self.winner_ix = None
        self.initial_stacks = initial_stacks
        self.acting_agent = None

        # pots
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = None  # individual player pots

        self.observation = None
        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = None
        self.array_everything = None
        self.legal_moves = None
        self.illegal_move_reward = -10
        self.action_space = Discrete(len(Action) - 2)
        # Define observation space with a placeholder shape
        from gymnasium.spaces import Box

        # We'll use a placeholder shape - actual shape will be determined in reset()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(124,), dtype=np.float32
        )
        self.first_action_for_hand = None

        # Track starting stacks for each hand
        self.starting_stacks = initial_stacks

        # Track previous stack for incremental reward calculation
        self.previous_stacks = None

        # Experience tracking
        self.experience_tuples = (
            []
        )  # List of (player_id, observation, action, reward, done)

        # Terminal rewards and observation
        self.terminal_rewards = None
        self.terminal_observation = None

        # Store each player's observations and actions for the hand
        self.player_observations = {}
        self.player_actions = {}
        self.player_rewards = {}
        self.player_dones = {}

        self.ledger = ""  # word representation of the game history and decision to make
        self.PARSD = (
            []
        )  # a growing list encoding the Player, Action, Reward, Stack, Done

    def reset(self, seed=None, options=None):
        """Reset after game over - now resets to a new hand.

        Args:
            seed: Random seed for reproducibility
            options: Optional dict with:
                - dealer_pos (int): Override dealer position (0 to num_players-1)
        """
        super().reset(seed=seed)

        self.observation = None
        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = pd.DataFrame()
        self.first_action_for_hand = [True] * len(self.players)

        # Reset experience tracking
        self.experience_tuples = []
        self.terminal_rewards = [0] * len(self.players)
        self.terminal_observation = None

        # Reset player-specific tracking
        self.player_observations = {i: [] for i in range(len(self.players))}
        self.player_actions = {i: [] for i in range(len(self.players))}
        self.player_rewards = {i: [] for i in range(len(self.players))}
        self.player_dones = {i: [] for i in range(len(self.players))}

        # Reset state history encoder
        self.ledger = ""
        self.encode = []

        if not self.players:
            log.warning("No agents added. Add agents before resetting the environment.")
            return self.observation, self.info

        # Reset stacks to initial amount for each hand
        for player in self.players:
            player.stack = self.initial_stacks

        # Track starting stacks for this hand
        self.hand_starting_stacks = [player.stack for player in self.players]

        # Initialize previous stacks for incremental reward calculation
        self.previous_stacks = [player.stack for player in self.players]

        # Allow override of dealer position via options
        if options and "dealer_pos" in options:
            self.dealer_pos = options["dealer_pos"]
            self.dealer_pos_override = True
            log.info(f"Dealer position set to seat {self.dealer_pos}")
            # self.state_history_encoder.record(
            #     f"Dealer position set to seat {self.dealer_pos}"
            # )
        else:
            self.dealer_pos = 0
            self.dealer_pos_override = False
        self.player_cycle = PlayerCycle(
            self.players,
            dealer_idx=-1,
        )
        self._start_new_hand()

    def step(self, action):  # pylint: disable=arguments-differ
        """
        Process a single action for the current player.
        Returns observation, reward, done, and info for this step only.
        Does NOT advance to next player or handle game-end logic.

        Args:
            action: Action to take (from agent or external training loop).
        """
        self.reward = 0
        self.acting_agent = self.player_cycle.idx

        # observation that drove to the action
        obs_before_action = self.observation if self.observation is not None else None

        # Validate and execute the action
        if Action(action) not in self.legal_moves:
            self._illegal_move(action)
        else:
            self._execute_step(Action(action))

        # Record the experience tuple
        if self.acting_agent is not None:
            self.experience_tuples.append(
                {
                    "player": self.acting_agent,
                    "observation": obs_before_action,
                    "action": action,
                    "reward": self.reward,
                    "done": self.done,
                }
            )

            # Store in player-specific lists
            self.player_observations[self.acting_agent].append(obs_before_action)
            self.player_actions[self.acting_agent].append(action)
            self.player_rewards[self.acting_agent].append(self.reward)
            self.player_dones[self.acting_agent].append(self.done)

            # Store in PARSD: [Player Seat, Action Index, Reward, All Player Stacks..., Done]
            self.PARSD.extend(
                [
                    self.players[self.acting_agent].seat,
                    action.value if isinstance(action, Action) else action,
                    self.reward,
                    *[int(player.stack) for player in self.players],
                    1 if self.done else 0,
                ]
            )

        return self.observation, self.reward, self.done, False, self.info

    def run(self, action=None):
        """
        Execute game flow iteratively until hand ends.
        Gets action from current player each iteration, steps, then handles game flow.
        Updates self.observation, self.reward, self.done, self.info as game progresses.
        """
        while not self.done:
            # Calculate legal moves before requesting action
            self._get_legal_moves()

            msg = f"legal moves: {[a.name for a in self.legal_moves]}"
            # self.state_history_encoder.record(msg)
            self._get_environment()
            action = self.current_player.agent_obj.action(
                self.legal_moves, self.observation, self.info
            )

            msg = f"Player (seat {self.current_player.seat}) chose action: {action}"
            # self.state_history_encoder.record(msg)
            # self.state_history_encoder.record_triplet(
            #     self.current_player.seat, action.value, self.reward
            # )

            # Process the action
            observation, reward, done, truncated, info = self.step(action)

            # Advance to next player
            self._next_player()

            # Only update environment if there's a valid current player
            # if self.current_player:
            #     self._get_environment()

            # Check if game should end (happens after next_player triggers stage change)
            if self.stage in [
                Stage.END_HIDDEN,
                Stage.SHOWDOWN,
            ]:
                self._end_hand()
                self.done = True
                self._calculate_terminal_rewards()
                self._print_hand_results()
                self._store_terminal_observation()

    def _execute_step(self, action):
        """Process a single decision action. Does NOT handle game flow."""
        self._process_decision(action)

        # Calculate incremental reward for non-terminal states
        if not (self.stage in [Stage.END_HIDDEN, Stage.SHOWDOWN]):
            self._calculate_reward(action)

    def _illegal_move(self, action):
        log.info(
            f"{action} is an Illegal move, try again. Currently allowed: {self.legal_moves}"
        )
        self.reward = self.illegal_move_reward

    def _print_hand_results(self):
        """Display hand results with initial and final stacks"""
        print("\n" + "=" * 50)
        print("HAND ENDED - RESULTS")
        print("=" * 50)
        for seat, player in enumerate(self.players):
            final_stack = player.stack
            initial_stack = self.hand_starting_stacks[seat]
            profit_loss = final_stack - initial_stack
            percentage = (profit_loss / initial_stack * 100) if initial_stack > 0 else 0

            print(
                f"Seat {seat}: Initial: {initial_stack:.2f} → Final: {final_stack:.2f} | P&L: {profit_loss:+.2f} ({percentage:+.1f}%)"
            )
        print("=" * 50)

    def _get_environment(self):
        """Observe the environment"""

        # Print current game state information
        stacks_vector = np.array([player.stack for player in self.players])
        print("\n" + "=" * 60)
        if self.current_player:
            print(f"CURRENT PLAYER: Seat {self.current_player.seat}")
            print(f"Hand: {self.current_player.cards}")
        print(self.current_player)
        print(f"Community Cards: {self.table_cards}")
        print(f"Player Stacks: {stacks_vector}")
        print(f"Callers: {self.callers}")
        print(f"Raisers: {self.raisers}")

        # Compile observation (state) vector
        obs_components = []

        # Player cards - always 2 cards, pad with 0 if needed
        player_cards = [0, 0]
        if (
            self.current_player
            and self.current_player.cards
            and len(self.current_player.cards) == 2
        ):
            encoded = self.encode_card(self.current_player.cards)
            if isinstance(encoded, np.ndarray):
                player_cards = encoded.tolist()
            else:
                player_cards = [encoded, 0]
        obs_components.extend(player_cards)

        # Table cards - always 5 cards, pad with 0s
        community_cards = [0, 0, 0, 0, 0]
        if self.table_cards:
            encoded = self.encode_card(self.table_cards)
            if isinstance(encoded, np.ndarray):
                encoded = encoded.tolist()
            else:
                encoded = [encoded]
            for i, card in enumerate(encoded[:5]):
                community_cards[i] = card
        obs_components.extend(community_cards)

        # Game state features
        obs_components.append(self.stage.value)  # Current stage as integer
        obs_components.append(sum(self.player_cycle.alive))  # Number of active players
        obs_components.append(len(self.callers))  # Number of players called
        obs_components.append(len(self.raisers))  # Number of players raised
        obs_components.append(self.bb_pos)

        # Player-specific features for each player
        for i, player in enumerate(self.players):
            # 1. Is player active (1 if in alive list, 0 otherwise)
            is_active = 1 if self.player_cycle.alive[i] else 0

            # 2. Did player raise in this round (1 if in raisers, 0 otherwise)
            did_raise = 1 if player.seat in self.raisers else 0

            # 3. Amount of money invested in this hand (initial - current)
            money_invested = self.hand_starting_stacks[i] - player.stack

            # 4. Is this the current player (1 if current, 0 otherwise)
            is_current = (
                1
                if self.current_player and self.current_player.seat == player.seat
                else 0
            )

            # 5. Is this player the big blind (1 if BB, 0 otherwise)
            is_bb = 1 if player.seat == self.bb_pos else 0

            obs_components.extend(
                [is_active, did_raise, money_invested, is_current, is_bb]
            )

        # Append PARSD
        obs_components.extend(self.PARSD)

        self.observation = np.array(obs_components, dtype=np.float32)

        # Initialize info dict with basic game state
        self.info = {}

    def _calculate_reward(self, last_action):
        """
        Reward function based on incremental stack change.

        The reward is the change in stack since the last action, allowing
        the agent to learn from immediate consequences of each decision.

        Calculates reward for the current acting agent (self.acting_agent).

        Note: Blinds (SB/BB) don't update previous_stacks, so the first voluntary
        action's reward includes the blind cost.
        """
        # Don't calculate reward for forced blind actions
        if last_action in [Action.SMALL_BLIND, Action.BIG_BLIND]:
            return

        # Calculate reward for the current acting agent
        i = self.acting_agent
        current_stack = self.players[i].stack
        previous_stack = self.previous_stacks[i]
        incremental_change = current_stack - previous_stack

        self.reward = incremental_change

        # Update previous stack for next step
        self.previous_stacks[i] = current_stack

        msg = f"Agent (seat {i}) incremental reward calculation: current_stack={current_stack:.2f}, previous_stack={previous_stack:.2f}, reward={self.reward:.2f}"
        log.info(msg)
        # self.state_history_encoder.record(msg)

    def _calculate_terminal_rewards(self):
        """
        Calculate final rewards for all players when hand ends.

        This ensures all players receive their terminal reward based on
        their final stack change, including pot winnings.

        Updates previous_stacks for all players so they're ready for next hand.
        Sets self.reward for the acting agent's terminal reward.
        """
        for i, player in enumerate(self.players):
            current_stack = player.stack
            previous_stack = self.previous_stacks[i]
            terminal_reward = current_stack - previous_stack

            # Store terminal reward for each player
            self.terminal_rewards[i] = terminal_reward

            # Update previous stack for all players
            self.previous_stacks[i] = current_stack

            log.info(
                f"Player (seat {i}) terminal reward: {terminal_reward} "
                f"(stack: {previous_stack:.2f} -> {current_stack:.2f})"
            )

    def _process_decision(self, action):  # pylint: disable=too-many-statements
        """Process the decisions that have been made by an agent."""
        if action not in [Action.SMALL_BLIND, Action.BIG_BLIND]:
            assert action in set(self.legal_moves), "Illegal decision"

        if action == Action.FOLD:
            self.player_cycle.deactivate_current()
            self.player_cycle.mark_folder()

        else:
            # Calculate current amount to call BEFORE this action
            current_call_needed = max(
                0, self.min_call - self.player_pots[self.current_player.seat]
            )

            if action == Action.CALL:
                contribution = min(current_call_needed, self.current_player.stack)
                self.callers.append(self.current_player.seat)
                self.last_caller = self.current_player.seat
                # Update player cycle contribution
                self.player_cycle.update_contribution(
                    self.current_player.seat, contribution
                )

            elif action == Action.CHECK:
                if current_call_needed > 0:
                    raise ValueError("Cannot check when there's a bet to call")
                contribution = 0
                self.player_cycle.mark_checker()

            elif action == Action.BET_1_4_POT:
                # Calculate pot-sized bet: total bet = call amount + pot after call
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 0.25)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_1_3_POT:
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 0.33)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_1_2_POT:
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 0.50)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_2_3_POT:
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 0.66)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_3_4_POT:
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 0.75)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_POT:
                # Pot-sized bet: total bet = call amount + pot after call
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + pot_after_call
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_3_2_POT:
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 1.50)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_2_POT:
                call_amount = current_call_needed
                pot_after_call = (
                    self.community_pot + self.current_round_pot + call_amount
                )
                total_bet = call_amount + (pot_after_call * 2.00)
                contribution = min(total_bet, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.BET_MIN_RAISE:
                # Minimum raise: match current bet + raise by at least last raise amount
                # or big blind if no previous raise
                min_raise_amount = max(self.last_raise_amount, self.big_blind)
                total_bet = self.min_call + min_raise_amount
                player_total_needed = (
                    total_bet - self.player_pots[self.current_player.seat]
                )
                contribution = min(player_total_needed, self.current_player.stack)
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.ALL_IN:
                contribution = self.current_player.stack
                self.raisers.append(self.current_player.seat)
                self.current_player.num_raises_in_street[self.stage] += 1
                # Mark as raiser in player cycle
                self.player_cycle.mark_raiser(contribution)

            elif action == Action.SMALL_BLIND:
                contribution = np.minimum(self.small_blind, self.current_player.stack)
                # Update player cycle contribution
                self.player_cycle.update_contribution(
                    self.current_player.seat, contribution
                )

            elif action == Action.BIG_BLIND:
                contribution = np.minimum(self.big_blind, self.current_player.stack)
                # Mark BB as default aggressor in player cycle
                self.player_cycle.mark_bb(contribution)
            else:
                raise RuntimeError("Illegal action.")

            # Update player's contribution
            self.current_player.stack -= contribution
            self.player_pots[self.current_player.seat] += contribution
            self.current_round_pot += contribution
            self.last_player_pot = self.player_pots[self.current_player.seat]

            # Calculate new total for this player
            new_player_total = self.player_pots[self.current_player.seat]

            # Update min_call (the maximum total contribution any player has made)
            if new_player_total > self.min_call:
                # If this is a raise (contribution > current_call_needed), update last_raise_amount
                if contribution > current_call_needed:
                    self.last_raise_amount = new_player_total - self.min_call
                self.min_call = new_player_total

            if self.current_player.stack == 0 and contribution > 0:
                self.player_cycle.mark_out_of_cash_but_contributed()

            self.current_player.actions.append(action)
            self.current_player.last_action_in_stage = action.name
            self.current_player.temp_stack.append(self.current_player.stack)

            self.player_max_win[self.current_player.seat] += contribution  # side pot

            pos = self.player_cycle.idx
            rnd = self.stage.value + self.player_cycle.round_number_in_street
            self.stage_data[rnd].calls[pos] = action == Action.CALL
            self.stage_data[rnd].raises[pos] = action in [
                Action.BET_2_POT,
                Action.BET_1_2_POT,
                Action.BET_POT,
                Action.BET_1_3_POT,
                Action.BET_1_4_POT,
                Action.BET_2_3_POT,
                Action.BET_3_4_POT,
                Action.BET_3_2_POT,
                Action.BET_MIN_RAISE,
                Action.ALL_IN,
            ]
            self.stage_data[rnd].min_call_at_action[pos] = self.min_call / (
                self.big_blind * 100
            )
            self.stage_data[rnd].community_pot_at_action[pos] = self.community_pot / (
                self.big_blind * 100
            )
            self.stage_data[rnd].contribution[pos] += contribution / (
                self.big_blind * 100
            )
            self.stage_data[rnd].stack_at_action[pos] = self.current_player.stack / (
                self.big_blind * 100
            )

        self.player_cycle.update_alive()

        msg = (
            f"Seat {self.current_player.seat} ({self.current_player.name}): {action} - "
            f"Remaining stack: {self.current_player.stack}, "
            f"Round pot: {self.current_round_pot}, Community pot: {self.community_pot}, "
            f"player pot: {self.player_pots[self.current_player.seat]}, min_call: {self.min_call}, "
            f"last_raise_amount: {self.last_raise_amount}"
        )
        log.info(msg)
        # self.state_history_encoder.record(msg)

    def _start_new_hand(self):
        """Deal new cards to players and reset table states."""
        log.info("++++++++++++++++++")
        msg = "Starting new hand."
        # self.state_history_encoder.record(msg)
        log.info(msg)
        log.info("++++++++++++++++++")
        self.table_cards = []
        self._create_card_deck()
        self.stage = Stage.PREFLOP

        # preflop round1,2, flop>: round 1,2, turn etc...
        self.stage_data = [StageData(len(self.players)) for _ in range(8)]

        # pots
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)
        self.player_max_win = [0] * len(self.players)
        self.last_player_pot = 0
        self.played_in_round = 0
        self.first_action_for_hand = [True] * len(self.players)

        for player in self.players:
            player.cards = []
            player.num_raises_in_street = {
                Stage.PREFLOP: 0,
                Stage.FLOP: 0,
                Stage.TURN: 0,
                Stage.RIVER: 0,
            }

        self._next_dealer()
        self._distribute_cards()
        self._initiate_round()

    def _check_game_over(self):
        """Check if hand is over - simplified for single hand episodes"""
        # In single hand mode, game is only over when hand ends (in _execute_step)
        # We don't check for player elimination since stacks reset each hand
        return False

    def _game_over(self):
        """End of an episode."""
        log.info("Hand over.")
        self.done = True
        log.info(f"Hand completed. Winner: Player {self.winner_ix}")

    def _initiate_round(self):
        """A new round (flop, turn, river) is initiated"""
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.min_call = 0
        self.last_raise_amount = self.big_blind  # Minimum raise is at least big blind
        for player in self.players:
            player.last_action_in_stage = ""
            # Reset raises counter for new street
            player.num_raises_in_street[self.stage] = 0

        self.player_cycle.new_street_reset()

        # advance headsup players by 1 step after preflop
        if self.stage != Stage.PREFLOP and self.num_of_players == 2:
            self.player_cycle.idx += 1

        if self.stage == Stage.PREFLOP:
            log.info("")
            log.info("===Round: Stage: PREFLOP")
            # self.state_history_encoder.record("===Round: Stage: PREFLOP")

            self._next_player()
            # Skip players with $0 stack when posting blinds
            while self.current_player.stack == 0:
                log.warning(
                    f"Player {self.current_player.seat} has $0 stack, skipping small blind"
                )
                self._next_player()
            self._process_decision(Action.SMALL_BLIND)

            self._next_player()
            # Skip players with $0 stack when posting blinds
            while self.current_player.stack == 0:
                log.warning(
                    f"Player {self.current_player.seat} has $0 stack, skipping big blind"
                )
                self._next_player()
            # Set bb_pos to current player's seat
            self.bb_pos = self.current_player.seat
            self._process_decision(Action.BIG_BLIND)

            self._next_player()
            # Skip players with $0 stack for first action
            while self.current_player.stack == 0:
                log.warning(
                    f"Player {self.current_player.seat} has $0 stack, skipping first action"
                )
                self._next_player()

        elif self.stage in [Stage.FLOP, Stage.TURN, Stage.RIVER]:
            self._next_player()

        elif self.stage == Stage.SHOWDOWN:
            log.info("Showdown")

        else:
            raise RuntimeError()

    def add_player(self, agent):
        """Add a player to the table. Has to happen at the very beginning"""
        self.num_of_players += 1
        player = PlayerShell(stack_size=self.initial_stacks, name=agent.name)
        player.agent_obj = agent
        player.seat = len(self.players)  # assign next seat number to player
        player.stack = self.initial_stacks
        self.players.append(player)
        self.player_status = [True] * len(self.players)
        self.player_pots = [0] * len(self.players)

    def _is_betting_round_complete(self):
        """Check if the current betting round is complete."""
        # Let the PlayerCycle handle the unified rule logic
        return self.player_cycle._should_end_betting_round()

    def _find_next_active_player(self, start_idx):
        """Find the next active player starting from start_idx."""
        num_players = len(self.players)

        for offset in range(1, num_players + 1):
            idx = (start_idx + offset) % num_players
            if (
                self.player_cycle.can_still_make_moves_in_this_hand[idx]
                or self.player_cycle.out_of_cash_but_contributed[idx]
            ):
                return idx

        return -1  # No active players found

    def _end_round(self):
        """End of preflop, flop, turn or river"""
        # First check if betting round is actually complete
        if not self._is_betting_round_complete():
            log.warning("Attempting to end round when betting is not complete!")
            # Try to find next player to act
            next_idx = self._find_next_active_player(self.player_cycle.idx)
            if next_idx >= 0:
                self.player_cycle.idx = next_idx
                self.current_player = self.players[next_idx]
                return  # Don't end round

        # Close the current round and move pots
        self._close_round()

        # Move to next street
        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self._distribute_cards_to_table(3)
            log.info("--- FLOP ---")
        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self._distribute_cards_to_table(1)
            log.info("--- TURN ---")
        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self._distribute_cards_to_table(1)
            log.info("--- RIVER ---")
        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN
            log.info("--- SHOWDOWN ---")
        msg = f"===ROUND: {self.stage} ==="
        log.info(msg)
        # self.state_history_encoder.record(msg)
        self._clean_up_pots()

    def _clean_up_pots(self):
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)

    def _end_hand(self):
        self._clean_up_pots()
        self.winner_ix = self._get_winner()
        self._award_winner(self.winner_ix)
        # Handled in run(): self.done = True

    def _store_terminal_observation(self):
        """Store terminal observation (same for all players)"""
        # Get observation from environment's current state
        self._get_environment()  # This updates self.observation
        self.terminal_observation = self.observation

        # Also record this as the last observation for all players who have any observations
        for player_id in range(len(self.players)):
            if self.player_observations[player_id]:
                # Append terminal observation as the last observation for each player
                self.player_observations[player_id].append(self.terminal_observation)

    def get_player_experiences(self):
        """
        Process experiences to create (state, action, reward, next_state, done) for each player.

        For each player:
        - state: observation at time of action
        - action: action taken
        - reward: incremental reward (terminal reward added to last reward)
        - next_state: next observation from same player's list, or terminal observation if last
        - done: True for last experience of each player

        Returns:
            dict: Dictionary mapping player_id to list of experiences
        """
        if not self.done:
            log.warning("Cannot get experiences before hand is done!")
            return {}

        processed_experiences = {}

        for player_id in range(len(self.players)):
            # Get this player's observations, actions, rewards, and dones
            observations = self.player_observations[player_id]
            actions = self.player_actions[player_id]
            rewards = self.player_rewards[player_id]
            dones = self.player_dones[player_id]

            # Skip players with no experiences
            if not observations or not actions:
                processed_experiences[player_id] = []
                continue

            # Add terminal reward to the last reward for this player
            if player_id < len(self.terminal_rewards):
                terminal_reward = self.terminal_rewards[player_id]
                if rewards:  # Add terminal reward to last incremental reward
                    rewards[-1] += terminal_reward

            # Create experiences for this player
            player_exps = []

            for i in range(len(actions)):
                # Get current state
                state = observations[i]

                # Get action
                action = actions[i]

                # Get reward
                reward = rewards[i]

                # Determine next_state:
                # If there's another observation from this player, use it
                # Otherwise, use terminal observation
                if i + 1 < len(observations):
                    next_state = observations[i + 1]
                else:
                    next_state = self.terminal_observation

                # Determine done:
                # True only for the last experience of this player
                done = i == len(actions) - 1

                player_exps.append(
                    {
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                    }
                )

            processed_experiences[player_id] = player_exps

        return processed_experiences

    def get_all_experience_tuples(self):
        """
        Get all raw experience tuples in order they occurred.

        Returns:
            list: List of experience tuples (player_id, observation, action, reward, done)
        """
        return self.experience_tuples

    def get_terminal_rewards(self):
        """
        Get terminal rewards vector for all players.

        Returns:
            list: Terminal rewards for each player
        """
        return self.terminal_rewards

    def get_terminal_observation(self):
        """
        Get terminal observation (same for all players).

        Returns:
            np.array: Terminal observation
        """
        return self.terminal_observation

    def _get_winner(self):
        """Determine which player has won the hand"""
        potential_winners = self.player_cycle.get_potential_winners()

        potential_winner_idx = [
            i
            for i, potential_winner in enumerate(potential_winners)
            if potential_winner
        ]
        if sum(potential_winners) == 1:
            winner_ix = [i for i, active in enumerate(potential_winners) if active][0]
            winning_card_type = "Only remaining player in round"

        else:
            assert self.stage == Stage.SHOWDOWN
            remaining_player_winner_ix, winning_card_type = get_winner(
                [
                    player.cards
                    for ix, player in enumerate(self.players)
                    if potential_winners[ix]
                ],
                self.table_cards,
            )
            winner_ix = potential_winner_idx[remaining_player_winner_ix]
        log.info(f"Player {winner_ix} won: {winning_card_type}")
        return winner_ix

    def _award_winner(self, winner_ix):
        """Hand the pot to the winner and handle side pots"""
        max_win_per_player_for_winner = self.player_max_win[winner_ix]
        total_winnings = sum(
            np.minimum(max_win_per_player_for_winner, self.player_max_win)
        )
        remains = np.maximum(
            0, np.array(self.player_max_win) - max_win_per_player_for_winner
        )  # to be returned

        self.players[winner_ix].stack += total_winnings
        self.winner_ix = winner_ix
        if total_winnings < sum(self.player_max_win):
            log.info("Returning side pots")
            # self.state_history_encoder.record("Returning side pots")
            for i, player in enumerate(self.players):
                player.stack += remains[i]

    def _next_dealer(self):
        """Move to next dealer, or use override if set on first hand."""
        if hasattr(self, "dealer_pos_override") and self.dealer_pos_override:
            # Use the override position on first hand only
            # Set player_cycle's dealer_idx to point to the correct player by seat
            for idx, player in enumerate(self.players):
                if player.seat == self.dealer_pos:
                    self.player_cycle.dealer_idx = idx - 1
                    break
            self.dealer_pos_override = False  # Only use override once
            log.info(f"Using overridden dealer position: seat {self.dealer_pos}")
        else:
            self.dealer_pos = self.player_cycle.next_dealer().seat

    def _next_player(self):
        """Move to the next player"""
        # First check if all non-folded players are all-in
        non_folded_indices = self.player_cycle.get_non_folded_players()
        if non_folded_indices:
            all_all_in = all(
                self.player_cycle.out_of_cash_but_contributed[i]
                for i in non_folded_indices
            )
            if all_all_in and len(non_folded_indices) >= 2:
                log.info(
                    "All non-folded players are all-in. Dealing remaining community cards and going to showdown."
                )
                # Skip to showdown by ending current round and subsequent rounds
                while self.stage != Stage.SHOWDOWN:
                    self._end_round()
                return

        self.current_player = self.player_cycle.next_player()

        if not self.current_player:
            # Player cycle indicates round should end based on unified rules
            if sum(self.player_cycle.alive) < 2:
                log.info("Only one player remaining in round")
                self.stage = Stage.END_HIDDEN
            else:
                msg = "Betting round complete."
                log.info(msg)
                # self.state_history_encoder.record(msg)
                self._end_round()
                # Only initiate new round if we're not at showdown
                if self.stage != Stage.SHOWDOWN:
                    self._initiate_round()

    def _get_legal_moves(self):
        """Determine what moves are allowed in the current state"""
        self.legal_moves = []

        # Check if player can check (has matched the current bet)
        if self.player_pots[self.current_player.seat] >= self.min_call:
            self.legal_moves.append(Action.CHECK)
        else:
            self.legal_moves.append(Action.CALL)
            self.legal_moves.append(Action.FOLD)

        # Calculate raise options
        player_current_contribution = self.player_pots[self.current_player.seat]
        call_amount_needed = max(0, self.min_call - player_current_contribution)

        # Minimum raise amount is the last raise amount or big blind, whichever is larger
        min_raise_increment = max(self.last_raise_amount, self.big_blind)
        min_total_for_raise = self.min_call + min_raise_increment

        # Player needs to have at least call_amount_needed + min_raise_increment to raise
        min_raise_contribution = min_total_for_raise - player_current_contribution

        # Check if player can make a minimum raise
        if (
            self.current_player.stack >= min_raise_contribution
            and min_raise_contribution > call_amount_needed
        ):
            self.legal_moves.append(Action.BET_MIN_RAISE)

        # Check other raise sizes (pot-sized bets)
        # For each bet size, calculate the total bet and check if player can afford it
        bet_sizes = [
            (Action.BET_1_4_POT, 0.25),
            (Action.BET_1_3_POT, 0.33),
            (Action.BET_1_2_POT, 0.50),
            (Action.BET_2_3_POT, 0.66),
            (Action.BET_3_4_POT, 0.75),
            (Action.BET_POT, 1.00),
            (Action.BET_3_2_POT, 1.50),
            (Action.BET_2_POT, 2.00),
        ]

        for action_type, multiplier in bet_sizes:
            # Calculate total bet for this action
            pot_after_call = (
                self.community_pot + self.current_round_pot + call_amount_needed
            )
            total_bet = call_amount_needed + (pot_after_call * multiplier)
            contribution_needed = total_bet - player_current_contribution

            # Check if player can afford it and it's at least a minimum raise
            if (
                self.current_player.stack >= contribution_needed
                and contribution_needed > call_amount_needed
                and (total_bet - self.min_call) >= min_raise_increment
            ):
                self.legal_moves.append(action_type)

        # ALL_IN: Always legal if player has stack
        if self.current_player.stack > 0:
            self.legal_moves.append(Action.ALL_IN)

        log.debug(
            f"Legal moves for seat {self.current_player.seat}: {self.legal_moves}"
        )

    def _close_round(self):
        """put player_pots into community pots"""
        # Log the state before closing
        msg = (
            f"Closing round. Community pot: {self.community_pot}, "
            f"Current round pot: {self.current_round_pot}, "
            f"Min call: {self.min_call}"
        )
        log.info(msg)
        # self.state_history_encoder.record(msg)

        for i, pot in enumerate(self.player_pots):
            if pot > 0:
                log.debug(f"Player {i} contributed {pot} this round")

        self.community_pot += sum(self.player_pots)
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)
        self.min_call = 0  # Reset for next round
        self.last_raise_amount = self.big_blind  # Reset to big blind
        self.played_in_round = 0

    def _create_card_deck(self):
        values = "23456789TJQKA"
        suites = "CDHS"
        self.deck = []  # contains cards in the deck
        _ = [self.deck.append(x + y) for x in values for y in suites]

    def encode_card(self, card):
        """
        Encode a poker card string or vector of card strings to number(s).

        Args:
            card (str or list/array): Card string (e.g., '2D', 'AS', 'KH')
                                     or list/array of card strings
                       First character is value (2-9, T, J, Q, K, A)
                       Second character is suit (C, D, H, S)

        Returns:
            int or np.array: Encoded card number (0-51) for single card,
                            or array of encoded numbers for multiple cards
                            Encoding: value_index * 4 + suit_index
        """
        values = "23456789TJQKA"
        suits = "CDHS"

        # Handle single card string
        if isinstance(card, str):
            if len(card) != 2:
                raise ValueError(
                    f"Invalid card format: {card}. Expected format: '2D', 'AS', etc."
                )

            value_char = card[0]
            suit_char = card[1]

            if value_char not in values or suit_char not in suits:
                raise ValueError(
                    f"Invalid card: {card}. Value must be in {values}, suit must be in {suits}"
                )

            value_index = values.index(value_char)
            suit_index = suits.index(suit_char)

            return value_index * 4 + suit_index

        # Handle list/array of card strings
        else:
            encoded_cards = []
            for c in card:
                if len(c) != 2:
                    raise ValueError(
                        f"Invalid card format: {c}. Expected format: '2D', 'AS', etc."
                    )

                value_char = c[0]
                suit_char = c[1]

                if value_char not in values or suit_char not in suits:
                    raise ValueError(
                        f"Invalid card: {c}. Value must be in {values}, suit must be in {suits}"
                    )

                value_index = values.index(value_char)
                suit_index = suits.index(suit_char)
                encoded_cards.append(value_index * 4 + suit_index)

            return np.array(encoded_cards)

    def _distribute_cards(self):
        log.info(f"Dealer is at position {self.dealer_pos}")
        for player in self.players:
            player.cards = []
            if player.stack <= 0:
                continue  # Skip players with $0 stack
            for _ in range(2):
                card = np.random.randint(0, len(self.deck))
                player.cards.append(self.deck.pop(card))
            log.info(f"Player {player.seat} got {player.cards} and ${player.stack}")

    def _distribute_cards_to_table(self, amount_of_cards):
        for _ in range(amount_of_cards):
            card = np.random.randint(0, len(self.deck))
            self.table_cards.append(self.deck.pop(card))
        msg = f"Cards on table: {self.table_cards}"
        # self.state_history_encoder.record(msg)
        log.info(msg)


class PlayerShell:
    """Player shell"""

    def __init__(self, stack_size, name):
        """Initiaization of an agent"""
        self.stack = stack_size
        self.seat = None
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ""
        self.temp_stack = []
        self.name = name
        self.agent_obj = None
        self.cards = None
        self.num_raises_in_street = {
            Stage.PREFLOP: 0,
            Stage.FLOP: 0,
            Stage.TURN: 0,
            Stage.RIVER: 0,
        }

    def __repr__(self):
        return f"Player {self.name} at seat {self.seat} with stack of {self.stack} and cards {self.cards}"
