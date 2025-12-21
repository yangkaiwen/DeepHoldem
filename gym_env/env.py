"""Groupier functions"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete

from gym_env.cycle import PlayerCycle
from gym_env.enums import Action, Stage
from gym_env.rendering import PygletWindow, WHITE, RED, GREEN, BLUE
from tools.hand_evaluator import get_winner
from tools.helper import flatten

# TODO: Make episode single round for faster training

# pylint: disable=import-outside-toplevel

log = logging.getLogger(__name__)

winner_in_episodes = []
MONTEACRLO_RUNS = 1000  # relevant for equity calculation if switched on


class CommunityData:
    """Data available to everybody"""

    def __init__(self, num_players):
        """data"""
        self.current_player_position = [False] * num_players  # ix[0] = dealer
        self.stage = [False] * 4  # one hot: preflop, flop, turn, river
        self.community_pot = None
        self.current_round_pot = None
        self.active_players = [False] * num_players  # one hot encoded, 0 = dealer
        self.big_blind = 0
        self.small_blind = 0
        self.legal_moves = [0 for action in Action]


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


class PlayerData:
    "Player specific information"

    def __init__(self):
        """data"""
        self.position = None
        self.equity_to_river_alive = 0
        self.equity_to_river_2plr = 0
        self.equity_to_river_3plr = 0
        self.stack = None


class HoldemTable(Env):
    """Pokergame environment"""

    def __init__(
        self,
        initial_stacks=100,
        small_blind=1,
        big_blind=2,
        render=False,
        funds_plot=True,
        max_raises_per_player_round=2,
        use_cpp_montecarlo=False,
        raise_illegal_moves=False,
        calculate_equity=False,
    ):
        """
        The table needs to be initialized once at the beginning

        Args:
            num_of_players (int): number of players that need to be added
            initial_stacks (real): initial stacks per placyer
            small_blind (real)
            big_blind (real)
            render (bool): render table after each move in graphical format
            funds_plot (bool): show plot of funds history at end of each episode
            max_raises_per_player_round (int): max raises per round per player

        """
        if use_cpp_montecarlo:
            import cppimport

            calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
            get_equity = calculator.montecarlo
        else:
            from tools.montecarlo_python import get_equity
        self.get_equity = get_equity
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.num_of_players = 0
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.render_switch = render
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
        self.funds_plot = funds_plot
        self.max_raises_per_player_round = max_raises_per_player_round
        self.calculate_equity = calculate_equity

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
        self.illegal_move_reward = -1
        self.action_space = Discrete(len(Action) - 2)
        # Define observation space with a placeholder shape
        from gymnasium.spaces import Box

        # We'll use a placeholder shape - actual shape will be determined in reset()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(124,), dtype=np.float32
        )
        self.first_action_for_hand = None

        self.raise_illegal_moves = raise_illegal_moves

    def reset(self, seed=None, options=None):
        """Reset after game over."""
        super().reset(seed=seed)

        self.observation = None
        self.reward = None
        self.info = None
        self.done = False
        self.funds_history = pd.DataFrame()
        self.first_action_for_hand = [True] * len(self.players)

        if not self.players:
            log.warning("No agents added. Add agents before resetting the environment.")
            return self.observation, self.info

        for player in self.players:
            player.stack = self.initial_stacks

        self.dealer_pos = 0
        self.player_cycle = PlayerCycle(
            self.players,
            dealer_idx=-1,
            max_raises_per_player_round=self.max_raises_per_player_round,
        )
        self._start_new_hand()
        self._get_environment()

        # Set observation space based on actual observation shape
        if self.observation_space is None:
            from gymnasium.spaces import Box

            obs_shape = self.observation.shape
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )

        # auto play for agents where autoplay is set
        if self._agent_is_autoplay() and not self.done:
            self.step(
                "initial_player_autoplay"
            )  # kick off the first action after bb by an autoplay agent

        return self.observation, self.info

    def step(self, action):  # pylint: disable=arguments-differ
        """
        Next player makes a move and a new environment is observed.

        Args:
            action: Used for testing only. Needs to be of Action type

        """
        # loop over step function, calling the agent's action method
        # until either the env id sone, or an agent is just a shell and
        # and will get a call from to the step function externally (e.g. via
        # keras-rl
        self.reward = 0
        self.acting_agent = self.player_cycle.idx
        if self._agent_is_autoplay():
            while self._agent_is_autoplay() and not self.done:
                log.debug("Autoplay agent. Call action method of agent.")
                self._get_environment()
                # call agent's action method
                action = self.current_player.agent_obj.action(
                    self.legal_moves, self.observation, self.info
                )
                if Action(action) not in self.legal_moves:
                    self._illegal_move(action)
                else:
                    self._execute_step(Action(action))
                    if self.first_action_for_hand[self.acting_agent] or self.done:
                        self.first_action_for_hand[self.acting_agent] = False
                        self._calculate_reward(action)

        else:  # action received from player shell (e.g. keras rl, not autoplay)
            self._get_environment()  # get legal moves
            if Action(action) not in self.legal_moves:
                self._illegal_move(action)
            else:
                self._execute_step(Action(action))
                if self.first_action_for_hand[self.acting_agent] or self.done:
                    self.first_action_for_hand[self.acting_agent] = False
                    self._calculate_reward(action)

            log.debug(
                f"Previous action reward for seat {self.acting_agent}: {self.reward}"
            )
        return self.array_everything, self.reward, self.done, self.info

    def _execute_step(self, action):
        self._process_decision(action)

        self._next_player()

        if self.stage in [Stage.END_HIDDEN, Stage.SHOWDOWN]:
            self._end_hand()
            self._start_new_hand()

        self._get_environment()

    def _illegal_move(self, action):
        log.warning(
            f"{action} is an Illegal move, try again. Currently allowed: {self.legal_moves}"
        )
        if self.raise_illegal_moves:
            raise ValueError(
                f"{action} is an Illegal move, try again. Currently allowed: {self.legal_moves}"
            )
        self.reward = self.illegal_move_reward

    def _agent_is_autoplay(self, idx=None):
        if not idx:
            return hasattr(self.current_player.agent_obj, "autoplay")
        return hasattr(self.players[idx].agent_obj, "autoplay")

    def _get_environment(self):
        """Observe the environment"""
        if not self.done:
            self._get_legal_moves()

        self.observation = None
        self.reward = 0
        self.info = None

        self.community_data = CommunityData(len(self.players))
        self.community_data.community_pot = self.community_pot / (self.big_blind * 100)
        self.community_data.current_round_pot = self.current_round_pot / (
            self.big_blind * 100
        )
        self.community_data.small_blind = self.small_blind
        self.community_data.big_blind = self.big_blind
        self.community_data.stage[np.minimum(self.stage.value, 3)] = (
            1  # pylint: disable= invalid-sequence-index
        )
        self.community_data.legal_moves = [
            action in self.legal_moves for action in Action
        ]
        # self.cummunity_data.active_players

        self.player_data = PlayerData()
        self.player_data.stack = [
            player.stack / (self.big_blind * 100) for player in self.players
        ]

        if not self.current_player:  # game over
            self.current_player = self.players[self.winner_ix]

        self.player_data.position = self.current_player.seat
        if self.calculate_equity:
            # Count only players who have cards (stack > 0 at start of hand)
            players_with_cards = sum(1 for player in self.players if player.cards)
            self.current_player.equity_alive = self.get_equity(
                set(self.current_player.cards),
                set(self.table_cards),
                players_with_cards,
                MONTEACRLO_RUNS,
            )
            self.player_data.equity_to_river_2plr = self.get_equity(
                set(self.current_player.cards),
                set(self.table_cards),
                players_with_cards,
                MONTEACRLO_RUNS,
            )
            self.player_data.equity_to_river_3plr = self.get_equity(
                set(self.current_player.cards),
                set(self.table_cards),
                players_with_cards,
                MONTEACRLO_RUNS,
            )
        else:
            self.current_player.equity_alive = np.nan
            self.player_data.equity_to_river_2plr = np.nan
            self.player_data.equity_to_river_3plr = np.nan
        self.current_player.equity_alive = self.get_equity(
            set(self.current_player.cards),
            set(self.table_cards),
            sum(1 for player in self.players if player.cards),
            1000,
        )
        self.player_data.equity_to_river_alive = self.current_player.equity_alive

        arr1 = np.array(list(flatten(self.player_data.__dict__.values())))
        arr2 = np.array(list(flatten(self.community_data.__dict__.values())))
        arr3 = np.array(
            [list(flatten(sd.__dict__.values())) for sd in self.stage_data]
        ).flatten()
        # arr_legal_only = np.array(self.community_data.legal_moves).flatten()

        self.array_everything = np.concatenate([arr1, arr2, arr3]).flatten()
        # Replace NaN values with 0
        self.array_everything = np.nan_to_num(self.array_everything, nan=0.0)

        self.observation = self.array_everything.astype(np.float32)
        self._get_legal_moves()

        self.info = {
            "player_data": self.player_data.__dict__,
            "community_data": self.community_data.__dict__,
            "stage_data": [stage.__dict__ for stage in self.stage_data],
            "legal_moves": self.legal_moves,
        }

        if self.render_switch:
            self.render()

    def _calculate_reward(self, last_action):
        """
        Preliminiary implementation of reward function

        - Currently missing potential additional winnings from future contributions
        """
        # if last_action == Action.FOLD:
        #     self.reward = -(
        #             self.community_pot + self.current_round_pot)
        # else:
        #     self.reward = self.player_data.equity_to_river_alive * (self.community_pot + self.current_round_pot) - \
        #                   (1 - self.player_data.equity_to_river_alive) * self.player_pots[self.current_player.seat]
        _ = last_action
        if self.done:
            won = 1 if not self._agent_is_autoplay(idx=self.winner_ix) else -1
            self.reward = self.initial_stacks * len(self.players) * won
            log.debug(f"Keras-rl agent has reward {self.reward}")

        elif len(self.funds_history) > 1:
            self.reward = (
                self.funds_history.iloc[-1, self.acting_agent]
                - self.funds_history.iloc[-2, self.acting_agent]
            )

        else:
            pass

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

        log.info(
            f"Seat {self.current_player.seat} ({self.current_player.name}): {action} - Remaining stack: {self.current_player.stack}, "
            f"Round pot: {self.current_round_pot}, Community pot: {self.community_pot}, "
            f"player pot: {self.player_pots[self.current_player.seat]}, min_call: {self.min_call}, "
            f"last_raise_amount: {self.last_raise_amount}"
        )

    def _start_new_hand(self):
        """Deal new cards to players and reset table states."""
        self._save_funds_history()

        if self._check_game_over():
            return

        log.info("")
        log.info("++++++++++++++++++")
        log.info("Starting new hand.")
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

    def _save_funds_history(self):
        """Keep track of player funds history"""
        funds_dict = {i: player.stack for i, player in enumerate(self.players)}
        self.funds_history = pd.concat(
            [self.funds_history, pd.DataFrame(funds_dict, index=[0])]
        )

    def _check_game_over(self):
        """Check if only one player has money left"""
        player_alive = []
        self.player_cycle.new_hand_reset()

        for idx, player in enumerate(self.players):
            if player.stack > 0:
                player_alive.append(True)
                # Reset player status to active if they have chips
                if idx < len(self.player_status):
                    self.player_status[idx] = True
                else:
                    self.player_status.append(True)
                # Ensure player is not marked as folder (unless they actually folded)
                if idx < len(self.player_cycle.folder):
                    self.player_cycle.folder[idx] = False
            else:
                # Player has $0 stack - they should be eliminated
                if idx < len(self.player_status):
                    self.player_status[idx] = False
                else:
                    self.player_status.append(False)
                # Mark as folder to remove from current hand
                if idx < len(self.player_cycle.folder):
                    self.player_cycle.folder[idx] = True
                # Deactivate player - they can't make moves
                self.player_cycle.deactivate_player(idx)
                # Mark as out of cash (but they didn't contribute this hand)
                if idx < len(self.player_cycle.out_of_cash_but_contributed):
                    self.player_cycle.out_of_cash_but_contributed[idx] = False

        remaining_players = sum(player_alive)
        if remaining_players < 2:
            self._game_over()
            return True
        return False

    def _game_over(self):
        """End of an episode."""
        log.info("Game over.")
        self.done = True
        player_names = [f"{i} - {player.name}" for i, player in enumerate(self.players)]
        self.funds_history.columns = player_names
        if self.funds_plot:
            self.funds_history.reset_index(drop=True).plot()
        log.info(self.funds_history)
        plt.show()

        winner_in_episodes.append(self.winner_ix)
        league_table = pd.Series(winner_in_episodes).value_counts()
        best_player = league_table.index[0]
        log.info(league_table)
        log.info(f"Best Player: {best_player}")

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
        """Check if the current betting round is complete using unified rules."""
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

        log.info(f"===ROUND: {self.stage} ===")
        self._clean_up_pots()

    def _clean_up_pots(self):
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)

    def _end_hand(self):
        self._clean_up_pots()
        self.winner_ix = self._get_winner()
        self._award_winner(self.winner_ix)

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
            for i, player in enumerate(self.players):
                player.stack += remains[i]

    def _next_dealer(self):
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
                log.info("Betting round complete based on unified rules")
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
        log.info(
            f"Closing round. Community pot: {self.community_pot}, "
            f"Current round pot: {self.current_round_pot}, "
            f"Min call: {self.min_call}"
        )

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
        log.info(f"Cards on table: {self.table_cards}")

    def render(self, mode="human"):
        """Render the current state"""
        if mode != "human":
            return
        screen_width = 600
        screen_height = 400
        table_radius = 200
        face_radius = 10

        if self.viewer is None:
            self.viewer = PygletWindow(screen_width + 50, screen_height + 50)
        self.viewer.reset()
        self.viewer.circle(
            screen_width / 2, screen_height / 2, table_radius, color=BLUE, thickness=0
        )

        for i in range(len(self.players)):
            degrees = i * (360 / len(self.players))
            radian = degrees * (np.pi / 180)
            x = (face_radius + table_radius) * np.cos(radian) + screen_width / 2
            y = (face_radius + table_radius) * np.sin(radian) + screen_height / 2
            if self.player_cycle.alive[i]:
                color = GREEN
            else:
                color = RED
            self.viewer.circle(x, y, face_radius, color=color, thickness=2)

            try:
                if i == self.current_player.seat:
                    self.viewer.rectangle(x - 60, y, 150, -50, (255, 0, 0, 10))
            except AttributeError:
                pass
            self.viewer.text(
                f"{self.players[i].name}", x - 60, y - 15, font_size=10, color=WHITE
            )
            self.viewer.text(
                f"Player {self.players[i].seat}: {self.players[i].cards}",
                x - 60,
                y,
                font_size=10,
                color=WHITE,
            )
            equity_alive = int(round(float(self.players[i].equity_alive) * 100))

            self.viewer.text(
                f"${self.players[i].stack} (EQ: {equity_alive}%)",
                x - 60,
                y + 15,
                font_size=10,
                color=WHITE,
            )
            try:
                self.viewer.text(
                    self.players[i].last_action_in_stage,
                    x - 60,
                    y + 30,
                    font_size=10,
                    color=WHITE,
                )
            except IndexError:
                pass
            x_inner = (-face_radius + table_radius - 60) * np.cos(
                radian
            ) + screen_width / 2
            y_inner = (-face_radius + table_radius - 60) * np.sin(
                radian
            ) + screen_height / 2
            self.viewer.text(
                f"${self.player_pots[i]}", x_inner, y_inner, font_size=10, color=WHITE
            )
            self.viewer.text(
                f"{self.table_cards}",
                screen_width / 2 - 40,
                screen_height / 2,
                font_size=10,
                color=WHITE,
            )
            self.viewer.text(
                f"${self.community_pot}",
                screen_width / 2 - 15,
                screen_height / 2 + 30,
                font_size=10,
                color=WHITE,
            )
            self.viewer.text(
                f"${self.current_round_pot}",
                screen_width / 2 - 15,
                screen_height / 2 + 50,
                font_size=10,
                color=WHITE,
            )

            x_button = (-face_radius + table_radius - 20) * np.cos(
                radian
            ) + screen_width / 2
            y_button = (-face_radius + table_radius - 20) * np.sin(
                radian
            ) + screen_height / 2
            try:
                if i == self.player_cycle.dealer_idx:
                    self.viewer.circle(x_button, y_button, 5, color=BLUE, thickness=2)
            except AttributeError:
                pass

        self.viewer.update()


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
