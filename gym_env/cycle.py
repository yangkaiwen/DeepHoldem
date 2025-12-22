"""Cycle class that handles the low level logic of the game."""

import logging
import numpy as np

# pylint: disable=import-outside-toplevel
log = logging.getLogger(__name__)


class PlayerCycle:
    """Handle the circularity of the Table."""

    def __init__(
        self,
        lst,
        start_idx=0,
        dealer_idx=0,
    ):
        """Cycle over a list"""
        self.lst = lst
        self.start_idx = start_idx
        self.size = len(lst)
        self.last_aggressor_idx = None  # Track the last aggressor (bettor/raiser)
        self.round_number_in_street = 0
        self.idx = 0
        self.dealer_idx = dealer_idx
        self.can_still_make_moves_in_this_hand = (
            []
        )  # if the player can still play in this round
        self.alive = [True] * len(
            self.lst
        )  # if the player can still play in the following rounds
        self.out_of_cash_but_contributed = [False] * len(self.lst)
        self.folder = [False] * len(self.lst)
        self.checkers = 0
        self.preflop_no_raise = True  # Track if preflop had no raises

        # Track contributions for unified rule checking
        self.player_contributions = [0] * len(self.lst)
        self.current_street_contributions = [0] * len(self.lst)
        self.new_hand_reset()
        self.preflop_no_raise = True  # Track if preflop had no raises

        # Track contributions for unified rule checking
        self.player_contributions = [0] * len(self.lst)
        self.current_street_contributions = [0] * len(self.lst)

    def new_hand_reset(self):
        """Reset state if a new hand is dealt"""
        self.idx = self.start_idx
        self.can_still_make_moves_in_this_hand = [True] * len(self.lst)
        self.out_of_cash_but_contributed = [False] * len(self.lst)
        self.folder = [False] * len(self.lst)
        self.player_contributions = [0] * len(self.lst)
        self.current_street_contributions = [0] * len(self.lst)
        self.last_aggressor_idx = None
        self.preflop_no_raise = True
        self.checkers = 0
        self.round_number_in_street = 0

    def new_street_reset(self):
        """Reset the state for the next stage: flop, turn or river"""
        self.round_number_in_street = 0
        self.idx = self.dealer_idx
        self.checkers = 0
        self.last_aggressor_idx = None
        self.current_street_contributions = [0] * len(self.lst)
        self.preflop_no_raise = False  # Only preflop can have the no-raise scenario

        # Reset can_still_make_moves_in_this_hand
        # Players who are all-in or folded cannot make moves
        for i in range(len(self.lst)):
            if self.folder[i]:
                # Folded players: cannot make moves
                self.can_still_make_moves_in_this_hand[i] = False
            elif self.out_of_cash_but_contributed[i]:
                # All-in players: cannot make moves but are still in the hand
                self.can_still_make_moves_in_this_hand[i] = False
            else:
                # Players with chips: can make moves
                self.can_still_make_moves_in_this_hand[i] = True

    def next_player(self, step=1):
        """Switch to the next player in the round."""
        # Get non-folded players
        non_folded_indices = self.get_non_folded_players()

        if len(non_folded_indices) < 2:
            log.debug("Less than 2 non-folded players remaining")
            return False

        # UNIFIED RULE CHECKING
        # Check if betting round should end
        if self._should_end_betting_round():
            log.debug("Betting round should end based on unified rules")
            return False

        self.idx += step
        self.idx %= len(self.lst)

        # Check if all players have checked (non-preflop scenario)
        # Only count checks from players who can still act
        players_who_can_act = [
            i for i in non_folded_indices if not self.out_of_cash_but_contributed[i]
        ]
        if (
            players_who_can_act
            and self.checkers >= len(players_who_can_act)
            and not self.preflop_no_raise
        ):
            log.debug("All players who can act have checked")
            return False

        # Find next player who can still make moves (not folded, not all-in)
        original_idx = self.idx
        found = False
        attempts = 0

        while attempts < len(self.lst):
            # Player can act if they are not folded and not all-in
            if self.can_still_make_moves_in_this_hand[self.idx]:
                found = True
                break

            self.idx += 1
            self.idx %= len(self.lst)
            attempts += 1

        if not found:
            log.debug("No active players found after searching all players")
            return False

        self.update_alive()
        return self.lst[self.idx]

    def _should_end_betting_round(self):
        """
        Check if betting round should end based on unified rules.
        """
        # Get non-folded players
        non_folded_indices = self.get_non_folded_players()

        if len(non_folded_indices) < 2:
            return True  # Only 0 or 1 non-folded players remain

        # Check if all non-folded players have contributed equally
        # Get max contribution among non-folded players
        max_contribution = 0
        for i in non_folded_indices:
            if self.player_contributions[i] > max_contribution:
                max_contribution = self.player_contributions[i]

        # Check if all non-folded players have either:
        # 1. Contributed the max amount, OR
        # 2. Are all-in (out_of_cash_but_contributed)
        all_contributed_equally = True
        for i in non_folded_indices:
            if (
                self.can_still_make_moves_in_this_hand[i]  # Can still act
                and not self.out_of_cash_but_contributed[i]  # Not all-in
                and self.player_contributions[i] < max_contribution
            ):  # Hasn't matched bet
                all_contributed_equally = False
                break

        if not all_contributed_equally:
            return False

        # Special case: all non-folded players are all-in
        all_all_in = all(
            self.out_of_cash_but_contributed[i] for i in non_folded_indices
        )
        if all_all_in:
            log.debug("All non-folded players are all-in")
            return True

        # If we get here, all players have contributed equally
        # Now check if action has returned to last aggressor

        # Special case: pre-flop no-raise scenario
        if self.preflop_no_raise and self.last_aggressor_idx is None:
            # BB is the default aggressor in pre-flop no-raise scenario
            bb_idx = (self.dealer_idx + 2) % len(self.lst)
            if self.idx == bb_idx:
                log.debug(
                    "Action returned to BB (default aggressor in pre-flop no-raise)"
                )
                return True

        # Check if we have a last aggressor and action has returned to them
        if self.last_aggressor_idx is not None:
            # Check if last aggressor can still act (not all-in or folded)
            if (
                self.last_aggressor_idx in non_folded_indices
                and not self.out_of_cash_but_contributed[self.last_aggressor_idx]
            ):
                if self.idx == self.last_aggressor_idx:
                    log.debug(
                        f"Action returned to last aggressor at position {self.last_aggressor_idx}"
                    )
                    return True
            else:
                # Last aggressor is all-in or folded, so betting round should end
                log.debug("Last aggressor is all-in or folded")
                return True

        # For rounds after pre-flop, if there's no aggressor (everyone checks),
        # we should automatically go to next round
        if self.last_aggressor_idx is None and not self.preflop_no_raise:
            # Check if all players who can act have checked
            players_who_can_act = [
                i for i in non_folded_indices if not self.out_of_cash_but_contributed[i]
            ]
            if len(players_who_can_act) == 0:
                # All players are all-in
                return True
            elif self.checkers >= len(players_who_can_act):
                log.debug("No aggressor and all players who can act have checked")
                return True

        return False

    def next_dealer(self):
        """Move the dealer to the next player that's still in the round."""
        self.dealer_idx += 1
        self.dealer_idx %= len(self.lst)

        while True:
            if not self.folder[self.dealer_idx]:
                break

            self.dealer_idx += 1
            self.dealer_idx %= len(self.lst)

        return self.lst[self.dealer_idx]

    def set_idx(self, idx):
        """Set the index to a specific player"""
        self.idx = idx

    def deactivate_player(self, idx):
        """Deactivate a player if he has folded or is out of cash."""
        assert self.can_still_make_moves_in_this_hand[idx], "Already deactivated"
        self.can_still_make_moves_in_this_hand[idx] = False
        # If the last aggressor folds, we need to update
        if idx == self.last_aggressor_idx:
            self.last_aggressor_idx = None

    def deactivate_current(self):
        """Deactivate the current player if he has folded or is out of cash."""
        assert self.can_still_make_moves_in_this_hand[self.idx], "Already deactivated"
        self.can_still_make_moves_in_this_hand[self.idx] = False
        # If the last aggressor folds, we need to update
        if self.idx == self.last_aggressor_idx:
            self.last_aggressor_idx = None

    def mark_folder(self):
        """Mark a player as no longer eligible to win cash from the current hand"""
        self.folder[self.idx] = True
        self.can_still_make_moves_in_this_hand[self.idx] = False

    def mark_raiser(self, contribution=0):
        """Mark a raise for the current player."""
        self.last_aggressor_idx = self.idx
        self.preflop_no_raise = False
        if contribution > 0:
            self.player_contributions[self.idx] += contribution
            self.current_street_contributions[self.idx] += contribution

    def mark_checker(self):
        """Counter the number of checks in the round"""
        self.checkers += 1

    def mark_out_of_cash_but_contributed(self):
        """Mark current player as a raiser or caller, but is out of cash."""
        self.out_of_cash_but_contributed[self.idx] = True
        self.can_still_make_moves_in_this_hand[self.idx] = False
        # Don't deactivate current - all-in players are still "active" for the hand
        # They just can't make more moves

    def mark_bb(self, contribution=0):
        """BB is the default aggressor in pre-flop no-raise scenario"""
        self.last_aggressor_idx = self.idx
        if contribution > 0:
            self.player_contributions[self.idx] += contribution
            self.current_street_contributions[self.idx] += contribution

    def update_contribution(self, idx, amount):
        """Update a player's contribution amount."""
        self.player_contributions[idx] += amount
        self.current_street_contributions[idx] += amount

    def update_alive(self):
        """Update the alive property"""
        self.alive = np.array(self.can_still_make_moves_in_this_hand) + np.array(
            self.out_of_cash_but_contributed
        )

    def get_potential_winners(self):
        """Players eligible to win the pot"""
        potential_winners = np.logical_and(
            np.logical_or(
                np.array(self.can_still_make_moves_in_this_hand),
                np.array(self.out_of_cash_but_contributed),
            ),
            np.logical_not(np.array(self.folder)),
        )
        return potential_winners

    def get_non_folded_players(self):
        """Get indices of players who haven't folded"""
        return [i for i in range(len(self.lst)) if not self.folder[i]]
