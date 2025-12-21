"""Enums"""

from enum import Enum


class Action(Enum):
    """Allowed actions"""

    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_1_4_POT = 3  # 0.25  # Blocking/small bet
    BET_1_3_POT = 4  # 0.33  # Standard small bet
    BET_1_2_POT = 5  # 0.50  # Medium bet
    BET_2_3_POT = 6  # 0.66  # 2/3 pot - very common
    BET_3_4_POT = 7  # 0.75  # 3/4 pot - standard
    BET_POT = 8  # 1.00  # Pot-sized - strong
    BET_3_2_POT = 9  # 1.50  # 1.5x pot - polarizing
    BET_2_POT = 10  # 2.00  # 2x pot - extreme overbet
    BET_MIN_RAISE = 11  # Minimum raise - safest raise
    ALL_IN = 12
    SMALL_BLIND = 13
    BIG_BLIND = 14


class Stage(Enum):
    """Allowed actions"""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5
