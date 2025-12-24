"""Entry point scripts for neuron poker commands."""

import sys
from main import command_line_parser


def main_random():
    """Run random agents"""
    sys.argv = ["main.py", "selfplay", "random"]
    command_line_parser()


def main_keypress():
    """Run keypress agent"""
    sys.argv = ["main.py", "selfplay", "keypress"]
    command_line_parser()
