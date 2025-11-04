#!/usr/bin/env python3
"""
CS170 Project #1 — Eight Puzzle Solver (initial setup)

This version only defines the basic state representation and utility functions.
"""

from typing import List, Tuple

State = Tuple[int, ...]  # Flattened tuple; 0 = blank


def chunk(state: State, k: int) -> List[List[int]]:
    """Convert flattened state into k x k 2D list for printing."""
    return [list(state[i * k:(i + 1) * k]) for i in range(k)]


def pretty_board(state: State, k: int, blank_char: str = "b") -> str:
    """Nicely format the board for output."""
    rows = []
    for row in chunk(state, k):
        rows.append(" ".join(blank_char if v == 0 else str(v) for v in row))
    return "\n".join(rows)


if __name__ == "__main__":
    example = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    print(pretty_board(example, 3))
