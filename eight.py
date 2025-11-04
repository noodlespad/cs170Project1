#!/usr/bin/env python3
"""
CS170 Project #1 — Eight Puzzle Solver

Implements:
1) Uniform Cost Search (UCS)
2) A* with Misplaced Tile heuristic
3) A* with Euclidean Distance heuristic

Run:
    python3 eight_puzzle_solver.py
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict, Set
import heapq
import math
import sys

State = Tuple[int, ...]  # Flattened tuple representation; 0 denotes the blank


def chunk(state: State, k: int) -> List[List[int]]:
    """Convert flattened state into k x k 2D list for pretty printing."""
    return [list(state[i * k:(i + 1) * k]) for i in range(k)]


def pretty_board(state: State, k: int, blank_char: str = "b") -> str:
    """String format to match the assignment's trace style."""
    rows = []
    for row in chunk(state, k):
        rows.append(" ".join(blank_char if v == 0 else str(v) for v in row))
    return "\n".join(rows)


@dataclass(order=True)
class PrioritizedItem:
    f: int
    tie: int
    node: "Node" = field(compare=False)


@dataclass
class Node:
    state: State
    parent: Optional["Node"]
    action: Optional[str]
    g: int  # path cost so far (depth if all step costs = 1)
    h: int  # heuristic estimate

    def path(self) -> List["Node"]:
        n: Optional["Node"] = self
        out: List["Node"] = []
        while n is not None:
            out.append(n)
            n = n.parent
        return list(reversed(out))


class PuzzleProblem:
    def __init__(self, initial: State, goal: Optional[State] = None) -> None:
        self.initial = initial
        k_float = math.sqrt(len(initial))
        if int(k_float) != k_float:
            raise ValueError("State length must be a perfect square (e.g., 9 for 8-puzzle).")
        self.k = int(k_float)
        if goal is None:
            self.goal = tuple(list(range(1, self.k*self.k)) + [0])
        else:
            self.goal = goal

        # Precompute goal positions for heuristics (value -> (r, c))
        self.goal_pos: Dict[int, Tuple[int, int]] = {}
        for idx, val in enumerate(self.goal):
            self.goal_pos[val] = (idx // self.k, idx % self.k)

    def is_goal(self, s: State) -> bool:
        return s == self.goal

    def neighbors(self, s: State) -> List[Tuple[str, State]]:
        """Return list of (action, new_state) from state s. Actions are 'Up','Down','Left','Right'."""
        k = self.k
        zero_idx = s.index(0)
        zr, zc = divmod(zero_idx, k)
        moves = []
        def swap(idx1: int, idx2: int) -> State:
            lst = list(s)
            lst[idx1], lst[idx2] = lst[idx2], lst[idx1]
            return tuple(lst)

        if zr > 0:
            moves.append(("Up", swap(zero_idx, zero_idx - k)))
        if zr < k - 1:
            moves.append(("Down", swap(zero_idx, zero_idx + k)))
        if zc > 0:
            moves.append(("Left", swap(zero_idx, zero_idx - 1)))
        if zc < k - 1:
            moves.append(("Right", swap(zero_idx, zero_idx + 1)))
        return moves

    def misplaced_tiles(self, s: State) -> int:
        """Heuristic: number of tiles out of place (excluding blank)."""
        return sum(1 for i, v in enumerate(s) if v != 0 and v != self.goal[i])

    def euclidean_distance(self, s: State) -> int:
        """Heuristic: sum of Euclidean distances between tiles and goal positions (excluding blank).
        Note: returns an int by flooring the sum to keep output compact; keeping it as float also works,
        but the sample interface shows integer g(n), h(n) values. You can switch to round() if desired.
        """
        total = 0.0
        k = self.k
        for idx, val in enumerate(s):
            if val == 0:
                continue
            r, c = divmod(idx, k)
            gr, gc = self.goal_pos[val]
            total += math.hypot(r - gr, c - gc)
        return int(total)

    def is_solvable(self, s: State) -> bool:
        """Check solvability using parity of inversions (and blank row for even-width)."""
        arr = [v for v in s if v != 0]
        inversions = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inversions += 1
        # For odd k, puzzle is solvable iff inversions is even.
        if self.k % 2 == 1:
            return inversions % 2 == 0
        # For even k, use blank row counting from bottom (row=1 at bottom)
        blank_row_from_bottom = self.k - (s.index(0) // self.k)
        if blank_row_from_bottom % 2 == 0:  # blank on even row from bottom
            return inversions % 2 == 1
        else:  # blank on odd row from bottom
            return inversions % 2 == 0


def a_star_search(
    problem: PuzzleProblem,
    heuristic: Callable[[State], int],
    trace: bool = True
) -> Tuple[Optional[Node], int, int]:
    """
    Generic A* / UCS search.
    Returns: (goal_node or None, nodes_expanded, max_frontier_size)
    """
    start_h = heuristic(problem.initial)
    start = Node(state=problem.initial, parent=None, action=None, g=0, h=start_h)

    frontier: List[PrioritizedItem] = []
    tie_counter = 0
    heapq.heappush(frontier, PrioritizedItem(f=start.g + start.h, tie=tie_counter, node=start))
    tie_counter += 1

    # For graph search: best known g for each seen state
    best_g: Dict[State, int] = {start.state: 0}
    explored: Set[State] = set()

    nodes_expanded = 0
    max_frontier_size = 1

    # No pre-loop print: avoid duplicating the initial state in the trace.

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_item = heapq.heappop(frontier)
        current = current_item.node

        # Trace line that matches the sample
        if trace:
            print("\nThe best state to expand with  g(n) = {} and h(n) = {} is…".format(current.g, current.h))
            print(pretty_board(current.state, problem.k))
            print("Expanding this node…")

        if problem.is_goal(current.state):
            return current, nodes_expanded, max_frontier_size

        explored.add(current.state)
        nodes_expanded += 1

        for action, child_state in problem.neighbors(current.state):
            new_g = current.g + 1  # uniform step cost = 1
            if child_state in explored and new_g >= best_g.get(child_state, math.inf):
                continue
            if new_g < best_g.get(child_state, math.inf):
                best_g[child_state] = new_g
                child_h = heuristic(child_state)
                child = Node(state=child_state, parent=current, action=action, g=new_g, h=child_h)
                heapq.heappush(frontier, PrioritizedItem(f=child.g + child.h, tie=tie_counter, node=child))
                tie_counter += 1

    return None, nodes_expanded, max_frontier_size


def read_puzzle_from_user(k: int) -> State:
    """Prompt user to enter k rows of the puzzle; blank is zero."""
    print(" Enter your puzzle, use a zero to represent the blank")
    data: List[int] = []
    for i in range(k):
        row = input(f"Enter the {['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth'][i] if i<9 else f'{i+1}th'} row, use space or tabs between numbers   ").strip()
        parts = row.split()
        if len(parts) != k:
            print(f"Expected {k} numbers; got {len(parts)}. Try again.", file=sys.stderr)
            sys.exit(1)
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            print("Non-integer value found. Aborting.", file=sys.stderr)
            sys.exit(1)
        data.extend(nums)
    if sorted(data) != list(range(k * k)):
        print(f"Input must contain all numbers 0..{k*k - 1} exactly once.", file=sys.stderr)
        sys.exit(1)
    return tuple(data)


def main():
    # You may change this ID string to your own student ID.
    student_id = "XXX"  # <-- replace with your ID if desired
    print(f"Welcome to {student_id} 8 puzzle solver.")

    # Choose default/own puzzle
    print('Type "1" to use a default puzzle, or "2" to enter your own puzzle.')
    choice = input().strip()
    if choice not in {"1", "2"}:
        print("Invalid selection.", file=sys.stderr)
        return

    # You can change the default initial state here if desired
    default_initial = (1, 2, 3,
                       4, 8, 0,
                       7, 6, 5)  # matches sample in the PDF
    k = 3  # default to 3x3 (8-puzzle). Change to 4 for 15-puzzle, etc.

    if choice == "1":
        initial = default_initial
    else:
        initial = read_puzzle_from_user(k)

    problem = PuzzleProblem(initial=initial, goal=None)

    # Algorithm selection
    print("\nEnter your choice of algorithm")
    print("1. Uniform Cost Search")
    print("2. A* with the Misplaced Tile heuristic.")
    print("3. A* with the Euclidean distance heuristic.")
    alg_choice = input().strip()

    if alg_choice == "1":
        heuristic = (lambda s: 0)
    elif alg_choice == "2":
        heuristic = problem.misplaced_tiles
    elif alg_choice == "3":
        heuristic = problem.euclidean_distance
    else:
        print("Invalid selection.", file=sys.stderr)
        return

    # Optional: warn if unsolvable
    if not problem.is_solvable(problem.initial):
        print("\nWarning: This initial puzzle is not solvable. The search will run but cannot reach the goal.")

    goal_node, nodes_expanded, max_q = a_star_search(problem, heuristic, trace=True)

    if goal_node is not None:
        print("\nGoal!!!")
        print(f"\nTo solve this problem the search algorithm expanded a total of {nodes_expanded} nodes.")
        print(f"The maximum number of nodes in the queue at any one time: {max_q}.")
        print(f"The depth of the goal node was {goal_node.g}.")

        # Optional: reconstruct and print the exact sequence of actions (extra credit in PDF).
        # You can comment this block out if not needed for grading.
        path_nodes = goal_node.path()
        actions = [n.action for n in path_nodes if n.action is not None]
        if actions:
            print("\nSolution (sequence of actions):")
            print(" -> ".join(actions))
    else:
        print("\nNo solution found (frontier exhausted).")
        print(f"Nodes expanded: {nodes_expanded}")
        print(f"Max queue size: {max_q}")


if __name__ == "__main__":
    main()
