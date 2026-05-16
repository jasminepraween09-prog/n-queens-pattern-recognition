"""
N-Queens — Exhaustive Depth-First Search (Backtracking)
========================================================

Approach
--------
A column-by-column depth-first search. At column `c` we try every row `r`
that does not collide with a queen already placed in some earlier column.
Collision-checks are O(1) thanks to three boolean sets:
    - `cols`           : rows already used
    - `diag1`           : "\\" diagonals already used (r - c)
    - `diag2`           : "/" diagonals already used (r + c)

The search backtracks the moment a partial placement becomes invalid, so the
effective branching factor is far smaller than N.  Still, the worst-case
complexity is O(N!) which makes the algorithm intractable for very large N.

Because the assignment also asks us to run N = 500, a wall-clock timeout
(default: 5 minutes) lets the search exit gracefully when it cannot finish.

The script tracks execution time (`time.perf_counter`) and peak memory
(`tracemalloc`) for every value of N requested on the command line, then
prints a small summary table.

Author : Jasmine Praween
Course : Pattern Recognition — Assignment (N-Queens)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Custom exception used by the timeout guard inside the recursive search.
# ---------------------------------------------------------------------------
class SearchTimeout(Exception):
    """Raised internally when the DFS exceeds the configured wall-clock cap."""


@dataclass
class DFSResult:
    """Container for one run of the DFS solver."""
    n: int
    solution: Optional[List[int]]      # solution[c] = row of the queen in column c
    elapsed_seconds: float
    peak_memory_kb: float
    nodes_expanded: int
    timed_out: bool


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------
def solve_nqueens_dfs(
    n: int,
    time_limit_s: float = 300.0,
    seed: Optional[int] = None,
) -> DFSResult:
    """
    Find ONE valid placement of n queens on an n x n board using exhaustive
    depth-first backtracking.

    Parameters
    ----------
    n            : board size.
    time_limit_s : wall-clock cap (seconds).  If the cap is reached the
                   function returns with `timed_out=True` and no solution.
    seed         : seed for the row-ordering shuffle (None ⇒ fully random).

    Returns
    -------
    DFSResult with timing, memory and the solution (or None if it timed out).
    """
    if seed is not None:
        random.seed(seed)

    placement: List[int] = [-1] * n
    cols, diag1, diag2 = set(), set(), set()
    nodes = 0
    deadline = time.perf_counter() + time_limit_s

    # We still explore *every* row in every column (this is exhaustive DFS),
    # but we randomise the order in which rows are tried so we are not stuck
    # with the pathological "always start at row 0" branching that makes
    # plain backtracking degenerate for medium N.
    row_orders = [random.sample(range(n), n) for _ in range(n)]

    tracemalloc.start()
    t0 = time.perf_counter()

    def backtrack(col: int) -> bool:
        nonlocal nodes
        if time.perf_counter() > deadline:
            raise SearchTimeout
        if col == n:
            return True
        for row in row_orders[col]:
            if row in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            placement[col] = row
            cols.add(row)
            diag1.add(row - col)
            diag2.add(row + col)
            nodes += 1
            if backtrack(col + 1):
                return True
            cols.remove(row)
            diag1.remove(row - col)
            diag2.remove(row + col)
            placement[col] = -1
        return False

    solution: Optional[List[int]] = None
    timed_out = False
    try:
        # Recursion limit needs to grow with N because Python's default is
        # only 1000 frames and the search is exactly N levels deep.
        sys.setrecursionlimit(max(10_000, 10 * n))
        if backtrack(0):
            solution = placement.copy()
    except SearchTimeout:
        timed_out = True

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return DFSResult(
        n=n,
        solution=solution,
        elapsed_seconds=elapsed,
        peak_memory_kb=peak / 1024.0,
        nodes_expanded=nodes,
        timed_out=timed_out,
    )


# ---------------------------------------------------------------------------
# Validation helper — confirms the returned board really is a solution.
# ---------------------------------------------------------------------------
def is_valid_solution(placement: List[int]) -> bool:
    n = len(placement)
    if any(r < 0 or r >= n for r in placement):
        return False
    if len(set(placement)) != n:
        return False
    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            if abs(placement[c1] - placement[c2]) == abs(c1 - c2):
                return False
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustive DFS solver for N-Queens.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[10, 30, 50, 100, 200, 500],
        help="Values of N to benchmark.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-run wall-clock cap in seconds (default 300).",
    )
    args = parser.parse_args()

    print(f"{'N':>5} | {'Time (s)':>10} | {'Peak Mem (KB)':>14} | "
          f"{'Nodes':>14} | Status")
    print("-" * 70)

    for n in args.sizes:
        result = solve_nqueens_dfs(n, time_limit_s=args.timeout)
        if result.timed_out:
            status = f"TIMEOUT (>{args.timeout:.0f}s)"
        elif result.solution and is_valid_solution(result.solution):
            status = "OK"
        else:
            status = "NO SOLUTION"
        print(f"{n:>5} | {result.elapsed_seconds:>10.4f} | "
              f"{result.peak_memory_kb:>14.2f} | {result.nodes_expanded:>14d} | {status}")


if __name__ == "__main__":
    main()
