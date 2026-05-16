"""
N-Queens — Greedy Local Search (Steepest-Ascent Hill Climbing with Random Restarts)
====================================================================================

State representation
--------------------
The board is encoded as a list of length N where `state[c]` is the row of the
queen in column c.  Exactly one queen per column means column-clashes are
impossible by construction; we only have to fight rows and diagonals.

Cost function (heuristic h)
---------------------------
h(state) = number of pairs of queens that attack each other.
A goal state has h = 0.

Move set
--------
A "neighbour" is obtained by moving a single queen to any other row in its
own column.  Each state therefore has N * (N-1) neighbours.

Algorithm
---------
Steepest-ascent Hill Climbing with random restarts:
    1. Start from a random permutation of rows.
    2. Examine every neighbour and pick the one with the lowest h.
    3. If no neighbour improves h, we are stuck on a local optimum / plateau.
       Restart from a fresh random state.
Restarts repeat until h = 0 is reached, `max_restarts` is exhausted, or the
wall-clock cap is hit.

Performance trick
-----------------
A naive "recompute h from scratch for every neighbour" would cost O(N^4) per
hill-climb step.  We instead maintain *line-counts* (row, "\" diag, "/" diag)
and evaluate every candidate move in O(1) using a closed-form ΔE.  Total
cost per step becomes O(N²), which keeps the solver responsive up to a few
hundred queens.

Author : Jasmine Praween
Course : Pattern Recognition — Assignment (N-Queens)
"""

from __future__ import annotations

import argparse
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class HCResult:
    n: int
    solution: Optional[List[int]]
    elapsed_seconds: float
    peak_memory_kb: float
    restarts: int
    total_steps: int
    timed_out: bool


# ---------------------------------------------------------------------------
# Line-count helpers — list-based for speed on big boards
# ---------------------------------------------------------------------------
def build_counts(state: List[int]) -> Tuple[List[int], List[int], List[int], int]:
    """
    Return (row_count, d1_count, d2_count, attacking_pairs) for `state`.
    Counters are plain Python lists indexed by:
        rows[r]                 — row r
        d1[r - c + N]           — "\\" diagonal (offset to keep index ≥ 0)
        d2[r + c]               — "/" diagonal
    Indexing a list is ~3× faster than a dict in CPython, which matters
    enormously when the inner loop runs N² times per step.
    """
    n = len(state)
    rows = [0] * n
    d1 = [0] * (2 * n)
    d2 = [0] * (2 * n)
    for c, r in enumerate(state):
        rows[r] += 1
        d1[r - c + n] += 1
        d2[r + c] += 1
    pairs = 0
    for arr in (rows, d1, d2):
        for v in arr:
            if v > 1:
                pairs += v * (v - 1) // 2
    return rows, d1, d2, pairs


def delta_move(
    state: List[int],
    col: int,
    new_row: int,
    rows: List[int],
    d1: List[int],
    d2: List[int],
    n: int,
) -> int:
    """O(1) cost change for moving the queen in `col` to `new_row`."""
    old_row = state[col]
    if old_row == new_row:
        return 0
    return (
        -(rows[old_row] - 1)
        - (d1[old_row - col + n] - 1)
        - (d2[old_row + col] - 1)
        + rows[new_row]
        + d1[new_row - col + n]
        + d2[new_row + col]
    )


def apply_move(
    state: List[int],
    col: int,
    new_row: int,
    rows: List[int],
    d1: List[int],
    d2: List[int],
    n: int,
) -> None:
    old_row = state[col]
    rows[old_row] -= 1
    d1[old_row - col + n] -= 1
    d2[old_row + col] -= 1
    rows[new_row] += 1
    d1[new_row - col + n] += 1
    d2[new_row + col] += 1
    state[col] = new_row


# ---------------------------------------------------------------------------
# One steepest-ascent climb (with limited sideways moves)
# ---------------------------------------------------------------------------
def climb(
    state: List[int],
    deadline: float,
    max_sideways: int = 100,
) -> Tuple[List[int], int, int]:
    """
    Steepest-ascent climb that also allows up to `max_sideways` consecutive
    *neutral* moves (Δ = 0).  This is the well-known plateau-escape trick
    from Russell & Norvig — it bumps the success rate of hill climbing on
    N-Queens dramatically (from ~14 % to ~94 % for N = 8).

    Returns (final_state, attacking_pairs, steps_taken).
    """
    n = len(state)
    rows, d1, d2, energy = build_counts(state)
    steps = 0
    sideways_used = 0

    while energy > 0 and time.perf_counter() < deadline:
        best_delta = 1                     # any non-positive delta is interesting
        best_moves: List[Tuple[int, int]] = []
        for c in range(n):
            cur_row = state[c]
            for r in range(n):
                if r == cur_row:
                    continue
                d = delta_move(state, c, r, rows, d1, d2, n)
                if d < best_delta:
                    best_delta = d
                    best_moves = [(c, r)]
                elif d == best_delta:
                    best_moves.append((c, r))
        if best_delta > 0 or not best_moves:
            break                           # truly stuck — no neutral move helps
        if best_delta == 0:
            if sideways_used >= max_sideways:
                break
            sideways_used += 1
        else:
            sideways_used = 0               # found a strict improvement again

        c, r = random.choice(best_moves)
        apply_move(state, c, r, rows, d1, d2, n)
        energy += best_delta
        steps += 1
    return state, energy, steps


# ---------------------------------------------------------------------------
# Random-restart driver
# ---------------------------------------------------------------------------
def solve_nqueens_hc(
    n: int,
    max_restarts: int = 200,
    time_limit_s: float = 300.0,
    seed: Optional[int] = None,
) -> HCResult:
    if seed is not None:
        random.seed(seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    deadline = t0 + time_limit_s

    restarts = 0
    total_steps = 0
    solution: Optional[List[int]] = None
    timed_out = False

    while restarts < max_restarts:
        if time.perf_counter() > deadline:
            timed_out = True
            break

        start = list(range(n))
        random.shuffle(start)
        final, energy, steps = climb(start, deadline)
        total_steps += steps
        restarts += 1
        if energy == 0:
            solution = final
            break

    if solution is None and time.perf_counter() > deadline:
        timed_out = True

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return HCResult(
        n=n,
        solution=solution,
        elapsed_seconds=elapsed,
        peak_memory_kb=peak / 1024.0,
        restarts=restarts,
        total_steps=total_steps,
        timed_out=timed_out,
    )


def attacking_pairs(state: List[int]) -> int:
    return build_counts(state)[3]


def is_valid_solution(placement: List[int]) -> bool:
    return attacking_pairs(placement) == 0 and len(set(placement)) == len(placement)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Hill Climbing solver for N-Queens.")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[10, 30, 50, 100, 200, 500])
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-run wall-clock cap in seconds (default 300).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"{'N':>5} | {'Time (s)':>10} | {'Peak Mem (KB)':>14} | "
          f"{'Restarts':>9} | {'Steps':>8} | Status")
    print("-" * 78)

    for n in args.sizes:
        result = solve_nqueens_hc(
            n,
            max_restarts=args.restarts,
            time_limit_s=args.timeout,
            seed=args.seed + n,
        )
        if result.timed_out:
            status = f"TIMEOUT (>{args.timeout:.0f}s)"
        elif result.solution and is_valid_solution(result.solution):
            status = "OK"
        else:
            status = "FAIL (stuck)"
        print(f"{n:>5} | {result.elapsed_seconds:>10.4f} | "
              f"{result.peak_memory_kb:>14.2f} | {result.restarts:>9d} | "
              f"{result.total_steps:>8d} | {status}")


if __name__ == "__main__":
    main()
