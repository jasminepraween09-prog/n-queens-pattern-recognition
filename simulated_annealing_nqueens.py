"""
N-Queens — Simulated Annealing
==============================

State and cost
--------------
Same representation as the Hill-Climbing solver:
    state[c] = row of the queen in column c.
The energy E(state) = number of attacking queen-pairs (lower is better).

Algorithm
---------
1. Start from a random permutation of rows.
2. At every iteration, propose a neighbour by picking a random column and
   moving its queen to a different random row.
3. Compute ΔE = E(new) − E(current).
4. If ΔE < 0, accept the move (it is a clear improvement).
   If ΔE ≥ 0, accept with probability exp(−ΔE / T).
5. Cool: T ← T × α (geometric cooling).
6. Stop on E = 0, on time-out, or when T falls below `T_min`.

Why simulated annealing helps
-----------------------------
Pure greedy hill climbing gets stuck on plateaus and shoulders.  SA lets the
search take *uphill* moves with a probability that shrinks over time, giving
it a chance to escape local minima early on while behaving like greedy
descent in the final, low-temperature phase.

Hyper-parameters (T0, α, T_min, iters per temperature) were chosen
empirically to handle the entire requested range (N = 10 … 500) without
needing per-size tuning.

Author : Jasmine Praween
Course : Pattern Recognition — Assignment (N-Queens)
"""

from __future__ import annotations

import argparse
import math
import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SAResult:
    n: int
    solution: Optional[List[int]]
    elapsed_seconds: float
    peak_memory_kb: float
    iterations: int
    final_temperature: float
    timed_out: bool


# ---------------------------------------------------------------------------
# Incremental cost — much faster than recomputing the full pair-count.
# ---------------------------------------------------------------------------
def initial_conflict_count(state: List[int]) -> int:
    n = len(state)
    rows: dict[int, int] = {}
    d1: dict[int, int] = {}
    d2: dict[int, int] = {}
    for c, r in enumerate(state):
        rows[r] = rows.get(r, 0) + 1
        d1[r - c] = d1.get(r - c, 0) + 1
        d2[r + c] = d2.get(r + c, 0) + 1
    pairs = 0
    for d in (rows, d1, d2):
        for v in d.values():
            if v > 1:
                pairs += v * (v - 1) // 2
    return pairs


def delta_for_move(
    state: List[int],
    col: int,
    new_row: int,
    row_count: dict[int, int],
    d1_count: dict[int, int],
    d2_count: dict[int, int],
) -> int:
    """
    Return ΔE if we move the queen in column `col` from its current row to
    `new_row`.  ΔE is computed in O(1) using the pre-computed line counts.
    """
    old_row = state[col]
    if old_row == new_row:
        return 0
    # Removing the old queen
    delta = 0
    delta -= row_count[old_row] - 1
    delta -= d1_count[old_row - col] - 1
    delta -= d2_count[old_row + col] - 1
    # Adding the new queen
    delta += row_count.get(new_row, 0)
    delta += d1_count.get(new_row - col, 0)
    delta += d2_count.get(new_row + col, 0)
    return delta


def apply_move(
    state: List[int],
    col: int,
    new_row: int,
    row_count: dict[int, int],
    d1_count: dict[int, int],
    d2_count: dict[int, int],
) -> None:
    old_row = state[col]
    # remove old
    row_count[old_row] -= 1
    d1_count[old_row - col] -= 1
    d2_count[old_row + col] -= 1
    # add new
    row_count[new_row] = row_count.get(new_row, 0) + 1
    d1_count[new_row - col] = d1_count.get(new_row - col, 0) + 1
    d2_count[new_row + col] = d2_count.get(new_row + col, 0) + 1
    state[col] = new_row


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def solve_nqueens_sa(
    n: int,
    T0: Optional[float] = None,
    alpha: Optional[float] = None,
    T_min: float = 1e-6,
    max_iters: Optional[int] = None,
    time_limit_s: float = 300.0,
    seed: Optional[int] = None,
) -> SAResult:
    if seed is not None:
        random.seed(seed)

    # All three SA hyper-parameters scale with N so the same code works
    # comfortably for N = 10 up to N = 500:
    #   * T0   ∝ N        — early acceptance probability stays meaningful.
    #   * α    → 1        — cooling slows down on bigger boards.
    #   * iters cap ∝ N²  — gives the chain enough time to mix.
    if T0 is None:
        T0 = max(4.0, float(n))
    if alpha is None:
        alpha = 1.0 - 1.0 / max(5000, 500 * n)
    if max_iters is None:
        max_iters = max(500_000, n * n * 500)

    tracemalloc.start()
    t0 = time.perf_counter()
    deadline = t0 + time_limit_s

    state = list(range(n))
    random.shuffle(state)

    row_count: dict[int, int] = {}
    d1_count: dict[int, int] = {}
    d2_count: dict[int, int] = {}
    for c, r in enumerate(state):
        row_count[r] = row_count.get(r, 0) + 1
        d1_count[r - c] = d1_count.get(r - c, 0) + 1
        d2_count[r + c] = d2_count.get(r + c, 0) + 1

    energy = initial_conflict_count(state)
    best_state = state[:]
    best_energy = energy

    T = T0
    iters = 0
    timed_out = False
    iters_since_best = 0
    reheat_count = 0
    reheat_after = max(5000, n * 200)         # iterations with no improvement

    while energy > 0 and iters < max_iters and T > T_min:
        if time.perf_counter() > deadline:
            timed_out = True
            break

        col = random.randrange(n)
        new_row = random.randrange(n)
        while new_row == state[col]:
            new_row = random.randrange(n)

        d = delta_for_move(state, col, new_row, row_count, d1_count, d2_count)
        if d < 0 or random.random() < math.exp(-d / max(T, 1e-12)):
            apply_move(state, col, new_row, row_count, d1_count, d2_count)
            energy += d
            if energy < best_energy:
                best_energy = energy
                best_state = state[:]
                iters_since_best = 0
            else:
                iters_since_best += 1
        else:
            iters_since_best += 1
        T *= alpha
        iters += 1

        # Reheat: if we have not improved for `reheat_after` iterations,
        # bump the temperature back up.  This is a classic SA escape trick
        # that revives the search when it freezes in a non-optimal basin.
        if iters_since_best >= reheat_after and reheat_count < 5:
            T = max(T, T0 * 0.5)
            iters_since_best = 0
            reheat_count += 1

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    solution = best_state if best_energy == 0 else None
    return SAResult(
        n=n,
        solution=solution,
        elapsed_seconds=elapsed,
        peak_memory_kb=peak / 1024.0,
        iterations=iters,
        final_temperature=T,
        timed_out=timed_out,
    )


def is_valid_solution(placement: List[int]) -> bool:
    return initial_conflict_count(placement) == 0 and len(set(placement)) == len(placement)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated Annealing solver for N-Queens.")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[10, 30, 50, 100, 200, 500])
    parser.add_argument("--alpha", type=float, default=None,
                        help="Cooling factor per iteration. "
                             "Default: auto-scale based on N "
                             "(α = 1 − 1/max(2000, 100·N)).")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-run wall-clock cap in seconds (default 300).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"{'N':>5} | {'Time (s)':>10} | {'Peak Mem (KB)':>14} | "
          f"{'Iters':>10} | {'Final T':>10} | Status")
    print("-" * 80)

    for n in args.sizes:
        result = solve_nqueens_sa(
            n,
            alpha=args.alpha,
            time_limit_s=args.timeout,
            seed=args.seed + n,
        )
        if result.timed_out:
            status = f"TIMEOUT (>{args.timeout:.0f}s)"
        elif result.solution and is_valid_solution(result.solution):
            status = "OK"
        else:
            status = "FAIL"
        print(f"{n:>5} | {result.elapsed_seconds:>10.4f} | "
              f"{result.peak_memory_kb:>14.2f} | {result.iterations:>10d} | "
              f"{result.final_temperature:>10.4g} | {status}")


if __name__ == "__main__":
    main()
