"""
N-Queens — Genetic Algorithm
============================

Chromosome
----------
Each chromosome is a *permutation* of {0, 1, ..., N-1} where the value at
index c is the row of the queen in column c.  Encoding the board as a
permutation guarantees that no two queens ever share a row or a column, so
the only remaining conflicts are the diagonal ones.  This shrinks the search
space dramatically compared to a free encoding.

Fitness
-------
Let H = number of attacking diagonal pairs.
    fitness = (N choose 2) − H
A perfect solution therefore has fitness = N*(N-1)/2 and H = 0.

Operators
---------
* Selection : binary *tournament* selection (k=3).  Tournaments scale well to
              large populations and require no fitness normalisation.
* Crossover : *Order-1 crossover (OX1)*.  OX1 preserves the permutation
              property of both parents — essential here because rows must
              remain unique in every child.
* Mutation  : *swap mutation* — pick two indices uniformly and swap their
              values.  Probability per child is `mutation_rate`.
* Elitism   : the top E members of the previous generation always survive
              into the next, guaranteeing monotonically non-decreasing best
              fitness.

Hyper-parameters
----------------
Population size, generation cap, mutation rate and tournament size all
scale gently with N so the same code can tackle N = 10 and N = 500.

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
class GAResult:
    n: int
    solution: Optional[List[int]]
    elapsed_seconds: float
    peak_memory_kb: float
    generations: int
    best_conflicts: int
    timed_out: bool


# ---------------------------------------------------------------------------
# Fitness: count *diagonal* conflicts only (rows/cols are guaranteed clean
# by the permutation encoding).  We use list-based counters indexed by
# (r-c)+N and (r+c) which is far faster than Python dicts when N is large.
# ---------------------------------------------------------------------------
def diagonal_conflicts(chrom: List[int]) -> int:
    n = len(chrom)
    size = 2 * n
    d1 = [0] * size                # offset by +n so the index is non-negative
    d2 = [0] * size
    for c, r in enumerate(chrom):
        d1[r - c + n] += 1
        d2[r + c] += 1
    pairs = 0
    for arr in (d1, d2):
        for v in arr:
            if v > 1:
                pairs += v * (v - 1) // 2
    return pairs


def fitness(chrom: List[int], max_pairs: int) -> int:
    return max_pairs - diagonal_conflicts(chrom)


# ---------------------------------------------------------------------------
# Selection / crossover / mutation
# ---------------------------------------------------------------------------
def tournament_select(
    population: List[List[int]],
    fitnesses: List[int],
    k: int = 3,
) -> List[int]:
    contenders = random.sample(range(len(population)), k)
    best = max(contenders, key=lambda i: fitnesses[i])
    return population[best]


def order_one_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    """Classic OX1 crossover for permutations."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    return _ox_child(p1, p2, a, b), _ox_child(p2, p1, a, b)


def _ox_child(parent_a: List[int], parent_b: List[int], a: int, b: int) -> List[int]:
    """Order-1 crossover, implemented in O(N) (no `in child` scan)."""
    n = len(parent_a)
    child: List[int] = [-1] * n
    in_child = [False] * n
    # 1. Copy slice [a:b+1] from parent_a
    for i in range(a, b + 1):
        v = parent_a[i]
        child[i] = v
        in_child[v] = True
    # 2. Fill the rest with parent_b's order, starting just after b.
    write_pos = (b + 1) % n
    filled = b - a + 1
    read_pos = (b + 1) % n
    while filled < n:
        candidate = parent_b[read_pos]
        if not in_child[candidate]:
            child[write_pos] = candidate
            in_child[candidate] = True
            write_pos = (write_pos + 1) % n
            filled += 1
        read_pos = (read_pos + 1) % n
    return child


def swap_mutate(chrom: List[int]) -> None:
    i, j = random.sample(range(len(chrom)), 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]


def local_polish(chrom: List[int], max_pairs: int,
                 max_passes: int = 3) -> List[int]:
    """
    Memetic helper: bounded pairwise-swap local search.

    Strategy:
      * Make repeated passes over a *shuffled* list of (i, j) pairs.
      * In every pass we accept any swap that strictly improves h.
      * After a strictly-improving pass we shuffle and try again.
      * If a full pass finds no improvement we make one final pass that
        also accepts neutral swaps; this often unblocks the h = 1 / h = 2
        plateau where vanilla swap-mutation rarely lands the right pair.

    Total work is bounded by `max_passes` × N² fitness evaluations, so the
    cost is predictable on big boards.
    """
    n = len(chrom)
    best = chrom[:]
    best_h = max_pairs - fitness(best, max_pairs)
    if best_h == 0:
        return best
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for attempt in range(max_passes):
        random.shuffle(pairs)
        improved = False
        for i, j in pairs:
            best[i], best[j] = best[j], best[i]
            h = max_pairs - fitness(best, max_pairs)
            if h < best_h:
                best_h = h
                improved = True
                if best_h == 0:
                    return best
            else:
                best[i], best[j] = best[j], best[i]
        if not improved:
            # one neutral-allowing pass to escape a plateau
            random.shuffle(pairs)
            sideways = 0
            for i, j in pairs:
                best[i], best[j] = best[j], best[i]
                h = max_pairs - fitness(best, max_pairs)
                if h < best_h:
                    best_h = h
                    if best_h == 0:
                        return best
                elif h == best_h and sideways < max(2, n // 10):
                    sideways += 1
                else:
                    best[i], best[j] = best[j], best[i]
            break
    return best


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------
def solve_nqueens_ga(
    n: int,
    population_size: Optional[int] = None,
    generations: Optional[int] = None,
    mutation_rate: float = 0.3,
    elitism: int = 2,
    tournament_k: int = 3,
    time_limit_s: float = 300.0,
    seed: Optional[int] = None,
) -> GAResult:
    if seed is not None:
        random.seed(seed)

    # Adaptive defaults — modest populations with strong local-search
    # ("memetic") polish converge faster on N-Queens than huge populations.
    if population_size is None:
        population_size = max(60, min(200, 4 * n))
    if generations is None:
        generations = max(2000, 50 * n)

    max_pairs = n * (n - 1) // 2

    tracemalloc.start()
    t0 = time.perf_counter()
    deadline = t0 + time_limit_s

    # Initial population: random permutations.
    population: List[List[int]] = []
    for _ in range(population_size):
        chrom = list(range(n))
        random.shuffle(chrom)
        population.append(chrom)
    fitnesses = [fitness(c, max_pairs) for c in population]

    best_idx = max(range(population_size), key=lambda i: fitnesses[i])
    best_chrom = population[best_idx][:]
    best_fit = fitnesses[best_idx]

    gen = 0
    timed_out = False
    stagnation = 0                              # generations without progress
    current_mutation = mutation_rate

    while gen < generations and best_fit < max_pairs:
        if time.perf_counter() > deadline:
            timed_out = True
            break

        # Adaptive mutation: when the best fitness hasn't improved for a
        # while, ramp the mutation rate up to inject diversity.  Reset to
        # the baseline as soon as we make progress again.
        if stagnation > 0 and stagnation % 50 == 0:
            current_mutation = min(0.9, current_mutation * 1.5)

        # Elitism — keep the top `elitism` chromosomes (and their cached
        # fitness) unchanged.
        sorted_idx = sorted(range(population_size),
                            key=lambda i: fitnesses[i], reverse=True)
        new_pop: List[List[int]] = [population[i][:] for i in sorted_idx[:elitism]]
        new_fits: List[int] = [fitnesses[i] for i in sorted_idx[:elitism]]

        # Fill the rest with offspring; compute fitness only for new ones.
        while len(new_pop) < population_size:
            parent_a = tournament_select(population, fitnesses, k=tournament_k)
            parent_b = tournament_select(population, fitnesses, k=tournament_k)
            child_a, child_b = order_one_crossover(parent_a, parent_b)
            if random.random() < current_mutation:
                swap_mutate(child_a)
            if random.random() < current_mutation:
                swap_mutate(child_b)
            new_pop.append(child_a)
            new_fits.append(fitness(child_a, max_pairs))
            if len(new_pop) < population_size:
                new_pop.append(child_b)
                new_fits.append(fitness(child_b, max_pairs))

        population = new_pop
        fitnesses = new_fits

        gen_best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] > best_fit:
            best_fit = fitnesses[gen_best_idx]
            best_chrom = population[gen_best_idx][:]
            stagnation = 0
            current_mutation = mutation_rate
        else:
            stagnation += 1

        # Memetic polish: when we stagnate, run a deterministic pairwise-
        # swap local search on the best chromosome.  This is what gets the
        # GA off the low-conflict plateau that vanilla swap-mutation rarely
        # escapes.  Triggered as soon as we have not improved for 20 gens.
        if stagnation >= 20 and (max_pairs - best_fit) > 0:
            polished = local_polish(best_chrom, max_pairs)
            polished_fit = fitness(polished, max_pairs)
            if polished_fit > best_fit:
                best_fit = polished_fit
                best_chrom = polished
                population[0] = polished        # inject back into population
                fitnesses[0] = polished_fit
                stagnation = 0

        # Severe stagnation → reseed half the population with fresh randoms.
        if stagnation >= 200:
            for i in range(elitism, population_size // 2):
                random.shuffle(population[i])
            fitnesses = [fitness(c, max_pairs) for c in population]
            stagnation = 0
            current_mutation = mutation_rate

        gen += 1

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    best_conflicts = max_pairs - best_fit
    solution = best_chrom if best_conflicts == 0 else None

    return GAResult(
        n=n,
        solution=solution,
        elapsed_seconds=elapsed,
        peak_memory_kb=peak / 1024.0,
        generations=gen,
        best_conflicts=best_conflicts,
        timed_out=timed_out,
    )


def is_valid_solution(placement: List[int]) -> bool:
    n = len(placement)
    return (
        len(set(placement)) == n
        and all(0 <= r < n for r in placement)
        and diagonal_conflicts(placement) == 0
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Genetic Algorithm solver for N-Queens.")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[10, 30, 50, 100, 200, 500])
    parser.add_argument("--mutation", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-run wall-clock cap in seconds (default 300).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"{'N':>5} | {'Time (s)':>10} | {'Peak Mem (KB)':>14} | "
          f"{'Gens':>8} | {'Conflicts':>9} | Status")
    print("-" * 76)

    for n in args.sizes:
        result = solve_nqueens_ga(
            n,
            mutation_rate=args.mutation,
            time_limit_s=args.timeout,
            seed=args.seed + n,
        )
        if result.timed_out:
            status = f"TIMEOUT (>{args.timeout:.0f}s)"
        elif result.solution and is_valid_solution(result.solution):
            status = "OK"
        else:
            status = f"FAIL (h={result.best_conflicts})"
        print(f"{n:>5} | {result.elapsed_seconds:>10.4f} | "
              f"{result.peak_memory_kb:>14.2f} | {result.generations:>8d} | "
              f"{result.best_conflicts:>9d} | {status}")


if __name__ == "__main__":
    main()
