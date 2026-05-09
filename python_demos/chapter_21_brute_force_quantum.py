"""
Chapter 21: Brute Force and Quantum Computers
==============================================

Demonstrates:
  * Combinatorial portfolio optimization (integer pigeonhole partitions)
  * Brute-force solver suitable for D-Wave style annealers
  * Static and dynamic feasible solutions
"""

import numpy as np
import pandas as pd
from itertools import product


# -----------------------------------------------------------------
# 21.5.1: Pigeonhole partitions of K balls into N pigeon-holes
# -----------------------------------------------------------------
def pigeonhole_partitions(n_assets, n_units):
    """All non-negative integer vectors of length n_assets summing to n_units."""
    if n_assets == 1:
        yield (n_units,)
        return
    for i in range(n_units + 1):
        for sub in pigeonhole_partitions(n_assets - 1, n_units - i):
            yield (i,) + sub


# -----------------------------------------------------------------
# 21.5.2: Feasible static solutions
# -----------------------------------------------------------------
def evaluate_portfolio(weights, mu, cov):
    """Sharpe ratio of static portfolio (mu/sigma, ignore risk-free rate)."""
    w = np.asarray(weights) / max(np.sum(np.abs(weights)), 1)
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    return ret / np.sqrt(var) if var > 0 else 0


# -----------------------------------------------------------------
# 21.5.3: Brute-force evaluation of trajectories
# -----------------------------------------------------------------
def brute_force_static(mu, cov, total_units=10):
    """Find best static integer allocation by exhaustive search."""
    n = len(mu)
    best, best_sr = None, -np.inf
    for w in pigeonhole_partitions(n, total_units):
        sr = evaluate_portfolio(np.array(w), mu, cov)
        if sr > best_sr:
            best, best_sr = w, sr
    return np.array(best), best_sr


# -----------------------------------------------------------------
# Dynamic solution: time-varying integer allocation across T periods
# -----------------------------------------------------------------
def brute_force_dynamic(mu_seq, cov_seq, total_units=5, max_combos=10_000):
    """
    Find best sequence of integer allocations across T periods.
    Limits to first max_combos combinations to keep demo tractable.
    """
    n = len(mu_seq[0])
    static_options = list(pigeonhole_partitions(n, total_units))
    static_options = static_options[:max_combos] if len(static_options) > max_combos \
                     else static_options
    best_sr_seq = []
    best_alloc_seq = []
    for mu, cov in zip(mu_seq, cov_seq):
        best, sr = -np.inf, None
        for w in static_options:
            sr_ = evaluate_portfolio(np.array(w), mu, cov)
            if sr_ > best:
                best, sr = sr_, w
        best_sr_seq.append(best)
        best_alloc_seq.append(sr)
    return best_alloc_seq, best_sr_seq


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    # Synthetic 4-asset problem
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.01, 0.05, 4)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T * 0.001

    # Static brute force
    w_best, sr_best = brute_force_static(mu, cov, total_units=10)
    print(f"Static: best weights={w_best}, SR={sr_best:.3f}")

    # Number of combinations
    n = 4
    total = 10
    print(f"Pigeonhole partitions for {n} assets, {total} units: "
          f"{sum(1 for _ in pigeonhole_partitions(n, total))} combinations")


if __name__ == "__main__":
    main()
