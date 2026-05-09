"""
Chapter 13: Backtesting on Synthetic Data
==========================================

Demonstrates:
  * Snippet 13.1/13.2: Numerical determination of optimal trading rules
    via Ornstein-Uhlenbeck (O-U) Monte Carlo simulation
  * Heat-map of Sharpe ratio over (profit-take, stop-loss) mesh
"""

import numpy as np
import pandas as pd
from itertools import product


# -----------------------------------------------------------------
# O-U process simulation: P_t = (1 - phi) * E[P_T] + phi * P_{t-1} + sigma * eps
# -----------------------------------------------------------------
def simulate_ou_path(forecast, half_life, sigma, max_hp, pt, sl, n_iter=10_000,
                     seed=0):
    """
    Simulate paths under O-U with given (forecast, half_life, sigma).
    For each path, exit when hitting profit-take pt, stop-loss sl, or max_hp.
    Return list of P&Ls.
    """
    rng = np.random.default_rng(seed)
    phi = 2 ** (-1.0 / half_life)
    pnls = []
    for _ in range(n_iter):
        p = 0.0  # initial price (relative)
        hp = 0
        while True:
            p = (1 - phi) * forecast + phi * p + sigma * rng.standard_normal()
            hp += 1
            if p > pt or p < -sl or hp > max_hp:
                pnls.append(p)
                break
    return np.array(pnls)


# -----------------------------------------------------------------
# Snippet 13.2: batch sweep over (pt, sl) mesh
# -----------------------------------------------------------------
def batch(coeffs, n_iter=1_000, max_hp=100,
           pt_grid=np.linspace(0.5, 10, 20),
           sl_grid=np.linspace(0.5, 10, 20), seed=0):
    out = []
    for pt, sl in product(pt_grid, sl_grid):
        pnls = simulate_ou_path(coeffs["forecast"], coeffs["hl"],
                                  coeffs["sigma"], max_hp, pt, sl,
                                  n_iter=n_iter, seed=seed)
        mean, std = pnls.mean(), pnls.std()
        sr = mean / std if std > 0 else 0.0
        out.append({"pt": pt, "sl": sl, "mean": mean, "std": std, "sr": sr})
    return pd.DataFrame(out)


def heatmap_pivot(df):
    return df.pivot(index="sl", columns="pt", values="sr")


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    coeffs_zero = {"forecast": 0, "hl": 5, "sigma": 1}
    out = batch(coeffs_zero, n_iter=500,
                pt_grid=np.linspace(0.5, 6, 6),
                sl_grid=np.linspace(0.5, 6, 6))
    pivot = heatmap_pivot(out)
    print("Sharpe heatmap (forecast=0, half-life=5, sigma=1):")
    print(pivot.round(2))

    coeffs_pos = {"forecast": 5, "hl": 5, "sigma": 1}
    out2 = batch(coeffs_pos, n_iter=500,
                  pt_grid=np.linspace(0.5, 6, 6),
                  sl_grid=np.linspace(0.5, 6, 6))
    pivot2 = heatmap_pivot(out2)
    print("\nSharpe heatmap (forecast=5, half-life=5, sigma=1):")
    print(pivot2.round(2))
    # With positive long-run equilibrium, optimal pt is small, sl can be wide
    # -- the "rectangular" pattern from the book.


if __name__ == "__main__":
    main()
