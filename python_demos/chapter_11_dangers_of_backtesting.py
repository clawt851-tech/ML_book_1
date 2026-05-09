"""
Chapter 11: The Dangers of Backtesting
=======================================

Demonstrates:
  * The Seven Sins of Quantitative Investing
  * Combinatorially Symmetric Cross-Validation (CSCV) - Probability of Backtest Overfitting (PBO)
  * Marcos' Second Law: Don't research while backtesting
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm


# -----------------------------------------------------------------
# CSCV: Combinatorially Symmetric Cross-Validation
# Estimates Probability of Backtest Overfitting (PBO)
# -----------------------------------------------------------------
def cscv_pbo(M, S=16):
    """
    M: T x N matrix of PnL series (T observations, N strategy configs).
    S: even number of disjoint submatrices to form combinations from.
    Returns: PBO (probability backtest is overfit), and rank logits.
    """
    T, N = M.shape
    # Trim T to multiple of S
    T_use = T - (T % S)
    M = M.iloc[:T_use] if hasattr(M, "iloc") else M[:T_use]
    Ms = np.array_split(np.arange(T_use), S)

    # All combinations of size S/2 from S submatrices
    combos = list(combinations(range(S), S // 2))
    logits = []
    for c in combos:
        train_rows = np.concatenate([Ms[i] for i in c])
        test_rows = np.concatenate([Ms[i] for i in range(S) if i not in c])
        M_train = M[train_rows] if isinstance(M, np.ndarray) else M.iloc[train_rows]
        M_test = M[test_rows] if isinstance(M, np.ndarray) else M.iloc[test_rows]
        # Sharpe ratio of each strategy in train and test
        sr_train = sharpe_ratio_array(M_train)
        sr_test = sharpe_ratio_array(M_test)
        n_star = np.argmax(sr_train)
        # rank of n_star in test set
        order = np.argsort(np.argsort(sr_test))  # ranks ascending
        rank_in_test = order[n_star] + 1
        omega = rank_in_test / (N + 1)
        if 0 < omega < 1:
            logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    pbo = (logits < 0).mean()
    return pbo, logits


def sharpe_ratio_array(R):
    """Per-column Sharpe ratio."""
    R = np.asarray(R)
    return R.mean(axis=0) / (R.std(axis=0) + 1e-12) * np.sqrt(R.shape[0])


# -----------------------------------------------------------------
# Seven Sins of Quantitative Investing (Luo et al. 2014)
# -----------------------------------------------------------------
SEVEN_SINS = [
    "Survivorship bias",
    "Look-ahead bias",
    "Storytelling",
    "Data mining and data snooping",
    "Transaction costs",
    "Outliers",
    "Shorting",
]

GENERAL_RECOMMENDATIONS = [
    "Develop models for entire asset classes, not single securities",
    "Apply bagging to reduce variance and overfitting",
    "Do not backtest until research is complete",
    "Record every backtest to estimate PBO and deflate Sharpe",
    "Simulate scenarios rather than just history (Ch.13)",
    "If backtest fails, start from scratch — do not reuse results",
]


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    print("=== Seven Sins of Quantitative Investing ===")
    for i, sin in enumerate(SEVEN_SINS, 1):
        print(f"  {i}. {sin}")

    print("\n=== General Recommendations ===")
    for r in GENERAL_RECOMMENDATIONS:
        print(f"  - {r}")

    # CSCV demo with random PnL
    rng = np.random.default_rng(0)
    T, N = 1000, 200
    M = rng.normal(0, 1, size=(T, N))
    pbo, logits = cscv_pbo(M, S=8)
    print(f"\nCSCV demo on random data: PBO = {pbo:.2%} (expect ~0.5)")
    print(f"Mean rank-logit: {logits.mean():.3f}")


if __name__ == "__main__":
    main()
