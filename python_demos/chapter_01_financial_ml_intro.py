"""
Chapter 1: Financial Machine Learning as a Distinct Subject
============================================================

This chapter introduces why financial ML differs from standard ML.
Demonstrates: meta-strategy paradigm, the production chain, common pitfalls.

Author: Demonstration code based on "Advances in Financial Machine Learning"
        by Marcos Lopez de Prado (2018)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------
# 1. Demonstration: Sisyphus paradigm vs Meta-strategy paradigm
# -----------------------------------------------------------------
def sisyphus_research(n_quants=50, n_strategies_per_quant=1, p_true_alpha=0.02):
    """Each quant works in silo. Most produce false positives."""
    rng = np.random.default_rng(42)
    results = []
    for q in range(n_quants):
        # Each quant tries multiple things; reports best
        trials = rng.normal(0, 1, size=200)
        best_sharpe = trials.max()  # selection bias
        is_true_alpha = rng.random() < p_true_alpha
        results.append({
            'quant_id': q,
            'reported_sharpe': best_sharpe,
            'true_alpha': is_true_alpha
        })
    return pd.DataFrame(results)


def meta_strategy_research(n_specialists=10, n_collaborations=50):
    """Specialists collaborate on shared infrastructure -> higher hit rate."""
    rng = np.random.default_rng(42)
    results = []
    for s in range(n_collaborations):
        # Each strategy uses pooled feature library, reviewed by team
        sharpe = rng.normal(0.8, 0.5)  # higher mean due to better process
        results.append({'strategy_id': s, 'sharpe': sharpe})
    return pd.DataFrame(results)


# -----------------------------------------------------------------
# 2. Demonstration: Common Pitfalls table
# -----------------------------------------------------------------
COMMON_PITFALLS = pd.DataFrame([
    {"Category": "Epistemological", "Pitfall": "The Sisyphus paradigm",
     "Solution": "The meta-strategy paradigm", "Chapter": 1},
    {"Category": "Epistemological", "Pitfall": "Research through backtesting",
     "Solution": "Feature importance analysis", "Chapter": 8},
    {"Category": "Data processing", "Pitfall": "Chronological sampling",
     "Solution": "The volume clock", "Chapter": 2},
    {"Category": "Data processing", "Pitfall": "Integer differentiation",
     "Solution": "Fractional differentiation", "Chapter": 5},
    {"Category": "Classification", "Pitfall": "Fixed-time horizon labeling",
     "Solution": "The triple-barrier method", "Chapter": 3},
    {"Category": "Classification", "Pitfall": "Learning side and size simultaneously",
     "Solution": "Meta-labeling", "Chapter": 3},
    {"Category": "Classification", "Pitfall": "Weighting of non-IID samples",
     "Solution": "Uniqueness weighting; sequential bootstrapping", "Chapter": 4},
    {"Category": "Evaluation", "Pitfall": "Cross-validation leakage",
     "Solution": "Purging and embargoing", "Chapter": 7},
    {"Category": "Evaluation", "Pitfall": "Walk-forward (historical) backtesting",
     "Solution": "Combinatorial purged cross-validation", "Chapter": 12},
    {"Category": "Evaluation", "Pitfall": "Backtest overfitting",
     "Solution": "Backtesting on synthetic data; deflated Sharpe ratio",
     "Chapter": "10-16"},
])


# -----------------------------------------------------------------
# 3. Production chain blueprint (assembly-line analogy)
# -----------------------------------------------------------------
PRODUCTION_CHAIN = [
    "Data Curators",       # Ch.2  -> raw data acquisition / cleaning
    "Feature Analysts",    # Ch.2-9, 17-19 -> informative signals
    "Strategists",         # Ch.10, 16  -> bet sizing, portfolio
    "Backtesters",         # Ch.11-16  -> evaluate / overfitting
    "Deployment Team",     # Ch.20-22  -> HPC implementation
    "Portfolio Oversight"  # Embargo / Paper-trade / Graduate
]


def main():
    print("=== Chapter 1: Financial ML as a Distinct Subject ===\n")
    print("Common Pitfalls:")
    print(COMMON_PITFALLS.to_string(index=False))

    print("\nProduction chain stages:")
    for i, station in enumerate(PRODUCTION_CHAIN, 1):
        print(f"  {i}. {station}")

    sisyphus = sisyphus_research()
    meta = meta_strategy_research()
    print(f"\nSisyphus paradigm: mean reported Sharpe = "
          f"{sisyphus['reported_sharpe'].mean():.2f} "
          f"(true alpha rate ~{sisyphus['true_alpha'].mean():.1%})")
    print(f"Meta-strategy paradigm: mean Sharpe = "
          f"{meta['sharpe'].mean():.2f}")


if __name__ == "__main__":
    main()
