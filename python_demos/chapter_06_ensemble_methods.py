"""
Chapter 6: Ensemble Methods
============================

Demonstrates:
  * Bias-variance decomposition
  * Bagging variance reduction (formula)
  * Snippet 6.1: Bagging classifier accuracy via Condorcet jury theorem
  * Snippet 6.2: Three ways of setting up a Random Forest with avgU sampling
  * Random Forest in financial setting (max_samples=avgU, class_weight='balanced')
"""

import numpy as np
import pandas as pd
from math import comb
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


# -----------------------------------------------------------------
# Bias-variance decomposition (illustration)
# -----------------------------------------------------------------
def bagging_variance(sigma_bar=1.0, rho_bar=0.0, N=20):
    """
    V[bagging] = sigma_bar^2 * (rho_bar + (1 - rho_bar) / N)
    rho_bar = average correlation among predictions.
    Bagging only helps when rho_bar < 1.
    """
    return sigma_bar ** 2 * (rho_bar + (1 - rho_bar) / N)


# -----------------------------------------------------------------
# Snippet 6.1: bagging classifier accuracy via majority voting
# -----------------------------------------------------------------
def bagging_accuracy(N=10, p=0.6, k=2):
    """
    P[X > N/k] under Binomial(N, p).
    For sufficiently large N, if p > 1/k, bagging accuracy > p.
    """
    p_ = sum(comb(N, i) * p ** i * (1 - p) ** (N - i)
             for i in range(int(N // k) + 1))
    return 1 - p_  # P[X > N/k]


# -----------------------------------------------------------------
# Snippet 6.2: Three ways of setting up RF with avgU sampling
# -----------------------------------------------------------------
def build_rf_variants(avg_uniqueness=0.5):
    """Three equivalent ways to set up a Random Forest."""
    # (1) Standard RF with balanced subsample
    clf0 = RandomForestClassifier(
        n_estimators=1000, class_weight="balanced_subsample",
        criterion="entropy",
    )

    # (2) Bagging of decision trees with max_samples = avgU
    clf1 = DecisionTreeClassifier(
        criterion="entropy", max_features="sqrt",
        class_weight="balanced",
    )
    bag1 = BaggingClassifier(
        estimator=clf1, n_estimators=1000,
        max_samples=avg_uniqueness,
    )

    # (3) Bagging of single-tree RF with max_samples = avgU
    clf2 = RandomForestClassifier(
        n_estimators=1, criterion="entropy",
        bootstrap=False, class_weight="balanced_subsample",
    )
    bag2 = BaggingClassifier(
        estimator=clf2, n_estimators=1000,
        max_samples=avg_uniqueness, max_features=1.0,
    )
    return clf0, bag1, bag2


# -----------------------------------------------------------------
# Demo: bagging vs single classifier
# -----------------------------------------------------------------
def main():
    print("=== Bagging variance under different correlations ===")
    for rho in [0.0, 0.2, 0.5, 0.9]:
        v = bagging_variance(sigma_bar=1.0, rho_bar=rho, N=30)
        print(f"rho={rho:.2f} -> V[bagged] = {v:.3f}")
    print()

    print("=== Condorcet-style bagging accuracy ===")
    for p in [0.45, 0.51, 0.6, 0.7]:
        acc = bagging_accuracy(N=100, p=p, k=2)
        print(f"single classifier p={p:.2f} -> bagging (N=100, k=2) = {acc:.3f}")
    print()

    # quick demo on synthetic data
    X, y = make_classification(n_samples=500, n_features=10, random_state=0)
    rf, bag1, bag2 = build_rf_variants(avg_uniqueness=0.5)
    rf.fit(X, y)
    print(f"RF train acc: {rf.score(X, y):.3f}")


if __name__ == "__main__":
    main()
