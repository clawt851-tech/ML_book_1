"""
Chapter 12: Backtesting through Cross-Validation
=================================================

Demonstrates:
  * Walk-forward backtesting
  * Standard k-fold CV-based backtesting (not recommended alone)
  * Combinatorial Purged Cross-Validation (CPCV) - the recommended approach
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------------------------------
# Walk-Forward Method
# -----------------------------------------------------------------
def walk_forward(model, X, y, n_splits=5):
    """Train on past, test on next chunk; repeat moving forward."""
    n = len(X)
    chunk = n // (n_splits + 1)
    out = []
    for k in range(n_splits):
        train_end = chunk * (k + 1)
        test_end = train_end + chunk
        m = model.fit(X.iloc[:train_end], y.iloc[:train_end])
        score = m.score(X.iloc[train_end:test_end], y.iloc[train_end:test_end])
        out.append({"fold": k, "train_end": train_end,
                    "test_end": test_end, "score": score})
    return pd.DataFrame(out)


# -----------------------------------------------------------------
# Combinatorial Purged Cross-Validation (CPCV)
# -----------------------------------------------------------------
class CombinatorialPurgedKFold:
    """
    Yields all C(N, k) train/test splits where k groups are test
    and N-k are train. Standard choice: N=6, k=2 -> 15 splits.
    Purges training labels overlapping any test group.
    """

    def __init__(self, n_groups=6, n_test_groups=2, t1=None, pct_embargo=0.0):
        self.N, self.k = n_groups, n_test_groups
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X):
        n = len(X)
        groups = np.array_split(np.arange(n), self.N)
        for test_idx in combinations(range(self.N), self.k):
            test = np.concatenate([groups[i] for i in test_idx])
            train = np.concatenate([groups[i] for i in range(self.N)
                                      if i not in test_idx])
            # Purge train indices overlapping test (if t1 provided)
            if self.t1 is not None:
                test_times = self.t1.iloc[test]
                test_start, test_end = test_times.index.min(), test_times.max()
                purge_mask = (
                    (self.t1.iloc[train].index <= test_end)
                    & (self.t1.iloc[train] >= test_start)
                )
                train = train[~purge_mask.values]
                # Embargo
                emb = int(n * self.pct_embargo)
                test_max_idx = test.max()
                train = train[(train < test.min()) |
                              (train >= test_max_idx + emb)]
            yield train, test

    def num_paths(self):
        """Number of unique backtest paths produced by the splits."""
        from math import comb
        return comb(self.N, self.k) * self.k // self.N


# -----------------------------------------------------------------
# Backtest paths from CPCV
# -----------------------------------------------------------------
def cpcv_backtest_paths(model, X, y, n_groups=6, n_test_groups=2):
    """
    Run CPCV and assemble the OOS predictions into multiple
    full-period backtest "paths".
    """
    cv = CombinatorialPurgedKFold(n_groups=n_groups, n_test_groups=n_test_groups)
    n = len(X)
    n_paths = cv.num_paths()
    paths = np.full((n, n_paths), np.nan)
    path_counter = {i: 0 for i in range(n_groups)}
    for train, test in cv.split(X):
        m = model.fit(X.iloc[train], y.iloc[train])
        preds = m.predict(X.iloc[test])
        # Distribute predictions to next available path slot for each group
        for i in test:
            for p in range(n_paths):
                if np.isnan(paths[i, p]):
                    paths[i, p] = preds[np.where(test == i)[0][0]]
                    break
    return paths


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=list("abcde"))
    y = pd.Series((X.sum(axis=1) > 0).astype(int))

    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    print("=== Walk-Forward ===")
    wf = walk_forward(rf, X, y, n_splits=5)
    print(wf)

    print("\n=== CPCV (N=6, k=2) ===")
    cv = CombinatorialPurgedKFold(n_groups=6, n_test_groups=2)
    n_splits, n_paths = 0, cv.num_paths()
    for tr, te in cv.split(X):
        n_splits += 1
    print(f"#splits = {n_splits}, #backtest paths = {n_paths}")


if __name__ == "__main__":
    main()
