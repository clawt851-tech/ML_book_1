"""
Chapter 7: Cross-Validation in Finance
=======================================

Demonstrates:
  * Snippet 7.1: getTrainTimes - purge overlapping train labels
  * Snippet 7.2: getEmbargoTimes - embargo training observations after test
  * Snippet 7.3: PurgedKFold class extending sklearn KFold
  * Snippet 7.4: cvScore wrapper that uses PurgedKFold and sample_weight correctly
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score


# -----------------------------------------------------------------
# Snippet 7.1: getTrainTimes - purge overlapping observations
# -----------------------------------------------------------------
def get_train_times(t1, test_times):
    """
    t1.index = obs start, t1.values = obs end.
    test_times: pandas Series with the times of testing observations.
    Returns the t1 series for the training set after purging.
    """
    trn = t1.copy(deep=True)
    for i, j in test_times.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index    # train start in test
        df1 = trn[(i <= trn) & (trn <= j)].index                 # train end in test
        df2 = trn[(trn.index <= i) & (j <= trn)].index            # train envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


# -----------------------------------------------------------------
# Snippet 7.2: getEmbargoTimes - embargo observations after test
# -----------------------------------------------------------------
def get_embargo_times(times, pct_embargo):
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        return pd.Series(times, index=times)
    mbrg = pd.Series(times[step:], index=times[:-step])
    last = pd.Series([times[-1]] * step, index=times[-step:])
    return pd.concat([mbrg, last])


# -----------------------------------------------------------------
# Snippet 7.3: PurgedKFold class
# -----------------------------------------------------------------
class PurgedKFold(KFold):
    """KFold that purges overlapping training observations and embargos."""

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label Through Dates must be a pandas series")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index")
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_starts = [(i[0], i[-1] + 1) for i in
                       np.array_split(np.arange(X.shape[0]), self.n_splits)]
        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            max_t1_idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())
            train_idx_left = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            train_idx_right = (indices[max_t1_idx + mbrg:]
                               if max_t1_idx + mbrg < X.shape[0] else np.array([], dtype=int))
            train_indices = np.concatenate([train_idx_left, train_idx_right])
            yield train_indices, test_indices


# -----------------------------------------------------------------
# Snippet 7.4: cvScore using PurgedKFold and weights properly
# -----------------------------------------------------------------
def cv_score(clf, X, y, sample_weight=None, scoring="neg_log_loss",
             t1=None, cv=5, cv_gen=None, pct_embargo=0.0):
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError("scoring must be neg_log_loss or accuracy")
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    score = []
    sw = sample_weight if sample_weight is not None else np.ones(len(y))
    for train, test in cv_gen.split(X=X):
        fit = clf.fit(X.iloc[train], y.iloc[train], sample_weight=sw[train])
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test])
            sc = -log_loss(y.iloc[test], prob, sample_weight=sw[test],
                            labels=clf.classes_)
        else:
            pred = fit.predict(X.iloc[test])
            sc = accuracy_score(y.iloc[test], pred, sample_weight=sw[test])
        score.append(sc)
    return np.array(score)


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="1H")
    t1 = pd.Series([idx[min(i + 5, n - 1)] for i in range(n)], index=idx)

    # simulate test set spanning idx[80:120]
    test_times = pd.Series(idx[119:120], index=idx[80:81])
    purged = get_train_times(t1, test_times)
    print(f"Original train length: {len(t1)}, after purge: {len(purged)}")

    # PurgedKFold demo
    X = pd.DataFrame(rng.normal(size=(n, 3)), index=idx)
    y = pd.Series(rng.integers(0, 2, n), index=idx)
    cvgen = PurgedKFold(n_splits=4, t1=t1, pct_embargo=0.01)
    for tr, te in cvgen.split(X, y):
        print(f"  train: {len(tr):4d}, test: {len(te):4d}")


if __name__ == "__main__":
    main()
