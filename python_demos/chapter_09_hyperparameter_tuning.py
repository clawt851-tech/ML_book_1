"""
Chapter 9: Hyper-Parameter Tuning with Cross-Validation
========================================================

Demonstrates:
  * Snippet 9.1: GridSearchCV with PurgedKFold
  * Snippet 9.2: MyPipeline subclass that supports sample_weight
  * Snippet 9.3: RandomizedSearchCV with PurgedKFold
  * Snippet 9.4: log-uniform random variable for SVM C/gamma
  * Why use neg_log_loss instead of accuracy
"""

import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, kstest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


# -----------------------------------------------------------------
# Snippet 9.2: MyPipeline class that accepts sample_weight in fit()
# -----------------------------------------------------------------
class MyPipeline(Pipeline):
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + "__sample_weight"] = sample_weight
        return super().fit(X, y, **fit_params)


# -----------------------------------------------------------------
# Snippet 9.4: log-uniform random variable
# -----------------------------------------------------------------
class LogUniformGen(rv_continuous):
    """logUniform(a, b): log(x) ~ U(log(a), log(b))."""

    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)


def log_uniform(a=1.0, b=np.exp(1)):
    return LogUniformGen(a=a, b=b, name="logUniform")


# -----------------------------------------------------------------
# Snippet 9.1: clfHyperFit - grid search with PurgedKFold support
# -----------------------------------------------------------------
def clf_hyper_fit(feat, lbl, t1, pipe_clf, param_grid, cv=3,
                  bagging=(0, None, 1.0), n_jobs=-1, pct_embargo=0.0,
                  scoring=None, **fit_params):
    """
    pipe_clf: a Pipeline ending in a classifier
    param_grid: dict of hyper-params
    bagging: (n_estimators, max_samples, max_features) for outer bagging
    """
    if scoring is None:
        scoring = "f1" if set(lbl.values) == {0, 1} else "neg_log_loss"

    # 1) hyper-parameter search on training data using purged CV
    from sklearn.model_selection import KFold  # for demo we use simple KFold
    inner_cv = KFold(n_splits=cv, shuffle=False)
    gs = GridSearchCV(estimator=pipe_clf, param_grid=param_grid,
                       scoring=scoring, cv=inner_cv, n_jobs=n_jobs)
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_

    # 2) optionally wrap in BaggingClassifier on whole data
    if bagging[1] is not None and bagging[0] > 0:
        from sklearn.ensemble import BaggingClassifier
        gs = BaggingClassifier(estimator=MyPipeline(gs.steps),
                                n_estimators=int(bagging[0]),
                                max_samples=float(bagging[1]),
                                max_features=float(bagging[2]),
                                n_jobs=n_jobs)
        gs = gs.fit(feat, lbl)
        gs = Pipeline([("bag", gs)])
    return gs


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    X, y = make_classification(n_samples=500, n_features=8, random_state=0)
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # log-uniform sampling for C and gamma
    lu = log_uniform(a=1e-2, b=1e2)
    sample = lu.rvs(size=10_000)
    print(f"log-uniform sample: min={sample.min():.4f}, "
          f"max={sample.max():.2f}, median={np.median(sample):.3f}")
    # KS test of log(sample) vs uniform
    ks = kstest(np.log(sample), "uniform",
                  args=(np.log(1e-2), np.log(1e2 / 1e-2)))
    print(f"KS test of log(sample) vs uniform: stat={ks.statistic:.3f}, "
          f"p={ks.pvalue:.3f}")

    # Pipeline grid search demo
    pipe = MyPipeline([("scaler", StandardScaler()),
                        ("svc", SVC(probability=True))])
    grid = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.01, 0.1]}
    gs = GridSearchCV(pipe, grid, cv=3, scoring="neg_log_loss", n_jobs=1)
    gs.fit(X, y)
    print(f"\nBest params: {gs.best_params_}")
    print(f"Best score: {gs.best_score_:.4f}")


if __name__ == "__main__":
    main()
