"""
Chapter 8: Feature Importance
==============================

Demonstrates:
  * Snippet 8.2: Mean Decrease Impurity (MDI)  -- in-sample, tree-based
  * Snippet 8.3: Mean Decrease Accuracy (MDA)  -- out-of-sample, model-agnostic
  * Snippet 8.4: Single Feature Importance (SFI) -- no substitution effects
  * Snippet 8.5: Orthogonal features via PCA
  * Snippet 8.6: Weighted Kendall's tau between feature importance and PCA rank
"""

import numpy as np
import pandas as pd
from scipy.stats import weightedtau
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_classification


# -----------------------------------------------------------------
# Snippet 8.2: MDI feature importance
# -----------------------------------------------------------------
def feat_imp_mdi(fit, feat_names):
    """In-sample importance from impurity decrease, averaged over trees."""
    df0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient="index")
    df0.columns = feat_names
    df0 = df0.replace(0, np.nan)        # because max_features=1, zero means not chosen
    imp = pd.concat({"mean": df0.mean(),
                     "std": df0.std() * df0.shape[0] ** -0.5}, axis=1)
    imp /= imp["mean"].sum()
    return imp


# -----------------------------------------------------------------
# Snippet 8.3: MDA feature importance (permutation)
# -----------------------------------------------------------------
def feat_imp_mda(clf, X, y, sample_weight=None, cv=5,
                  scoring="neg_log_loss", t1=None, pct_embargo=0.0):
    sw = sample_weight if sample_weight is not None else np.ones(len(y))
    cvg = KFold(n_splits=cv, shuffle=False)
    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvg.split(X)):
        X0, y0, w0 = X.iloc[train], y.iloc[train], sw[train]
        X1, y1, w1 = X.iloc[test], y.iloc[test], sw[test]
        fit = clf.fit(X0, y0, sample_weight=w0)
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1, labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1)
        for j in X.columns:
            X1_ = X1.copy(deep=True)
            X1_[j] = np.random.permutation(X1_[j].values)
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, j] = -log_loss(y1, prob, sample_weight=w1,
                                            labels=clf.classes_)
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, j] = accuracy_score(y1, pred, sample_weight=w1)
    imp = (-1 * scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)
    imp = pd.concat({"mean": imp.mean(),
                     "std": imp.std() * imp.shape[0] ** -0.5}, axis=1)
    return imp, scr0.mean()


# -----------------------------------------------------------------
# Snippet 8.4: Single Feature Importance (SFI)
# -----------------------------------------------------------------
def feat_imp_sfi(feat_names, clf, X, y, scoring="neg_log_loss", cv=5):
    imp = pd.DataFrame(columns=["mean", "std"])
    cvg = KFold(n_splits=cv, shuffle=False)
    for feat in feat_names:
        scores = []
        for tr, te in cvg.split(X):
            fit = clf.fit(X[[feat]].iloc[tr], y.iloc[tr])
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X[[feat]].iloc[te])
                scores.append(-log_loss(y.iloc[te], prob, labels=clf.classes_))
            else:
                pred = fit.predict(X[[feat]].iloc[te])
                scores.append(accuracy_score(y.iloc[te], pred))
        imp.loc[feat] = [np.mean(scores), np.std(scores) / np.sqrt(cv)]
    return imp


# -----------------------------------------------------------------
# Snippet 8.5: Orthogonal features via PCA
# -----------------------------------------------------------------
def get_orthogonal_features(dfX, var_thres=0.95):
    dfZ = dfX.sub(dfX.mean(), axis=1).div(dfX.std(), axis=1)
    dot = dfZ.T @ dfZ
    eVal, eVec = np.linalg.eigh(dot)
    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]
    cum_var = np.cumsum(eVal) / eVal.sum()
    dim = np.searchsorted(cum_var, var_thres) + 1
    eVal, eVec = eVal[:dim], eVec[:, :dim]
    dfP = dfZ.values @ eVec
    cols = [f"PC_{i+1}" for i in range(dim)]
    return pd.DataFrame(dfP, index=dfX.index, columns=cols), eVal


# -----------------------------------------------------------------
# Snippet 8.6: Weighted Kendall's tau between feature importance and PCA rank
# -----------------------------------------------------------------
def importance_pca_corr(feat_imp, pc_rank):
    """Higher = more agreement between PCA-derived structure and feature importance."""
    return weightedtau(feat_imp, pc_rank ** -1)[0]


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    X, y = make_classification(n_samples=500, n_features=8, n_informative=4,
                                random_state=0)
    cols = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y)

    rf = RandomForestClassifier(n_estimators=100, max_features=1,
                                  criterion="entropy")
    rf.fit(X, y)

    mdi = feat_imp_mdi(rf, cols)
    print("MDI importance:")
    print(mdi.round(3))

    mda, base = feat_imp_mda(RandomForestClassifier(n_estimators=50,
                                                       max_features=1),
                              X, y, cv=3, scoring="accuracy")
    print(f"\nMDA importance (base accuracy={base:.3f}):")
    print(mda.round(3))


if __name__ == "__main__":
    main()
