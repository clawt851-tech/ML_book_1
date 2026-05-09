"""
Chapter 16: Machine Learning Asset Allocation
==============================================

Implements Hierarchical Risk Parity (HRP):
  * 16.4.1: Tree clustering on correlation distance
  * 16.4.2: Quasi-diagonalization
  * 16.4.3: Recursive bisection for inverse-variance weights
  * Comparison with inverse-variance and Markowitz minimum-variance
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


# -----------------------------------------------------------------
# 16.A.1: Correlation distance
# -----------------------------------------------------------------
def correl_dist(corr):
    """d[i,j] = sqrt((1 - corr[i,j]) / 2)"""
    return np.sqrt((1 - corr) / 2)


# -----------------------------------------------------------------
# 16.4.2: Quasi-diagonalization
# -----------------------------------------------------------------
def get_quasi_diag(link):
    """Reorder rows/cols so similar items end up adjacent."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


# -----------------------------------------------------------------
# 16.A.2: Inverse Variance Portfolio
# -----------------------------------------------------------------
def get_ivp(cov):
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def get_cluster_var(cov, c_items):
    cov_ = cov.iloc[c_items, c_items]
    w_ = get_ivp(cov_).reshape(-1, 1)
    return float((w_.T @ cov_.values @ w_)[0, 0])


# -----------------------------------------------------------------
# 16.4.3: Recursive bisection
# -----------------------------------------------------------------
def get_rec_bipart(cov, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while c_items:
        new_items = []
        for items in c_items:
            if len(items) <= 1:
                continue
            half = len(items) // 2
            c0, c1 = items[:half], items[half:]
            v0 = get_cluster_var(cov, c0)
            v1 = get_cluster_var(cov, c1)
            alpha = 1 - v0 / (v0 + v1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
            new_items.extend([c0, c1])
        c_items = new_items
    return w


# -----------------------------------------------------------------
# Hierarchical Risk Parity (HRP) main
# -----------------------------------------------------------------
def hrp(cov, corr):
    dist = correl_dist(corr)
    link = linkage(squareform(dist.values, checks=False), method="single")
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    return get_rec_bipart(cov, sort_ix)


# -----------------------------------------------------------------
# Markowitz minimum-variance for comparison
# -----------------------------------------------------------------
def min_var_portfolio(cov):
    inv = np.linalg.pinv(cov.values)
    ones = np.ones(cov.shape[0])
    w = inv @ ones
    return pd.Series(w / w.sum(), index=cov.index)


# -----------------------------------------------------------------
# Demo on synthetic returns
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n_assets, T = 10, 1000
    # block correlation: 2 sectors of 5 assets each
    factor1 = rng.normal(0, 1, T)
    factor2 = rng.normal(0, 1, T)
    rets = pd.DataFrame()
    for i in range(5):
        rets[f"A{i}"] = 0.7 * factor1 + 0.3 * rng.normal(0, 1, T)
    for i in range(5):
        rets[f"B{i}"] = 0.7 * factor2 + 0.3 * rng.normal(0, 1, T)

    cov, corr = rets.cov(), rets.corr()
    w_hrp = hrp(cov, corr)
    w_ivp = pd.Series(get_ivp(cov), index=cov.index)
    w_mv = min_var_portfolio(cov)

    df = pd.concat({"HRP": w_hrp, "IVP": w_ivp, "MinVar": w_mv}, axis=1)
    print("Portfolio weights:")
    print(df.round(3))
    print(f"\nWeight std (concentration): "
          f"HRP={w_hrp.std():.3f}, IVP={w_ivp.std():.3f}, "
          f"MinVar={w_mv.std():.3f}")
    # HRP usually has more diversified (lower std) weights than MinVar.


if __name__ == "__main__":
    main()
