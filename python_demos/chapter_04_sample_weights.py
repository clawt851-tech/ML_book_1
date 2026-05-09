"""
Chapter 4: Sample Weights
==========================

Demonstrates:
  * Snippet 4.1: mpNumCoEvents - number of concurrent labels
  * Snippet 4.2: mpSampleTW - average uniqueness
  * Snippet 4.3-4.5: Sequential bootstrap
  * Snippet 4.10: Sample weight by absolute return attribution
  * Snippet 4.11: Time-decay factors
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# Snippet 4.1: number of concurrent events at each time
# -----------------------------------------------------------------
def mp_num_co_events(close_idx, t1, molecule):
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1] + 1])
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0]:t1[molecule].max()]


# -----------------------------------------------------------------
# Snippet 4.2: average uniqueness for each label
# -----------------------------------------------------------------
def mp_sample_tw(t1, num_co_events, molecule):
    wght = pd.Series(index=molecule, dtype=float)
    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (1.0 / num_co_events.loc[t_in:t_out]).mean()
    return wght


# -----------------------------------------------------------------
# Snippet 4.3: indicator matrix (which bars influence which label)
# -----------------------------------------------------------------
def get_ind_matrix(bar_ix, t1):
    ind_m = pd.DataFrame(0, index=bar_ix, columns=range(t1.shape[0]))
    for i, (t0, t1_v) in enumerate(t1.items()):
        ind_m.loc[t0:t1_v, i] = 1.0
    return ind_m


# -----------------------------------------------------------------
# Snippet 4.4: average uniqueness from indicator matrix
# -----------------------------------------------------------------
def get_avg_uniqueness(ind_m):
    c = ind_m.sum(axis=1)         # concurrency at each bar
    u = ind_m.div(c, axis=0)      # per-label uniqueness at each bar
    return u[u > 0].mean()         # average over each label's lifespan


# -----------------------------------------------------------------
# Snippet 4.5: sequential bootstrap
# -----------------------------------------------------------------
def seq_bootstrap(ind_m, s_length=None):
    if s_length is None:
        s_length = ind_m.shape[1]
    phi = []
    while len(phi) < s_length:
        avg_u = pd.Series(dtype=float)
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]
        prob = avg_u / avg_u.sum()
        phi += [np.random.choice(ind_m.columns, p=prob.values)]
    return phi


# -----------------------------------------------------------------
# Snippet 4.10: Sample weight by absolute return attribution
# -----------------------------------------------------------------
def mp_sample_w(t1, num_co_events, close, molecule):
    ret = np.log(close).diff()
    wght = pd.Series(index=molecule, dtype=float)
    for t_in, t_out in t1.loc[wght.index].items():
        wght.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()
    return wght.abs()


# -----------------------------------------------------------------
# Snippet 4.11: Time-decay factors (piecewise-linear)
# -----------------------------------------------------------------
def get_time_decay(tW, clf_last_w=1.0):
    """tW: average uniqueness sorted by index. Newest gets w=1, oldest=clf_last_w."""
    clf_w = tW.sort_index().cumsum()
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / clf_w.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1) * clf_w.iloc[-1])
    const = 1.0 - slope * clf_w.iloc[-1]
    clf_w = const + slope * clf_w
    clf_w[clf_w < 0] = 0
    return clf_w


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1H")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, n)), index=idx)
    starts = idx[::25]
    t1 = pd.Series([idx[min(close.index.searchsorted(s) + 30, n - 1)]
                    for s in starts], index=starts)

    ind_m = get_ind_matrix(close.index, t1)
    avg_u = get_avg_uniqueness(ind_m)
    print(f"Average uniqueness per label (head):\n{avg_u.head()}")
    print(f"\nMean uniqueness: {avg_u.mean():.3f} "
          f"(IID would be ~1.0)")

    decay = get_time_decay(avg_u, clf_last_w=0.5)
    print(f"\nTime-decay weights (newest -> 1.0, oldest -> 0.5):")
    print(decay.iloc[[0, len(decay) // 2, -1]])


if __name__ == "__main__":
    main()
