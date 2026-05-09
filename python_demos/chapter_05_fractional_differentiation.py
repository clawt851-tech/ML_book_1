"""
Chapter 5: Fractionally Differentiated Features
================================================

Demonstrates:
  * Snippet 5.1: get_weights (binomial expansion, expanding window)
  * Snippet 5.2: frac_diff using expanding window
  * Snippet 5.3: get_weights_FFD (fixed-width window cutoff)
  * Snippet 5.4: frac_diff_FFD (fixed-width window)
  * Snippet 5.5: find minimum d that passes ADF stationarity test
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# -----------------------------------------------------------------
# Snippet 5.1: get weights for fractional differentiation
# -----------------------------------------------------------------
def get_weights(d, size):
    """w_k = -w_{k-1} * (d - k + 1) / k, w_0 = 1."""
    w = [1.0]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


# -----------------------------------------------------------------
# Snippet 5.2: Standard expanding-window fractional differentiation
# -----------------------------------------------------------------
def frac_diff(series, d, thres=0.01):
    """Expanding window: weights drop below thres are skipped per-row."""
    w = get_weights(d, series.shape[0])
    w_ = np.cumsum(abs(w))
    w_ /= w_[-1]
    skip = (w_ > thres).sum()
    out = {}
    for name in series.columns:
        s = series[[name]].ffill().dropna()
        df_ = pd.Series(index=s.index, dtype=float)
        for iloc in range(skip, s.shape[0]):
            loc = s.index[iloc]
            if not np.isfinite(series.loc[loc, name]):
                continue
            df_[loc] = float(np.dot(w[-(iloc + 1):, :].T,
                                     s.loc[:loc].values)[0, 0])
        out[name] = df_.copy()
    return pd.concat(out, axis=1)


# -----------------------------------------------------------------
# Snippet 5.3: Fixed-width window weights (drop terms where |w_k| < tau)
# -----------------------------------------------------------------
def get_weights_ffd(d, thres):
    w, k = [1.0], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


# -----------------------------------------------------------------
# Snippet 5.4: Fixed-width window fractional differentiation
# -----------------------------------------------------------------
def frac_diff_ffd(series, d, thres=1e-4):
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    out = {}
    for name in series.columns:
        s = series[[name]].ffill().dropna()
        df_ = pd.Series(index=s.index[width:], dtype=float)
        for iloc in range(width, s.shape[0]):
            loc0, loc1 = s.index[iloc - width], s.index[iloc]
            if not np.isfinite(series.loc[loc1, name]):
                continue
            df_.loc[loc1] = float(np.dot(w.T, s.loc[loc0:loc1].values)[0, 0])
        out[name] = df_.copy()
    return pd.concat(out, axis=1)


# -----------------------------------------------------------------
# Snippet 5.5: search for min d s.t. ADF passes
# -----------------------------------------------------------------
def plot_min_ffd(series, d_range=np.linspace(0, 1, 11), thres=1e-5):
    """Returns table of (d, ADF stat, p-value, lags, n_obs, 95% conf, corr w/ original)."""
    out = pd.DataFrame(columns=["adf_stat", "p_val", "lags", "n_obs",
                                 "conf_95", "corr"])
    for d in d_range:
        df1 = np.log(series[["close"]]).resample("1D").last().dropna()
        df2 = frac_diff_ffd(df1, d, thres)
        corr = np.corrcoef(df1.loc[df2.index, "close"],
                           df2["close"])[0, 1]
        adf = adfuller(df2["close"].dropna(), maxlag=1, regression="c", autolag=None)
        out.loc[d] = [adf[0], adf[1], adf[2], adf[3], adf[4]["5%"], corr]
    return out


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n = 1500
    idx = pd.date_range("2020-01-01", periods=n, freq="1H")
    log_price = np.log(100) + np.cumsum(rng.normal(0, 0.005, n))
    series = pd.DataFrame({"close": np.exp(log_price)}, index=idx)

    # weights for d in {0.1, 0.5, 0.9}
    for d in [0.1, 0.5, 0.9]:
        w = get_weights_ffd(d, thres=1e-4)
        print(f"d={d}: window width = {len(w)}")

    out = plot_min_ffd(series, d_range=np.linspace(0, 1, 6))
    print("\nADF test on FFD-transformed series:")
    print(out.round(3))

    # Expected pattern: as d -> 1, ADF stat becomes more negative (stationary),
    # but corr with original log-price decreases (less memory).


if __name__ == "__main__":
    main()
