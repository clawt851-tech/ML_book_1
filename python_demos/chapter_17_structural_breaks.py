"""
Chapter 17: Structural Breaks
==============================

Demonstrates:
  * Snippet 17.1-17.4: SADF (Supremum Augmented Dickey-Fuller) test
  * Chu-Stinchcombe-White CUSUM test on log-prices
  * Sub/Super-Martingale tests
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# Snippet 17.3: Apply lags to dataframe
# -----------------------------------------------------------------
def lag_df(df0, lags):
    df1 = pd.DataFrame()
    if isinstance(lags, int):
        lags = list(range(lags + 1))
    else:
        lags = [int(lag) for lag in lags]
    for lag in lags:
        df_ = df0.shift(lag).copy()
        df_.columns = [f"{c}_{lag}" for c in df_.columns]
        df1 = df1.join(df_, how="outer") if not df1.empty else df_
    return df1


# -----------------------------------------------------------------
# Snippet 17.2: Prepare X, y for ADF specification
# -----------------------------------------------------------------
def get_yx(series, constant="ct", lags=1):
    series_ = series.diff().dropna()
    x = lag_df(series_.to_frame(), lags).dropna()
    x.iloc[:, 0] = series.values[-x.shape[0] - 1:-1, 0]
    y = series_.iloc[-x.shape[0]:].values
    if constant != "nc":
        x = pd.concat([x, pd.Series(1.0, index=x.index, name="const")], axis=1)
        if constant[:2] == "ct":
            trend = pd.Series(np.arange(x.shape[0]), index=x.index, name="trend")
            x = pd.concat([x, trend], axis=1)
        if constant == "ctt":
            x["trend2"] = trend ** 2
    return y, x


# -----------------------------------------------------------------
# Snippet 17.4: ADF coefficient and t-stat
# -----------------------------------------------------------------
def get_betas(y, x):
    xy = x.T @ y
    xx = x.T @ x
    xx_inv = np.linalg.inv(xx)
    b_mean = xx_inv @ xy
    err = y - x @ b_mean
    b_var = (err.T @ err) / (x.shape[0] - x.shape[1]) * xx_inv
    return b_mean, b_var


# -----------------------------------------------------------------
# Snippet 17.1: SADF inner loop
# -----------------------------------------------------------------
def get_bsadf(log_p, min_sl, constant="ct", lags=1):
    """Compute backward-expanding SADF at the rightmost time point."""
    y, x = get_yx(log_p.to_frame(), constant=constant, lags=lags)
    start_points = range(0, y.shape[0] + lags - min_sl + 1)
    all_adf = []
    for start in start_points:
        y_, x_ = y[start:], x.iloc[start:].values
        try:
            b_mean, b_var = get_betas(y_, x_)
            adf_t = b_mean[0] / b_var[0, 0] ** 0.5
            all_adf.append(adf_t)
        except np.linalg.LinAlgError:
            continue
    bsadf = max(all_adf) if all_adf else np.nan
    return {"Time": log_p.index[-1], "gsadf": bsadf}


# -----------------------------------------------------------------
# Chu-Stinchcombe-White CUSUM test on levels
# -----------------------------------------------------------------
def cs_white_cusum(y, n=50):
    """One-sided test on standardized departure of log-price from level y_n."""
    if len(y) < n + 2:
        return np.nan
    y_n = y.iloc[n]
    diff = y.diff().dropna()
    sigma_t2 = (diff ** 2).expanding().mean()
    s_t = (y - y_n) / (np.sqrt(sigma_t2) * np.sqrt(np.arange(len(y))[::-1] + 1))
    return s_t.iloc[n + 1:]


# -----------------------------------------------------------------
# Sub/Super-Martingale test (SMT) for polynomial trend
# -----------------------------------------------------------------
def smt_polynomial(y, phi=1.0):
    """y_t = alpha + gamma*t + beta*t^2 + eps; test |beta| / sigma_beta^phi."""
    n = len(y)
    t = np.arange(n)
    X = np.column_stack([np.ones(n), t, t ** 2])
    b_mean, b_var = get_betas(y.values, X)
    return abs(b_mean[2]) / b_var[2, 2] ** (phi / 2)


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="1D")
    # simulate a process with a regime switch in the middle
    log_p = np.zeros(n)
    log_p[0] = np.log(100)
    for i in range(1, n):
        if i < n // 2:
            log_p[i] = log_p[i - 1] + rng.normal(0, 0.01)
        else:
            log_p[i] = log_p[i - 1] * 1.001 + rng.normal(0, 0.01)  # explosive
    log_p = pd.Series(log_p, index=idx)

    # SADF over the final window
    out = get_bsadf(log_p.iloc[-100:], min_sl=20, constant="ct", lags=1)
    print(f"SADF at end: {out['gsadf']:.3f}")

    # CUSUM
    s_t = cs_white_cusum(log_p, n=50)
    print(f"CS-White max abs CUSUM: {s_t.abs().max():.3f}")

    # SMT
    smt = smt_polynomial(log_p, phi=1.0)
    print(f"SMT (poly, phi=1): {smt:.2f}")


if __name__ == "__main__":
    main()
