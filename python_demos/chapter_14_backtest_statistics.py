"""
Chapter 14: Backtest Statistics
================================

Demonstrates:
  * Snippet 14.1: derive bet timestamps from target positions
  * Snippet 14.2: holding period estimator
  * Snippet 14.3: HHI concentration of returns
  * Snippet 14.4: Drawdown (DD) and Time under Water (TuW)
  * Snippet 14.5: Marcos' 3rd Law - report all trials with backtest
  * Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# -----------------------------------------------------------------
# Snippet 14.1: derive bet timestamps from target positions
# -----------------------------------------------------------------
def get_bet_timestamps(t_pos):
    """A bet takes place between flat positions or position flips."""
    df0 = t_pos[t_pos == 0].index
    df1 = t_pos.shift(1)
    df1 = df1[df1 != 0].index
    bets = df0.intersection(df1)         # flattenings
    df2 = t_pos.iloc[1:] * t_pos.iloc[:-1].values
    bets = bets.union(df2[df2 < 0].index).sort_values()  # flips
    if t_pos.index[-1] not in bets:
        bets = bets.append(pd.Index([t_pos.index[-1]]))
    return bets


# -----------------------------------------------------------------
# Snippet 14.2: holding period estimator
# -----------------------------------------------------------------
def get_holding_period(t_pos):
    """Average holding period (in days), weighted by entry size."""
    hp, t_entry = pd.DataFrame(columns=["dT", "w"]), 0.0
    p_diff = t_pos.diff()
    t_diff = (t_pos.index - t_pos.index[0]) / np.timedelta64(1, "D")
    for i in range(1, t_pos.shape[0]):
        if p_diff.iloc[i] * t_pos.iloc[i - 1] >= 0:    # increased / unchanged
            if t_pos.iloc[i] != 0:
                t_entry = (t_entry * t_pos.iloc[i - 1]
                            + t_diff[i] * p_diff.iloc[i]) / t_pos.iloc[i]
        else:                                             # decreased
            if t_pos.iloc[i] * t_pos.iloc[i - 1] < 0:    # flip
                hp.loc[t_pos.index[i]] = [
                    t_diff[i] - t_entry, abs(t_pos.iloc[i - 1])
                ]
                t_entry = t_diff[i]
            else:
                hp.loc[t_pos.index[i]] = [
                    t_diff[i] - t_entry, abs(p_diff.iloc[i])
                ]
    if hp["w"].sum() > 0:
        return (hp["dT"] * hp["w"]).sum() / hp["w"].sum()
    return np.nan


# -----------------------------------------------------------------
# Snippet 14.3: HHI concentration
# -----------------------------------------------------------------
def get_hhi(bet_ret):
    """Herfindahl-Hirschman Index of bet returns concentration."""
    if bet_ret.shape[0] <= 2:
        return np.nan
    wght = bet_ret / bet_ret.sum()
    hhi = (wght ** 2).sum()
    return (hhi - bet_ret.shape[0] ** -1) / (1.0 - bet_ret.shape[0] ** -1)


# -----------------------------------------------------------------
# Snippet 14.4: Drawdown and Time under Water
# -----------------------------------------------------------------
def compute_dd_tuw(series, dollars=False):
    df0 = series.to_frame("pnl")
    df0["hwm"] = series.expanding().max()
    df1 = df0.groupby("hwm").min().reset_index()
    df1.columns = ["hwm", "min"]
    df1.index = df0["hwm"].drop_duplicates(keep="first").index
    df1 = df1[df1["hwm"] > df1["min"]]
    dd = (df1["hwm"] - df1["min"]) if dollars else \
         (1 - df1["min"] / df1["hwm"])
    tuw = ((df1.index[1:] - df1.index[:-1]) /
           np.timedelta64(1, "Y")).values
    tuw = pd.Series(tuw, index=df1.index[:-1])
    return dd, tuw


# -----------------------------------------------------------------
# Probabilistic Sharpe Ratio (PSR)
# -----------------------------------------------------------------
def probabilistic_sharpe_ratio(sr, sr_benchmark, T, skew, kurt):
    """PSR[SR*]: probability that observed SR > benchmark SR*."""
    num = (sr - sr_benchmark) * np.sqrt(T - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return norm.cdf(num / den)


# -----------------------------------------------------------------
# Deflated Sharpe Ratio (DSR)
# -----------------------------------------------------------------
def deflated_sharpe_ratio(sr, T, skew, kurt, n_trials, var_trials):
    """DSR adjusts PSR by accounting for multiple trials."""
    gamma = 0.5772156649  # Euler-Mascheroni
    sr_star = np.sqrt(var_trials) * (
        (1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
        + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    return probabilistic_sharpe_ratio(sr, sr_star, T, skew, kurt)


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=500, freq="1D")

    # Simulate target positions
    t_pos = pd.Series(rng.choice([-1, 0, 1], size=500, p=[0.3, 0.4, 0.3]),
                       index=idx)
    bets = get_bet_timestamps(t_pos)
    print(f"Number of bets: {len(bets)}")
    hp = get_holding_period(t_pos)
    print(f"Average holding period: {hp:.1f} days")

    # PnL
    rets = pd.Series(rng.normal(0.001, 0.02, 500), index=idx)
    cum_pnl = rets.cumsum()
    dd, tuw = compute_dd_tuw(np.exp(cum_pnl))
    print(f"\nMax drawdown: {dd.max():.2%}")
    print(f"95th-pct DD: {dd.quantile(0.95):.2%}")

    # PSR / DSR
    sr_obs = rets.mean() / rets.std() * np.sqrt(252)
    print(f"\nObserved annualized SR: {sr_obs:.3f}")
    psr = probabilistic_sharpe_ratio(sr_obs, 0.0, len(rets),
                                       rets.skew(), rets.kurtosis() + 3)
    print(f"PSR (benchmark=0): {psr:.3f}")
    dsr = deflated_sharpe_ratio(sr_obs, len(rets), rets.skew(),
                                  rets.kurtosis() + 3,
                                  n_trials=100, var_trials=0.5)
    print(f"DSR (100 trials, var=0.5): {dsr:.3f}")


if __name__ == "__main__":
    main()
