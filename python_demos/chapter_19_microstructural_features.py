"""
Chapter 19: Microstructural Features
=====================================

Demonstrates:
  * Tick rule (Snippet 19.1)
  * Roll model: implied bid-ask spread from serial covariance
  * High-Low volatility estimator (Beckers, Parkinson)
  * Corwin-Schultz spread estimator
  * Kyle's lambda (price impact)
  * Amihud's lambda (illiquidity)
  * Hasbrouck's lambda (signed dollar volume regression)
  * VPIN (Volume-Synchronized Probability of Informed Trading)
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# Tick rule
# -----------------------------------------------------------------
def tick_rule(prices):
    diff = prices.diff().fillna(0).values
    b = np.zeros_like(diff)
    last = 1
    for i, d in enumerate(diff):
        if d > 0: last = 1
        elif d < 0: last = -1
        b[i] = last
    return pd.Series(b, index=prices.index)


# -----------------------------------------------------------------
# Roll model: implied half-spread from serial covariance of changes
# -----------------------------------------------------------------
def roll_spread(prices):
    """c = sqrt(-Cov(Δp_t, Δp_{t-1}))   (only valid when Cov < 0)"""
    delta_p = prices.diff().dropna()
    cov = np.cov(delta_p[1:], delta_p[:-1])[0, 1]
    return 2.0 * np.sqrt(-cov) if cov < 0 else np.nan


# -----------------------------------------------------------------
# Parkinson (1980) High-Low volatility estimator
# -----------------------------------------------------------------
def parkinson_vol(high, low, k=4 * np.log(2)):
    """sigma^2 = (1/(4 ln 2)) * E[(ln H - ln L)^2]"""
    return np.sqrt((np.log(high / low) ** 2).mean() / k)


# -----------------------------------------------------------------
# Corwin & Schultz (2012) spread estimator
# -----------------------------------------------------------------
def corwin_schultz_spread(high, low):
    """Two-day high-low spread estimator (closed-form)."""
    beta = (np.log(high / low) ** 2).rolling(2).sum()
    gamma = np.log(high.rolling(2).max() / low.rolling(2).min()) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) \
            - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread


# -----------------------------------------------------------------
# Kyle's lambda: price impact regression
# -----------------------------------------------------------------
def kyle_lambda(returns, signed_volume):
    """r_t = lambda * b_t * v_t + eps  =>  lambda via OLS slope."""
    x = signed_volume.values.reshape(-1, 1)
    y = returns.values
    return float(np.linalg.lstsq(x, y, rcond=None)[0][0])


# -----------------------------------------------------------------
# Amihud's lambda: illiquidity
# -----------------------------------------------------------------
def amihud_lambda(abs_returns, dollar_volume):
    """ILLIQ_t = |r_t| / DV_t,  Amihud = average."""
    return (abs_returns / dollar_volume.replace(0, np.nan)).dropna().mean()


# -----------------------------------------------------------------
# Hasbrouck's lambda: log-return on signed sqrt-dollar-volume
# -----------------------------------------------------------------
def hasbrouck_lambda(log_returns, signed_dollar_vol):
    """log r_t = lambda * sign(Δp_t) sqrt(|p_t * v_t|) + eps"""
    x = (np.sign(signed_dollar_vol) *
          np.sqrt(np.abs(signed_dollar_vol))).values.reshape(-1, 1)
    return float(np.linalg.lstsq(x, log_returns.values, rcond=None)[0][0])


# -----------------------------------------------------------------
# VPIN: Volume-Synchronized Probability of Informed Trading
# -----------------------------------------------------------------
def vpin(volume, buy_volume, n=50):
    """VPIN = E[|V_buy - V_sell|] / E[V_total] over rolling n bars."""
    sell_volume = volume - buy_volume
    imbalance = (buy_volume - sell_volume).abs()
    return imbalance.rolling(n).sum() / volume.rolling(n).sum()


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="1H")
    price = pd.Series(100 + np.cumsum(rng.normal(0, 0.1, n)), index=idx)
    high = price + np.abs(rng.normal(0, 0.05, n))
    low = price - np.abs(rng.normal(0, 0.05, n))
    vol = pd.Series(rng.integers(100, 1000, n), index=idx)

    print(f"Roll spread: {roll_spread(price):.4f}")
    print(f"Parkinson vol: {parkinson_vol(high, low):.4f}")
    print(f"Corwin-Schultz spread (mean): "
          f"{corwin_schultz_spread(high, low).mean():.4f}")

    rets = price.pct_change().dropna()
    signed_vol = tick_rule(price).iloc[1:] * vol.iloc[1:]
    print(f"\nKyle's lambda: {kyle_lambda(rets, signed_vol):.2e}")
    print(f"Amihud's lambda: "
          f"{amihud_lambda(rets.abs(), price.iloc[1:] * vol.iloc[1:]):.2e}")

    buy_vol = vol * (tick_rule(price) > 0).astype(float)
    vp = vpin(vol, buy_vol, n=20).dropna()
    print(f"\nVPIN (mean): {vp.mean():.3f}")


if __name__ == "__main__":
    main()
