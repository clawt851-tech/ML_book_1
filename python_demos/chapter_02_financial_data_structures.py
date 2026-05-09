"""
Chapter 2: Financial Data Structures
=====================================

Demonstrates:
  * Standard bars: time, tick, volume, dollar
  * Information-driven bars: tick imbalance, volume imbalance, runs bars
  * The ETF trick / single future roll
  * PCA weights (Snippet 2.1)
  * CUSUM event-based sampling (Snippet 2.4)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------
# Synthetic tick generator (so demo is self-contained)
# -----------------------------------------------------------------
def generate_synthetic_ticks(n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2024-01-01 09:30", periods=n, freq="100ms")
    price = 100 + np.cumsum(rng.normal(0, 0.05, n))
    volume = rng.integers(1, 50, size=n)
    return pd.DataFrame({"price": price, "volume": volume}, index=dt)


# -----------------------------------------------------------------
# 2.3.1 Standard Bars
# -----------------------------------------------------------------
def time_bars(ticks, freq="1min"):
    """OHLCV bars at fixed time intervals."""
    return ticks.groupby(pd.Grouper(freq=freq)).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
    ).dropna()


def tick_bars(ticks, threshold=500):
    """Sample every `threshold` ticks."""
    out, buf = [], []
    count = 0
    for ts, row in ticks.iterrows():
        buf.append((ts, row.price, row.volume))
        count += 1
        if count >= threshold:
            df = pd.DataFrame(buf, columns=["ts", "price", "volume"])
            out.append({
                "ts": df.ts.iloc[-1],
                "open": df.price.iloc[0], "high": df.price.max(),
                "low": df.price.min(), "close": df.price.iloc[-1],
                "volume": df.volume.sum(),
            })
            buf, count = [], 0
    return pd.DataFrame(out).set_index("ts") if out else pd.DataFrame()


def volume_bars(ticks, threshold=10_000):
    """Sample whenever cumulative volume crosses threshold."""
    out, buf = [], []
    cum = 0
    for ts, row in ticks.iterrows():
        buf.append((ts, row.price, row.volume))
        cum += row.volume
        if cum >= threshold:
            df = pd.DataFrame(buf, columns=["ts", "price", "volume"])
            out.append({
                "ts": df.ts.iloc[-1],
                "open": df.price.iloc[0], "high": df.price.max(),
                "low": df.price.min(), "close": df.price.iloc[-1],
                "volume": df.volume.sum(),
            })
            buf, cum = [], 0
    return pd.DataFrame(out).set_index("ts") if out else pd.DataFrame()


def dollar_bars(ticks, threshold=1_000_000):
    """Sample whenever cumulative dollar value crosses threshold."""
    out, buf = [], []
    cum = 0
    for ts, row in ticks.iterrows():
        buf.append((ts, row.price, row.volume))
        cum += row.price * row.volume
        if cum >= threshold:
            df = pd.DataFrame(buf, columns=["ts", "price", "volume"])
            out.append({
                "ts": df.ts.iloc[-1],
                "open": df.price.iloc[0], "high": df.price.max(),
                "low": df.price.min(), "close": df.price.iloc[-1],
                "volume": df.volume.sum(),
            })
            buf, cum = [], 0
    return pd.DataFrame(out).set_index("ts") if out else pd.DataFrame()


# -----------------------------------------------------------------
# 2.3.2 Information-driven bars: Tick Imbalance Bars (TIBs)
# -----------------------------------------------------------------
def tick_rule(prices):
    """Lopez de Prado's tick rule: b_t = sign(Delta p_t) (carry forward zeros)."""
    diff = prices.diff().fillna(0).values
    b = np.zeros_like(diff)
    last = 1
    for i, d in enumerate(diff):
        if d > 0: last = 1
        elif d < 0: last = -1
        b[i] = last
    return pd.Series(b, index=prices.index)


def tick_imbalance_bars(ticks, ewm_span=100):
    """Bar terminates when |theta_T| >= E[T] * |2P[b=1]-1|."""
    b = tick_rule(ticks.price)
    out, theta = [], 0.0
    bar_lengths, bar_imbalances = [50], [b.iloc[:50].mean()]
    start = 0
    for i, bt in enumerate(b.values):
        theta += bt
        E_T = pd.Series(bar_lengths).ewm(span=ewm_span).mean().iloc[-1]
        E_b = pd.Series(bar_imbalances).ewm(span=ewm_span).mean().iloc[-1]
        thresh = E_T * abs(2 * E_b - 1) if E_b != 0 else E_T
        if abs(theta) >= max(thresh, 5):
            seg = ticks.iloc[start:i + 1]
            out.append({
                "ts": seg.index[-1], "open": seg.price.iloc[0],
                "close": seg.price.iloc[-1], "imbalance": theta,
                "n_ticks": len(seg),
            })
            bar_lengths.append(len(seg))
            bar_imbalances.append(seg.pipe(lambda x: tick_rule(x.price)).mean())
            theta, start = 0.0, i + 1
    return pd.DataFrame(out).set_index("ts") if out else pd.DataFrame()


# -----------------------------------------------------------------
# Snippet 2.1: PCA Weights from Risk Distribution
# -----------------------------------------------------------------
def pca_weights(cov, risk_dist=None, risk_target=1.0):
    eVal, eVec = np.linalg.eigh(cov)
    idx = eVal.argsort()[::-1]
    eVal, eVec = eVal[idx], eVec[:, idx]
    if risk_dist is None:
        risk_dist = np.zeros(cov.shape[0])
        risk_dist[-1] = 1.0
    loads = risk_target * np.sqrt(risk_dist / eVal)
    wghts = eVec @ loads.reshape(-1, 1)
    return wghts.flatten()


# -----------------------------------------------------------------
# Snippet 2.4: Symmetric CUSUM Filter
# -----------------------------------------------------------------
def cusum_filter(g_raw, h):
    """Sample event timestamps when run-up/down exceeds threshold h."""
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = g_raw.diff().dropna()
    for i, d in diff.items():
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -h:
            s_neg = 0.0
            t_events.append(i)
        elif s_pos > h:
            s_pos = 0.0
            t_events.append(i)
    return pd.DatetimeIndex(t_events)


# -----------------------------------------------------------------
# Snippet 2.2: Form a gaps series for futures roll
# -----------------------------------------------------------------
def roll_gaps(series, instrument_col="symbol", open_col="open", close_col="close",
              match_end=True):
    roll_dates = series[instrument_col].drop_duplicates(keep="first").index
    gaps = series[close_col] * 0
    iloc_list = list(series.index)
    iloc_idx = [iloc_list.index(i) - 1 for i in roll_dates]
    gaps.loc[roll_dates[1:]] = (series[open_col].loc[roll_dates[1:]].values
                                 - series[close_col].iloc[iloc_idx[1:]].values)
    gaps = gaps.cumsum()
    if match_end:
        gaps -= gaps.iloc[-1]
    return gaps


def main():
    ticks = generate_synthetic_ticks(20_000)

    tb = time_bars(ticks, "5min")
    tkb = tick_bars(ticks, 500)
    vb = volume_bars(ticks, 10_000)
    db = dollar_bars(ticks, 1_000_000)

    print(f"Time bars (5min): {len(tb)} bars")
    print(f"Tick bars (500):  {len(tkb)} bars")
    print(f"Volume bars (10k): {len(vb)} bars")
    print(f"Dollar bars (1M):  {len(db)} bars")

    # CUSUM events
    events = cusum_filter(ticks.price, h=0.5)
    print(f"\nCUSUM events (h=0.5): {len(events)}")

    # PCA weights demo
    cov = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    w = pca_weights(cov)
    print(f"\nPCA weights (risk on smallest eigval): {w}")


if __name__ == "__main__":
    main()
