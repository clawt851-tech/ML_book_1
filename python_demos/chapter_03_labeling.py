"""
Chapter 3: Labeling
===================

Demonstrates:
  * Snippet 3.1: Daily volatility estimates (dynamic threshold)
  * Snippet 3.2-3.3: Triple-barrier method
  * Snippet 3.4: Vertical barrier
  * Snippet 3.5/3.7: getBins (with optional meta-labeling)
  * Snippet 3.8: dropLabels (drop under-populated classes)
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# Snippet 3.1: Daily volatility (EWM std of close-to-close returns)
# -----------------------------------------------------------------
def get_daily_vol(close, span=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0 > 0]
    df0 = pd.Series(close.index[df0 - 1],
                    index=close.index[close.shape[0] - df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    return df0.ewm(span=span).std()


# -----------------------------------------------------------------
# Snippet 3.2: Apply pt/sl on t1
# -----------------------------------------------------------------
def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)
    pt = pt_sl[0] * events_["trgt"] if pt_sl[0] > 0 else pd.Series(index=events_.index, dtype=float)
    sl = -pt_sl[1] * events_["trgt"] if pt_sl[1] > 0 else pd.Series(index=events_.index, dtype=float)
    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close.loc[loc] - 1) * events_.at[loc, "side"]
        out.at[loc, "sl"] = df0[df0 < sl[loc]].index.min() if pt_sl[1] > 0 else pd.NaT
        out.at[loc, "pt"] = df0[df0 > pt[loc]].index.min() if pt_sl[0] > 0 else pd.NaT
    return out


# -----------------------------------------------------------------
# Snippet 3.3 + 3.6: getEvents (with optional meta-labeling side)
# -----------------------------------------------------------------
def get_events(close, t_events, pt_sl, trgt, min_ret, t1=None, side=None):
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    if t1 is None:
        t1 = pd.Series(pd.NaT, index=t_events)
    if side is None:
        side_, pt_sl_ = pd.Series(1.0, index=trgt.index), [pt_sl[0], pt_sl[0]]
    else:
        side_, pt_sl_ = side.loc[trgt.index], pt_sl[:2]
    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])
    df0 = apply_pt_sl_on_t1(close, events, pt_sl_, events.index)
    events["t1"] = df0.dropna(how="all").min(axis=1)
    if side is None:
        events = events.drop("side", axis=1)
    return events


# -----------------------------------------------------------------
# Snippet 3.4: Add vertical barrier
# -----------------------------------------------------------------
def add_vertical_barrier(t_events, close, num_days=1):
    t1 = close.index.searchsorted(t_events + pd.Timedelta(days=num_days))
    t1 = t1[t1 < close.shape[0]]
    return pd.Series(close.index[t1], index=t_events[:t1.shape[0]])


# -----------------------------------------------------------------
# Snippet 3.7: getBins (with meta-labeling case)
# -----------------------------------------------------------------
def get_bins(events, close):
    """
    bin in {-1, 0, 1} when 'side' is NOT in events  -> directional learning
    bin in {0, 1} when 'side' IS in events         -> meta-labeling (size only)
    """
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(events_["t1"].values).drop_duplicates()
    px = close.reindex(px, method="bfill")
    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].values].values / px.loc[events_.index] - 1
    if "side" in events_:
        out["ret"] *= events_["side"]
    out["bin"] = np.sign(out["ret"])
    if "side" in events_:
        out.loc[out["ret"] <= 0, "bin"] = 0  # meta-labeling: 1 = take bet
    return out


# -----------------------------------------------------------------
# Snippet 3.8: drop under-populated labels
# -----------------------------------------------------------------
def drop_labels(events, min_pct=0.05):
    while True:
        df0 = events["bin"].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        print(f"dropped label {df0.idxmin()} ({df0.min():.2%})")
        events = events[events["bin"] != df0.idxmin()]
    return events


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=2000, freq="1H")
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.2, 2000)), index=idx)

    vol = get_daily_vol(close, span=50).dropna()
    t_events = vol.index[::20]
    t1 = add_vertical_barrier(t_events, close, num_days=2)

    events = get_events(close, t_events, pt_sl=[1, 1], trgt=vol,
                        min_ret=0.0, t1=t1)
    bins = get_bins(events, close)
    print("Triple-barrier label distribution:")
    print(bins["bin"].value_counts())
    print(f"\nMean return per label:\n{bins.groupby('bin')['ret'].mean()}")


if __name__ == "__main__":
    main()
