"""
Chapter 10: Bet Sizing
=======================

Demonstrates:
  * Snippet 10.1: getSignal - bet size from predicted probabilities
  * Snippet 10.2: avgActiveSignals - average bet sizes among active bets
  * Snippet 10.3: discreteSignal - size discretization to prevent overtrading
  * Snippet 10.4: dynamic position size and limit price (sigmoid)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# -----------------------------------------------------------------
# Snippet 10.1: from probabilities to bet size (one-vs-rest)
# -----------------------------------------------------------------
def get_signal(events, step_size, prob, pred, num_classes):
    """
    z = (p - 1/k) / sqrt(p * (1 - p))
    bet size = pred * (2 * Phi(z) - 1)        in [-1, 1]
    """
    if prob.shape[0] == 0:
        return pd.Series(dtype=float)
    signal0 = (prob - 1.0 / num_classes) / np.sqrt(prob * (1 - prob))
    signal0 = pred * (2 * norm.cdf(signal0) - 1)
    if "side" in events:
        signal0 *= events.loc[signal0.index, "side"]
    df0 = signal0.to_frame("signal").join(events[["t1"]], how="left")
    df0 = avg_active_signals(df0)
    return discrete_signal(df0, step_size)


# -----------------------------------------------------------------
# Snippet 10.2: average bets across all bets still active
# -----------------------------------------------------------------
def avg_active_signals(signals):
    """Compute mean signal among bets active at each time."""
    t_pnts = set(signals["t1"].dropna().values) | set(signals.index.values)
    out = pd.Series(0.0, index=sorted(t_pnts))
    for loc in out.index:
        active = signals.index[
            (signals.index <= loc)
            & ((signals["t1"] > loc) | signals["t1"].isna())
        ]
        if len(active):
            out.loc[loc] = signals.loc[active, "signal"].mean()
    return out


# -----------------------------------------------------------------
# Snippet 10.3: size discretization
# -----------------------------------------------------------------
def discrete_signal(signal0, step_size=0.05):
    """Round to multiples of step_size, cap at +/- 1."""
    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1
    signal1[signal1 < -1] = -1
    return signal1


# -----------------------------------------------------------------
# Snippet 10.4: dynamic position size + limit price (sigmoid)
# -----------------------------------------------------------------
def bet_size_sigmoid(w, x):
    """m = x / sqrt(w + x^2). Maps R -> [-1, 1]."""
    return x * (w + x ** 2) ** -0.5


def get_target_position(w, f, mP, max_pos):
    """Target int position from forecast f, market price mP."""
    return int(bet_size_sigmoid(w, f - mP) * max_pos)


def inv_price_sigmoid(f, w, m):
    """Inverse: given size m and forecast f, what price gives m?"""
    return f - m * (w / (1 - m ** 2)) ** 0.5


def limit_price_sigmoid(t_pos, pos, f, w, max_pos):
    """Compute breakeven limit price for the order size t_pos - pos."""
    sgn = 1 if t_pos >= pos else -1
    lp = 0.0
    for j in range(abs(pos + sgn), abs(t_pos + 1)):
        lp += inv_price_sigmoid(f, w, j / float(max_pos))
    return lp / (t_pos - pos)


def calibrate_w(divergence, m):
    """Solve for w given (divergence x, target bet size m)."""
    return divergence ** 2 * (m ** -2 - 1)


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    # bet sizes from probabilities
    probs = np.linspace(0.5, 0.99, 10)
    sizes = (2 * norm.cdf((probs - 0.5) / np.sqrt(probs * (1 - probs))) - 1)
    print("Bet size for various predicted probabilities (binary):")
    for p, s in zip(probs, sizes):
        print(f"  p={p:.2f} -> size={s:+.3f}")

    # discretization
    raw = np.linspace(-1, 1, 21)
    disc = discrete_signal(pd.Series(raw), step_size=0.2)
    print(f"\nDiscretization at step=0.2: {disc.unique()}")

    # dynamic position size
    pos, max_pos, mP, f = 0, 100, 100, 115
    w = calibrate_w(divergence=10, m=0.95)
    t_pos = get_target_position(w, f, mP, max_pos)
    print(f"\nDynamic position: w={w:.2f}, target={t_pos}")
    if t_pos != pos:
        lp = limit_price_sigmoid(t_pos, pos, f, w, max_pos)
        print(f"Limit price for buy order of {t_pos - pos}: {lp:.4f}")


if __name__ == "__main__":
    main()
