"""
Chapter 18: Entropy Features
=============================

Demonstrates:
  * Snippet 18.1: Plug-in (maximum likelihood) entropy estimator
  * Snippet 18.2: Lempel-Ziv estimator (compression-based)
  * Snippet 18.3: Probabilistic LZ estimator (Kontoyiannis)
  * Snippet 18.4: Encoding schemes - binary, quantile, sigma
  * 18.6: Entropy of a Gaussian process
  * 18.8: Financial applications (market efficiency, portfolio concentration)
"""

import numpy as np
import pandas as pd
from collections import Counter
from math import log2


# -----------------------------------------------------------------
# Snippet 18.1: Plug-in / Maximum Likelihood entropy estimator
# -----------------------------------------------------------------
def plug_in_entropy(message, word_length=1):
    """H = -sum_w p(w) log2 p(w)"""
    n = len(message) - word_length + 1
    counter = Counter(message[i:i + word_length] for i in range(n))
    probs = np.array(list(counter.values())) / n
    return -np.sum(probs * np.log2(probs + 1e-12))


# -----------------------------------------------------------------
# Snippet 18.2: Lempel-Ziv (basic) entropy estimator
# -----------------------------------------------------------------
def lempel_ziv_entropy(message):
    """Compression-based entropy approximation."""
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            msg_ = message[i:j + 1]
            if msg_ not in lib:
                lib.append(msg_)
                break
        i = j + 1
    return len(lib) / len(message)


# -----------------------------------------------------------------
# Snippet 18.3: Probabilistic LZ (Kontoyiannis 1998)
# -----------------------------------------------------------------
def kontoyiannis_entropy(message, window=None):
    """h_hat = (1/n) * sum log2(L_i) / L_i, with bias correction."""
    out = {"num": 0, "sum": 0, "subS": []}
    if window is None:
        points = range(1, len(message) // 2 + 1)
        get_pre = lambda i: message[:i]
    else:
        points = range(window, len(message) - window + 1)
        get_pre = lambda i: message[i - window:i]
    for i in points:
        l, msg_ = match_length(message, i, window)
        out["sum"] += np.log2(i + 1) / l
        out["subS"].append(msg_)
        out["num"] += 1
    out["h"] = out["sum"] / out["num"] if out["num"] else 0
    out["r"] = 1 - out["h"] / np.log2(len(message)) if len(message) > 1 else 0
    return out


def match_length(msg, i, n):
    """Maximum length L such that msg[i:i+L] appears in msg[:i] (window-bounded)."""
    sub_s = ""
    for l in range(1, len(msg) - i + 1):
        sub_s = msg[i:i + l]
        prev = msg[max(0, i - (n or i)):i] if n else msg[:i]
        if sub_s not in prev:
            return l, sub_s
    return len(sub_s) + 1, sub_s


# -----------------------------------------------------------------
# Snippet 18.4: Encoding schemes
# -----------------------------------------------------------------
def encode_binary(returns):
    """Binary encoding: '1' if return > 0 else '0'."""
    return "".join("1" if r > 0 else "0" for r in returns)


def encode_quantile(values, n_bins=4):
    """Quantile encoding: each value -> bin index."""
    quantiles = pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
    return "".join(str(int(q)) for q in quantiles if not pd.isna(q))


def encode_sigma(values, sigma):
    """Sigma encoding: each value -> int(round(value / sigma))."""
    return "".join(str(int(round(v / sigma))) for v in values)


# -----------------------------------------------------------------
# 18.6: Entropy of a Gaussian process
# -----------------------------------------------------------------
def gaussian_entropy(sigma):
    """Differential entropy of N(0, sigma^2) in bits."""
    return 0.5 * np.log2(2 * np.pi * np.e * sigma ** 2)


# -----------------------------------------------------------------
# 18.8.3: Portfolio concentration via entropy
# -----------------------------------------------------------------
def portfolio_entropy(weights):
    """Effective number of bets via Shannon entropy."""
    w = np.abs(weights) / np.abs(weights).sum()
    h = -np.sum(w * np.log2(w + 1e-12))
    return 2 ** h


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)

    # Random vs structured signal
    rand_msg = "".join(rng.choice(["0", "1"], 1000))
    struct_msg = "01" * 500
    print(f"Plug-in H (random): {plug_in_entropy(rand_msg, 1):.3f}")
    print(f"Plug-in H (structured): {plug_in_entropy(struct_msg, 1):.3f}")
    print(f"Plug-in H (random, 4-grams): {plug_in_entropy(rand_msg, 4):.3f}")

    # LZ
    print(f"\nLZ H (random): {lempel_ziv_entropy(rand_msg):.4f}")
    print(f"LZ H (structured): {lempel_ziv_entropy(struct_msg):.4f}")

    # Returns -> binary encoding
    returns = rng.normal(0, 0.01, 500)
    enc = encode_binary(returns)
    h_returns = plug_in_entropy(enc, 1)
    print(f"\nBinary-encoded return entropy: {h_returns:.4f} (max=1.0)")

    # Portfolio entropy
    w = np.array([0.5, 0.3, 0.1, 0.1])
    print(f"Effective bets for w={w}: {portfolio_entropy(w):.2f}")
    print(f"Effective bets for equal weights: "
          f"{portfolio_entropy(np.array([0.25]*4)):.2f}")


if __name__ == "__main__":
    main()
