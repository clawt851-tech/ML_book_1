"""
Chapter 15: Understanding Strategy Risk
========================================

Demonstrates:
  * Symmetric payouts: SR as a function of (precision p, payout pi, n bets)
  * Asymmetric payouts: SR as a function of (p, pi+, pi-)
  * The Probability of Strategy Failure
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom, norm


# -----------------------------------------------------------------
# Symmetric payouts: bet pays +pi with prob p, -pi with prob 1-p
# -----------------------------------------------------------------
def sharpe_ratio_symmetric(p, n_bets_per_year):
    """
    For symmetric payouts ({+pi, -pi}):
    SR = (2p - 1) / (2 sqrt(p(1-p))) * sqrt(n_bets_per_year)
    """
    return (2 * p - 1) / (2 * np.sqrt(p * (1 - p))) * np.sqrt(n_bets_per_year)


def required_precision_for_sr(target_sr, n_bets_per_year):
    """Inverse: minimum p needed to achieve target SR."""
    def equation(p):
        return sharpe_ratio_symmetric(p, n_bets_per_year) - target_sr
    return brentq(equation, 0.5001, 0.999)


def required_freq_for_sr(target_sr, p):
    """Bets per year needed to achieve target SR with given precision p."""
    sr_per_bet = sharpe_ratio_symmetric(p, 1)
    if sr_per_bet <= 0:
        return np.inf
    return (target_sr / sr_per_bet) ** 2


# -----------------------------------------------------------------
# Asymmetric payouts: bet pays +pi+ with prob p, -pi- with prob 1-p
# -----------------------------------------------------------------
def sharpe_ratio_asymmetric(p, pi_plus, pi_minus, n_bets_per_year):
    """E[X] = p*pi+ + (1-p)*(-pi-),  V[X] = p*(1-p)*(pi+ + pi-)^2"""
    mu = p * pi_plus - (1 - p) * pi_minus
    var = p * (1 - p) * (pi_plus + pi_minus) ** 2
    if var <= 0:
        return np.inf if mu > 0 else -np.inf
    return mu / np.sqrt(var) * np.sqrt(n_bets_per_year)


def implied_precision(pi_plus, pi_minus, target_sr, n):
    """Given payouts pi+/pi- and target SR, what p is required?"""
    def f(p):
        return sharpe_ratio_asymmetric(p, pi_plus, pi_minus, n) - target_sr
    return brentq(f, 0.001, 0.999)


# -----------------------------------------------------------------
# 15.4: Probability of Strategy Failure
# -----------------------------------------------------------------
def prob_strategy_failure(p_observed, n_bets, n_simulations=10_000,
                            target_sr_min=1.0, seed=0):
    """
    Estimate P[SR_realized < target_sr_min] given true precision p_observed,
    via Monte Carlo of binomial outcomes.
    """
    rng = np.random.default_rng(seed)
    fail = 0
    for _ in range(n_simulations):
        wins = rng.binomial(n_bets, p_observed)
        p_real = wins / n_bets
        sr = sharpe_ratio_symmetric(p_real, n_bets)
        if sr < target_sr_min:
            fail += 1
    return fail / n_simulations


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    print("=== Symmetric payouts ===")
    for p in [0.51, 0.55, 0.6]:
        for n in [10, 100, 1000]:
            sr = sharpe_ratio_symmetric(p, n)
            print(f"p={p}, n_bets={n:>4}/yr -> SR = {sr:.3f}")

    print("\n=== Required precision for SR=1 (annual) ===")
    for n in [50, 100, 252, 1000]:
        p_req = required_precision_for_sr(1.0, n)
        print(f"  n={n:>4}/yr -> p* = {p_req:.4f}")

    print("\n=== Asymmetric payouts: pi+=2, pi-=1 ===")
    for p in [0.3, 0.4, 0.5]:
        sr = sharpe_ratio_asymmetric(p, 2.0, 1.0, 100)
        print(f"  p={p} -> SR = {sr:.3f}")

    print("\n=== P[strategy fails to deliver SR>=1] ===")
    pf = prob_strategy_failure(p_observed=0.55, n_bets=100,
                                  target_sr_min=1.0)
    print(f"  true p=0.55, n=100 bets, target SR=1: P[fail] = {pf:.2%}")


if __name__ == "__main__":
    main()
