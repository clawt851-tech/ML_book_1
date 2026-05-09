"""
Chapter 22: High-Performance Computational Intelligence
========================================================

Demonstrates use cases discussed in Wu & Simon's contribution:
  * VPIN calibration (high-frequency informed-trading detection)
  * Non-uniform Fast Fourier Transform (NUFFT) for irregular tick data
  * In-situ processing concept (compute as data is generated)
  * Message Passing Interface (MPI) workflow stub
"""

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# 22.6.5: VPIN calibration (Volume-Synchronized PIN)
# -----------------------------------------------------------------
def calibrate_vpin(tick_returns, tick_volumes, bucket_size, n_buckets=50):
    """
    Bulk-classify each tick using normal CDF on standardized return,
    then aggregate into volume buckets, compute VPIN over rolling n_buckets.
    """
    from scipy.stats import norm
    sigma = tick_returns.std()
    if sigma <= 0:
        return pd.Series(dtype=float)
    z = tick_returns / sigma
    # Buy probability = Phi(z), Sell probability = 1 - Phi(z)
    p_buy = norm.cdf(z)
    buy_v = tick_volumes * p_buy
    sell_v = tick_volumes * (1 - p_buy)
    # Aggregate into volume buckets
    cum_v = tick_volumes.cumsum()
    bucket_id = (cum_v / bucket_size).astype(int)
    df = pd.DataFrame({"buy": buy_v, "sell": sell_v,
                        "v": tick_volumes, "bucket": bucket_id})
    bucket_agg = df.groupby("bucket").agg({"buy": "sum", "sell": "sum",
                                             "v": "sum"})
    imbalance = (bucket_agg["buy"] - bucket_agg["sell"]).abs()
    vpin = imbalance.rolling(n_buckets).sum() / bucket_agg["v"].rolling(n_buckets).sum()
    return vpin


# -----------------------------------------------------------------
# 22.6.6: Non-Uniform FFT for irregularly sampled time series
# -----------------------------------------------------------------
def nufft_simple(t, x, freqs):
    """
    Direct evaluation of the discrete Fourier transform at arbitrary
    times t (not the FFT, but mathematically equivalent for irregular grids).
    """
    t = np.asarray(t)
    x = np.asarray(x)
    F = np.array([np.sum(x * np.exp(-2j * np.pi * f * t)) for f in freqs])
    return F


# -----------------------------------------------------------------
# In-situ processing stub: compute summary stats as data streams in
# -----------------------------------------------------------------
class StreamingMoments:
    """Welford-style online mean / variance computation."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)


# -----------------------------------------------------------------
# MPI workflow stub (using mpi4py if available, otherwise sequential)
# -----------------------------------------------------------------
def mpi_workflow_stub(jobs, worker_func):
    """
    Distributes `jobs` to workers via mpi4py; falls back to serial if mpi4py
    is not installed.
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_jobs = jobs[rank::size]
        local_results = [worker_func(j) for j in local_jobs]
        all_results = comm.gather(local_results, root=0)
        if rank == 0:
            return [item for sublist in all_results for item in sublist]
        return None
    except ImportError:
        return [worker_func(j) for j in jobs]


# -----------------------------------------------------------------
# Demo
# -----------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)

    # VPIN calibration on synthetic ticks
    n = 5_000
    rets = pd.Series(rng.normal(0, 0.001, n))
    vols = pd.Series(rng.integers(1, 100, n))
    vp = calibrate_vpin(rets, vols, bucket_size=500, n_buckets=20).dropna()
    print(f"VPIN: {len(vp)} buckets, mean = {vp.mean():.3f}, "
          f"max = {vp.max():.3f}")

    # NUFFT
    t_irreg = np.sort(rng.uniform(0, 10, 100))
    x = np.sin(2 * np.pi * 1.0 * t_irreg) + 0.1 * rng.normal(size=100)
    freqs = np.linspace(0.1, 5.0, 50)
    F = nufft_simple(t_irreg, x, freqs)
    peak_freq = freqs[np.argmax(np.abs(F))]
    print(f"NUFFT peak frequency: {peak_freq:.2f} Hz (true=1.0)")

    # Streaming moments
    sm = StreamingMoments()
    for v in rng.normal(5, 2, 1000):
        sm.update(v)
    print(f"Streaming moments: mean={sm.mean:.3f}, std={sm.std:.3f}")


if __name__ == "__main__":
    main()
