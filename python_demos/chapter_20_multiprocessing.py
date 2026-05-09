"""
Chapter 20: Multiprocessing and Vectorization
==============================================

Demonstrates:
  * Vectorization vs. for-loop performance
  * Single-thread, multi-thread, multi-process comparison
  * Snippet 20.5: linParts for linear partitioning
  * Snippet 20.6: nestedParts for two-nested-loops partitioning
  * Snippet 20.7: mpPandasObj wrapper for parallel DataFrame ops
"""

import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor


# -----------------------------------------------------------------
# 20.2: Vectorization example
# -----------------------------------------------------------------
def loop_squared_diff(a, b):
    out = []
    for i in range(len(a)):
        out.append((a[i] - b[i]) ** 2)
    return np.array(out)


def vec_squared_diff(a, b):
    return (a - b) ** 2


# -----------------------------------------------------------------
# Snippet 20.5: linear partitions
# -----------------------------------------------------------------
def lin_parts(num_atoms, num_threads):
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# -----------------------------------------------------------------
# Snippet 20.6: nested partitions for upper-triangular workloads
# -----------------------------------------------------------------
def nested_parts(num_atoms, num_threads, upper_triang=False):
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)
    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] +
                          num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + np.sqrt(part)) / 2.0
        parts.append(part)
    parts = np.round(parts).astype(int)
    if upper_triang:
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
    return parts


# -----------------------------------------------------------------
# Snippet 20.7: mpPandasObj wrapper
# -----------------------------------------------------------------
def mp_pandas_obj(func, pd_obj, num_threads=4, mp_batches=1, lin_mols=True,
                   **kwargs):
    """
    func: callable that processes molecule
    pd_obj[0]: name of argument that takes molecules
    pd_obj[1]: array of indices to be passed
    """
    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)
    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], "func": func}
        job.update(kwargs)
        jobs.append(job)
    if num_threads == 1:
        out = process_jobs_sequential(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)
    if isinstance(out[0], pd.DataFrame):
        return pd.concat(out, sort=False)
    elif isinstance(out[0], pd.Series):
        return pd.concat(out)
    return out


def process_jobs_sequential(jobs):
    return [expand_call(job) for job in jobs]


def process_jobs(jobs, num_threads):
    with Pool(processes=num_threads) as pool:
        return pool.map(expand_call, jobs)


def expand_call(kwargs):
    func = kwargs["func"]
    del kwargs["func"]
    return func(**kwargs)


# -----------------------------------------------------------------
# Demo: vectorization speed
# -----------------------------------------------------------------
def main():
    a = np.random.normal(size=1_000_000)
    b = np.random.normal(size=1_000_000)
    t0 = time.time()
    _ = loop_squared_diff(a[:10_000], b[:10_000])
    t_loop = time.time() - t0
    t0 = time.time()
    _ = vec_squared_diff(a, b)
    t_vec = time.time() - t0
    print(f"For-loop on 10k items: {t_loop:.3f}s")
    print(f"Vectorized on 1M items: {t_vec:.3f}s")
    print(f"Speedup: ~{t_loop / t_vec * 100:.0f}x per element")

    # Linear partitioning
    parts = lin_parts(num_atoms=23, num_threads=4)
    print(f"\nLinear partitions (23 atoms / 4 threads): {parts}")
    # Nested partitions
    parts2 = nested_parts(num_atoms=23, num_threads=4, upper_triang=False)
    print(f"Nested partitions: {parts2}")


if __name__ == "__main__":
    main()
