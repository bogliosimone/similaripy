"""
Common benchmarking functions for similaripy performance testing.

This module provides standardized functions for benchmarking similarity
computation across different datasets and versions.
"""

import time
import platform
import os
import subprocess
import sys
import threading
from datetime import datetime
import similaripy as sim
import numpy as np
import scipy as sp


def get_system_info():
    """Collect system and environment information for benchmark reports.

    Returns
    -------
    dict
        Dictionary with keys: similaripy_version, numpy_version, scipy_version, python_version, system,
        arch, cpu_model, cpu_count, git_hash, timestamp, os_version, memory_gb,
        rustc, gil_enabled.
    """
    try:
        similaripy_version = sim.__version__
    except AttributeError:
        similaripy_version = "unknown"

    try:
        numpy_version = np.__version__
    except AttributeError:
        numpy_version = "unknown"
    
    try:        
        scipy_version = sp.__version__
    except AttributeError:        
        scipy_version = "unknown"

    # CPU model name
    cpu_model = "unknown"
    try:
        system = platform.system()
        if system == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                cpu_model = r.stdout.strip()
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        cpu_model = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass

    # Git commit hash
    git_hash = "unknown"
    try:
        r = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    os_version = platform.platform()
    if platform.system() == "Darwin":
        os_version = f"macOS {platform.mac_ver()[0]}"

    memory_gb = "unknown"
    try:
        memory_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass

    def tool_version(cmd):
        try:
            r = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                return r.stdout.splitlines()[0].strip()
        except Exception:
            pass
        return "unknown"

    rustc = tool_version("rustc")

    return {
        "similaripy_version": similaripy_version,
        "numpy_version": numpy_version,
        "scipy_version": scipy_version,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count() or "unknown",
        "git_hash": git_hash,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os_version": os_version,
        "memory_gb": memory_gb,
        "rustc": rustc,
        "gil_enabled": getattr(sys, "_is_gil_enabled", lambda: True)(),
    }


def benchmark_similarity(
    URM,
    similarity_type="cosine",
    k=100,
    shrink=0,
    threshold=0,
    num_threads=0,
    verbose=True,
    **similarity_kwargs
):
    """
    Benchmark similarity computation on a User Rating Matrix (URM).

    This function computes item-item similarity and measures performance metrics.

    Parameters
    ----------
    URM : scipy.sparse.csr_array
        User-item rating matrix (users x items)
    similarity_type : str, optional
        Type of similarity to compute: "cosine", "asymmetric_cosine",
        "jaccard", "dice", "tversky", "p3alpha", "rp3beta", "splus"
        (default: "cosine")
    k : int, optional
        Number of top similar items to keep for each item (default: 100)
    shrink : float, optional
        Shrinkage parameter (default: 0)
    threshold : float, optional
        Minimum similarity threshold (default: 0)
    num_threads : int, optional
        Number of threads for parallel computation (0 = auto) (default: 0)
    verbose : bool, optional
        Unused for timing: the similarity call always runs with verbose=False
        so the progress bar stays out of the timed region (default: True)
    **similarity_kwargs : dict
        Additional similarity-specific parameters

    Returns
    -------
    dict
        Benchmark results with keys:
        - 'similarity_matrix': computed similarity matrix
        - 'computation_time': time taken for similarity computation (seconds)
        - 'kernel_time': time spent inside the compute kernel (seconds)
        - 'n_items': number of items
        - 'nnz': number of non-zero entries in similarity matrix
        - 'density': similarity matrix density
        - 'avg_neighbors': average number of neighbors per item
        - 'throughput': items processed per second
        - 'similarity_type': type of similarity computed
        - 'k': k parameter used
        - 'shrink': shrink parameter used
        - 'threshold': threshold parameter used

    Examples
    --------
    >>> from dataset_loaders import load_URM
    >>> from benchmark import benchmark_similarity
    >>>
    >>> # Load dataset
    >>> URM, meta = load_URM("movielens", version="25m")
    >>>
    >>> # Benchmark cosine similarity
    >>> results = benchmark_similarity(
    ...     URM,
    ...     similarity_type="cosine",
    ...     k=100,
    ...     shrink=0,
    ...     verbose=True
    ... )
    >>>
    >>> print(f"Computation time: {results['computation_time']:.2f}s")
    >>> print(f"Throughput: {results['throughput']:.1f} items/s")
    """
    # Transpose for item-item similarity
    item_matrix = URM.T

    # Select similarity function
    similarity_func = _get_similarity_function(similarity_type)

    block_size = similarity_kwargs.pop('block_size', 0)

    start_time = time.perf_counter()

    similarity_matrix = similarity_func(
        item_matrix,
        k=k,
        shrink=shrink,
        threshold=threshold,
        verbose=False,
        num_threads=num_threads,
        block_size=block_size,
        **similarity_kwargs
    )

    end_time = time.perf_counter()
    computation_time = end_time - start_time
    kernel_time = sim._s_plus._last_kernel_time

    # Calculate metrics
    n_items = similarity_matrix.shape[0]
    nnz = similarity_matrix.nnz
    density = nnz / (n_items * n_items)
    avg_neighbors = nnz / n_items
    throughput = n_items / computation_time

    # Validate results
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], \
        "Similarity matrix should be square"
    assert similarity_matrix.shape[0] == n_items, \
        "Similarity matrix should have same dimension as number of items"
    assert nnz > 0, \
        "Similarity matrix should have non-zero entries"

    results = {
        'similarity_matrix': similarity_matrix,
        'computation_time': computation_time,
        'kernel_time': kernel_time,
        'n_items': n_items,
        'nnz': nnz,
        'density': density,
        'avg_neighbors': avg_neighbors,
        'throughput': throughput,
        'similarity_type': similarity_type,
        'k': k,
        'shrink': shrink,
        'threshold': threshold,
        'block_size': block_size,
    }

    return results


def peak_rss(func):
    """Run func() while sampling the process memory every 1 ms.

    ru_maxrss can't be reset and the dataset load peaks higher than a call,
    so the per-call peak is sampled. Spikes shorter than ~1 ms are missed;
    irrelevant for the GB-scale temporaries this is meant to catch.

    On macOS this samples phys_footprint (Activity Monitor's "Memory"), not
    RSS: freed malloc pages stay in RSS, so after the dataset load new
    temporaries reuse resident pages and RSS barely moves (+1 MB measured for
    an ML-32M call). Elsewhere: psutil RSS.

    Returns
    -------
    tuple
        (func's result, memory before the call in MB, peak above it in MB)
    """
    if platform.system() == "Darwin":
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        buf = ctypes.create_string_buffer(256)
        pid = os.getpid()

        def current():
            # rusage_info_v0: uuid[16], then 7 uint64 before ri_phys_footprint
            if libc.proc_pid_rusage(pid, 0, buf) != 0:
                raise OSError("proc_pid_rusage failed")
            return int.from_bytes(buf.raw[72:80], "little")
    else:
        import psutil  # only needed for --rss

        proc = psutil.Process()

        def current():
            return proc.memory_info().rss

    base = peak = current()
    done = threading.Event()

    def sample():
        nonlocal peak
        while not done.wait(0.001):
            peak = max(peak, current())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        out = func()
    finally:
        done.set()
        sampler.join()
    peak = max(peak, current())
    return out, base / 2**20, (peak - base) / 2**20


def _get_similarity_function(similarity_type):
    """Get similarity function from similaripy."""
    similarity_map = {
        'dot_product': sim.dot_product,
        'cosine': sim.cosine,
        'asymmetric_cosine': sim.asymmetric_cosine,
        'jaccard': sim.jaccard,
        'dice': sim.dice,
        'tversky': sim.tversky,
        'p3alpha': sim.p3alpha,
        'rp3beta': sim.rp3beta,
        'splus': sim.s_plus,
    }

    if similarity_type not in similarity_map:
        raise ValueError(
            f"Unknown similarity type '{similarity_type}'. "
            f"Available: {list(similarity_map.keys())}"
        )

    return similarity_map[similarity_type]
