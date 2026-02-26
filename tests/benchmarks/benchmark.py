"""
Common benchmarking functions for similaripy performance testing.

This module provides standardized functions for benchmarking similarity
computation across different datasets and versions.
"""

import time
import platform
import os
import subprocess
from datetime import datetime
import similaripy as sim


def get_system_info():
    """Collect system and environment information for benchmark reports.

    Returns
    -------
    dict
        Dictionary with keys: similaripy_version, python_version, system,
        arch, cpu_model, cpu_count, git_hash, timestamp.
    """
    try:
        similaripy_version = sim.__version__
    except AttributeError:
        similaripy_version = "unknown"

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
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            git_hash = r.stdout.strip()
    except Exception:
        pass

    return {
        "similaripy_version": similaripy_version,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "arch": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count() or "unknown",
        "git_hash": git_hash,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        Print progress and results (default: True)
    **similarity_kwargs : dict
        Additional similarity-specific parameters

    Returns
    -------
    dict
        Benchmark results with keys:
        - 'similarity_matrix': computed similarity matrix
        - 'computation_time': time taken for similarity computation (seconds)
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
        verbose=verbose,
        num_threads=num_threads,
        block_size=block_size,
        **similarity_kwargs
    )

    end_time = time.perf_counter()
    computation_time = end_time - start_time

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
