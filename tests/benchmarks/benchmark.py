"""
Common benchmarking functions for similaripy performance testing.

This module provides standardized functions for benchmarking similarity
computation across different datasets and versions.
"""

import time
import platform
import os
import numpy as np
import similaripy as sim


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
    URM : scipy.sparse.csr_matrix
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

    start_time = time.perf_counter()

    similarity_matrix = similarity_func(
        item_matrix,
        k=k,
        shrink=shrink,
        threshold=threshold,
        verbose=verbose,
        num_threads=num_threads,
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


def profile_similarities(
    URM,
    similarity_types=("dot_product", "cosine", "rp3beta"),
    k=100,
    shrink=0,
    verbose=True,
    **kwargs
):
    """
    Profile multiple similarity algorithms on the same dataset.

    Parameters
    ----------
    URM : scipy.sparse.csr_matrix
        User-item rating matrix
    similarity_types : tuple or list, optional
        List of similarity types to profile (default: ("dot_product", "cosine", "rp3beta"))
    k : int, optional
        Number of top similar items to keep (default: 100)
    shrink : float, optional
        Shrinkage parameter (default: 0)
    verbose : bool, optional
        Print progress and results (default: True)
    **kwargs : dict
        Additional parameters passed to benchmark_similarity

    Returns
    -------
    dict
        Dictionary mapping similarity type to benchmark results

    Examples
    --------
    >>> from dataset_loaders import load_URM
    >>> from benchmark import profile_similarities
    >>>
    >>> URM, meta = load_URM("movielens", version="25m")
    >>> results = profile_similarities(
    ...     URM,
    ...     similarity_types=("dot_product", "cosine", "rp3beta"),
    ...     k=100
    ... )
    >>>
    >>> for sim_type, result in results.items():
    ...     print(f"{sim_type}: {result['computation_time']:.2f}s")
    """
    results = {}

    for sim_type in similarity_types:
        result = benchmark_similarity(
            URM,
            similarity_type=sim_type,
            k=k,
            shrink=shrink,
            verbose=verbose,
            **kwargs
        )
        results[sim_type] = result

    if verbose and len(similarity_types) > 1:
        # Get system information
        try:
            similaripy_version = sim.__version__
        except AttributeError:
            similaripy_version = "unknown"

        arch = platform.machine()
        system = platform.system()
        cpu_count = os.cpu_count() or "unknown"

        print(f"\n{'='*60}")
        print(f"Profiling Summary")
        print(f"{'='*60}")
        print(f"Similaripy version: {similaripy_version}")
        print(f"Architecture: {system} {arch}")
        print(f"CPU cores available: {cpu_count}")
        print(f"{'-'*60}")
        print(f"{'Similarity':<20} {'Time (s)':<12} {'Throughput (items/s)':<20}")
        print(f"{'-'*60}")
        for sim_type, result in results.items():
            print(f"{sim_type:<20} {result['computation_time']:<12.2f} {result['throughput']:<20.1f}")
        print(f"{'='*60}\n")

    return results
