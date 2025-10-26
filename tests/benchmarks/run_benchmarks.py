"""
Unified benchmark script for similaripy similarity algorithms.

This script allows you to:
1. Benchmark single or multiple similarity algorithms
2. Test on multiple datasets (MovieLens, Yambda)
3. Get a comprehensive summary table of all results

Usage:
    # Benchmark all default similarities on MovieLens 32M
    python run_benchmarks.py --dataset movielens --version 32m

    # Benchmark specific similarities on multiple datasets
    python run_benchmarks.py --dataset movielens yambda --similarities cosine rp3beta

    # Benchmark with custom parameters
    python run_benchmarks.py --dataset movielens --version 25m --k 200 --shrink 10

Requirements:
    pip install -e ".[bench]"
"""

import argparse
import time
import platform
import os
from datetime import datetime
from pathlib import Path
from dataset_loaders import load_URM
from benchmark import benchmark_similarity
import similaripy as sim


def run_benchmarks(datasets, similarity_types, k=100, shrink=0, threshold=0,
                   num_threads=0, rounds=1, verbose=True, **dataset_kwargs):
    """
    Run benchmarks on multiple datasets and similarity types.

    Parameters
    ----------
    datasets : list of tuples
        List of (dataset_name, version) tuples
    similarity_types : list of str
        List of similarity types to benchmark
    k : int
        Number of top similar items to keep
    shrink : float
        Shrinkage parameter
    threshold : float
        Minimum similarity threshold
    num_threads : int
        Number of threads (0 = auto)
    rounds : int
        Number of times to run each configuration (results will be averaged)
    verbose : bool
        Print progress
    **dataset_kwargs : dict
        Additional dataset-specific parameters

    Returns
    -------
    tuple
        (all_results, dataset_info) where all_results is a nested dictionary
        and dataset_info contains URM statistics
    """
    all_results = {}
    dataset_info = {}

    for dataset_name, version in datasets:
        # Print loading header to terminal
        if verbose:
            print(f"\n{'='*70}")
            print(f"Loading dataset: {dataset_name.upper()} (version: {version})")
            print(f"{'='*70}")

        # Load dataset (not timed)
        load_start = time.perf_counter()
        URM, metadata = load_URM(
            dataset_name,
            version=version,
            verbose=verbose,
            **dataset_kwargs.get(dataset_name, {})
        )
        load_time = time.perf_counter() - load_start

        # Store dataset info for report
        density = URM.nnz / (URM.shape[0] * URM.shape[1])
        dataset_info[(dataset_name, version)] = {
            'shape': URM.shape,
            'nnz': URM.nnz,
            'density': density
        }

        # Print loading summary to terminal
        if verbose:
            print(f"Dataset loaded in {load_time:.2f}s")
            print(f"URM shape: {URM.shape}")
            print(f"URM nnz: {URM.nnz:,}")
            print(f"URM density: {density:.6%}")

        dataset_results = {}

        # Benchmark each similarity type
        for sim_type in similarity_types:
            # Print similarity header to terminal
            if verbose:
                print(f"\n{'-'*70}")
                print(f"Benchmarking {sim_type.upper()}")
                if rounds > 1:
                    print(f"Running {rounds} rounds...")
                print(f"{'-'*70}")

            # Run multiple rounds and collect results
            round_results = []
            for round_num in range(rounds):
                if verbose and rounds > 1:
                    print(f"Round {round_num + 1}/{rounds}")

                results = benchmark_similarity(
                    URM,
                    similarity_type=sim_type,
                    k=k,
                    shrink=shrink,
                    threshold=threshold,
                    num_threads=num_threads,
                    verbose=verbose if rounds == 1 else False
                )
                round_results.append(results)

            # Average the results across rounds
            if rounds > 1:
                avg_results = {
                    'similarity_matrix': round_results[-1]['similarity_matrix'],  # Use last round's matrix
                    'computation_time': sum(r['computation_time'] for r in round_results) / rounds,
                    'n_items': round_results[0]['n_items'],
                    'nnz': round_results[0]['nnz'],
                    'density': round_results[0]['density'],
                    'avg_neighbors': round_results[0]['avg_neighbors'],
                    'throughput': sum(r['throughput'] for r in round_results) / rounds,
                    'similarity_type': sim_type,
                    'k': k,
                    'shrink': shrink,
                    'threshold': threshold,
                    'rounds': rounds,
                    'all_times': [r['computation_time'] for r in round_results],
                    'std_time': (sum((r['computation_time'] - sum(r2['computation_time'] for r2 in round_results) / rounds) ** 2 for r in round_results) / rounds) ** 0.5,
                }
                dataset_results[sim_type] = avg_results

                # Print average to terminal
                if verbose:
                    print(f"\n  Average time: {avg_results['computation_time']:.2f}s ± {avg_results['std_time']:.2f}s")
                    print(f"  Average throughput: {avg_results['throughput']:.1f} items/s")
            else:
                dataset_results[sim_type] = round_results[0]

        all_results[(dataset_name, version)] = dataset_results

    return all_results, dataset_info


def print_summary_table(all_results):
    """
    Print a comprehensive summary table of all benchmark results.

    Parameters
    ----------
    all_results : dict
        Results from run_benchmarks()
    """
    # Get system information
    try:
        similaripy_version = sim.__version__
    except AttributeError:
        similaripy_version = "unknown"

    arch = platform.machine()
    system = platform.system()
    cpu_count = os.cpu_count() or "unknown"

    # Check if any results have multiple rounds
    has_rounds = any(
        'rounds' in result and result['rounds'] > 1
        for dataset_results in all_results.values()
        for result in dataset_results.values()
    )

    print(f"\n{'='*120}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*120}")
    print(f"Similaripy version: {similaripy_version}")
    print(f"Architecture: {system} {arch}")
    print(f"CPU cores available: {cpu_count}")
    print(f"{'='*120}")

    # Table header
    if has_rounds:
        header = f"{'Dataset':<20} {'Version':<10} {'Similarity':<20} {'Time (s)':<18} {'Throughput':<15} {'Avg Neighbors':<15} {'Rounds':<10}"
    else:
        header = f"{'Dataset':<20} {'Version':<10} {'Similarity':<20} {'Time (s)':<12} {'Throughput':<15} {'Avg Neighbors':<15}"
    print(header)
    print('-' * 120)

    # Sort results by dataset and similarity
    sorted_results = sorted(all_results.items(), key=lambda x: (x[0][0], x[0][1]))

    for (dataset_name, version), dataset_results in sorted_results:
        first_row = True
        for sim_type in sorted(dataset_results.keys()):
            result = dataset_results[sim_type]

            dataset_col = f"{dataset_name}" if first_row else ""
            version_col = f"{version}" if first_row else ""

            # Format time with std if available
            if 'std_time' in result:
                time_str = f"{result['computation_time']:.2f} ± {result['std_time']:.2f}"
            else:
                time_str = f"{result['computation_time']:.2f}"

            if has_rounds:
                rounds_str = f"{result.get('rounds', 1)}"
                row = (
                    f"{dataset_col:<20} "
                    f"{version_col:<10} "
                    f"{sim_type:<20} "
                    f"{time_str:<18} "
                    f"{result['throughput']:<15.1f} "
                    f"{result['avg_neighbors']:<15.1f} "
                    f"{rounds_str:<10}"
                )
            else:
                row = (
                    f"{dataset_col:<20} "
                    f"{version_col:<10} "
                    f"{sim_type:<20} "
                    f"{time_str:<12} "
                    f"{result['throughput']:<15.1f} "
                    f"{result['avg_neighbors']:<15.1f}"
                )
            print(row)
            first_row = False

        # Add separator between datasets
        if sorted_results[-1][0] != (dataset_name, version):
            print('-' * 120)

    print('=' * 120)


def write_report_file(output_path, datasets, similarities, k, shrink, threshold, num_threads, rounds,
                      dataset_info, all_results):
    """
    Write a clean report file with benchmark configuration and results.

    Parameters
    ----------
    output_path : Path
        Path to output file
    datasets : list
        List of (dataset_name, version) tuples
    similarities : list
        List of similarity types
    k, shrink, threshold, num_threads, rounds : parameters
        Benchmark parameters
    dataset_info : dict
        Dataset statistics
    all_results : dict
        Benchmark results
    """
    # Get system information
    try:
        similaripy_version = sim.__version__
    except AttributeError:
        similaripy_version = "unknown"

    arch = platform.machine()
    system = platform.system()
    cpu_count = os.cpu_count() or "unknown"

    with open(output_path, 'w') as f:
        # Section 1: Configuration
        f.write("=" * 70 + "\n")
        f.write("SIMILARIPY BENCHMARK SUITE\n")
        f.write("=" * 70 + "\n")
        f.write(f"Datasets: {', '.join([f'{d}:{v}' for d, v in datasets])}\n")
        f.write(f"Similarities: {', '.join(similarities)}\n")
        f.write(f"Parameters: k={k}, shrink={shrink}, threshold={threshold}\n")
        f.write(f"Threads: {num_threads if num_threads > 0 else 'auto'}\n")
        f.write(f"Rounds: {rounds}\n")
        f.write("=" * 70 + "\n")

        # Dataset statistics
        for (dataset_name, version), info in dataset_info.items():
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Loading dataset: {dataset_name.upper()} (version: {version})\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"URM shape: {info['shape']}\n")
            f.write(f"URM nnz: {info['nnz']:,}\n")
            f.write(f"URM density: {info['density']:.6%}\n")

        # Section 2: Benchmark Summary
        f.write(f"\n{'=' * 120}\n")
        f.write("BENCHMARK SUMMARY\n")
        f.write(f"{'=' * 120}\n")
        f.write(f"Similaripy version: {similaripy_version}\n")
        f.write(f"Architecture: {system} {arch}\n")
        f.write(f"CPU cores available: {cpu_count}\n")
        f.write(f"{'=' * 120}\n")

        # Check if any results have multiple rounds
        has_rounds = any(
            'rounds' in result and result['rounds'] > 1
            for dataset_results in all_results.values()
            for result in dataset_results.values()
        )

        # Table header
        if has_rounds:
            f.write(f"{'Dataset':<20} {'Version':<10} {'Similarity':<20} {'Time (s)':<18} {'Throughput':<15} {'Avg Neighbors':<15} {'Rounds':<10}\n")
        else:
            f.write(f"{'Dataset':<20} {'Version':<10} {'Similarity':<20} {'Time (s)':<12} {'Throughput':<15} {'Avg Neighbors':<15}\n")
        f.write('-' * 120 + "\n")

        # Sort results by dataset and similarity
        sorted_results = sorted(all_results.items(), key=lambda x: (x[0][0], x[0][1]))

        for (dataset_name, version), dataset_results in sorted_results:
            first_row = True
            for sim_type in sorted(dataset_results.keys()):
                result = dataset_results[sim_type]

                dataset_col = f"{dataset_name}" if first_row else ""
                version_col = f"{version}" if first_row else ""

                # Format time with std if available
                if 'std_time' in result:
                    time_str = f"{result['computation_time']:.2f} ± {result['std_time']:.2f}"
                else:
                    time_str = f"{result['computation_time']:.2f}"

                if has_rounds:
                    rounds_str = f"{result.get('rounds', 1)}"
                    row = (
                        f"{dataset_col:<20} "
                        f"{version_col:<10} "
                        f"{sim_type:<20} "
                        f"{time_str:<18} "
                        f"{result['throughput']:<15.1f} "
                        f"{result['avg_neighbors']:<15.1f} "
                        f"{rounds_str:<10}"
                    )
                else:
                    row = (
                        f"{dataset_col:<20} "
                        f"{version_col:<10} "
                        f"{sim_type:<20} "
                        f"{time_str:<12} "
                        f"{result['throughput']:<15.1f} "
                        f"{result['avg_neighbors']:<15.1f}"
                    )
                f.write(row + "\n")
                first_row = False

            # Add separator between datasets
            if sorted_results[-1][0] != (dataset_name, version):
                f.write('-' * 120 + "\n")

        f.write('=' * 120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmark script for similaripy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark on MovieLens 32M with default similarities
  python run_benchmarks.py --dataset movielens --version 32m

  # Benchmark specific similarities on multiple datasets
  python run_benchmarks.py --dataset movielens yambda --similarities cosine rp3beta

  # Custom parameters
  python run_benchmarks.py --dataset movielens --k 200 --shrink 10 --threads 4
        """
    )

    parser.add_argument(
        '--dataset',
        nargs='+',
        default=['movielens'],
        choices=['movielens', 'yambda'],
        help='Dataset(s) to benchmark (default: movielens)'
    )

    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Dataset version (e.g., "25m", "32m" for MovieLens; "50m", "500m" for Yambda). Default: dataset default'
    )

    parser.add_argument(
        '--similarities',
        nargs='+',
        default=['dot_product', 'cosine', 'rp3beta'],
        help='Similarity types to benchmark (default: dot_product cosine rp3beta)'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help='Number of top similar items to keep (default: 100)'
    )

    parser.add_argument(
        '--shrink',
        type=float,
        default=0,
        help='Shrinkage parameter (default: 0)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0,
        help='Minimum similarity threshold (default: 0)'
    )

    parser.add_argument(
        '--threads',
        type=int,
        default=0,
        help='Number of threads (0 = auto) (default: 0)'
    )

    parser.add_argument(
        '--rounds',
        type=int,
        default=1,
        help='Number of times to run each configuration (results will be averaged) (default: 1)'
    )

    parser.add_argument(
        '--event-type',
        type=str,
        default='multi_event',
        choices=['likes', 'listens', 'multi_event'],
        help='Yambda event type (default: multi_event)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output during benchmarking'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='bench_results',
        help='Output directory for benchmark results (default: bench_results)'
    )

    args = parser.parse_args()

    # Prepare datasets list
    datasets = []
    for dataset_name in args.dataset:
        if args.version:
            version = args.version
        elif dataset_name == 'movielens':
            version = '32m'
        elif dataset_name == 'yambda':
            version = '50m'
        else:
            version = None

        datasets.append((dataset_name, version))

    # Prepare dataset-specific kwargs
    dataset_kwargs = {
        'yambda': {'event_type': args.event_type}
    }

    # Print benchmark configuration header
    print("="*70)
    print("SIMILARIPY BENCHMARK SUITE")
    print("="*70)
    print(f"Datasets: {', '.join([f'{d}:{v}' for d, v in datasets])}")
    print(f"Similarities: {', '.join(args.similarities)}")
    print(f"Parameters: k={args.k}, shrink={args.shrink}, threshold={args.threshold}")
    print(f"Threads: {args.threads if args.threads > 0 else 'auto'}")
    print(f"Rounds: {args.rounds}")
    print("="*70)

    # Run benchmarks
    all_results, dataset_info = run_benchmarks(
        datasets=datasets,
        similarity_types=args.similarities,
        k=args.k,
        shrink=args.shrink,
        threshold=args.threshold,
        num_threads=args.threads,
        rounds=args.rounds,
        verbose=not args.quiet,
        dataset_kwargs=dataset_kwargs
    )

    # Print summary table to terminal
    print_summary_table(all_results)

    # Write report file
    try:
        similaripy_version = sim.__version__
    except AttributeError:
        similaripy_version = "unknown"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"result_{similaripy_version}_{timestamp}.txt"
    output_path = output_dir / output_filename

    write_report_file(
        output_path=output_path,
        datasets=datasets,
        similarities=args.similarities,
        k=args.k,
        shrink=args.shrink,
        threshold=args.threshold,
        num_threads=args.threads,
        rounds=args.rounds,
        dataset_info=dataset_info,
        all_results=all_results
    )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
