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
import json
import time
from datetime import datetime
from pathlib import Path
from dataset_loaders import load_URM
from benchmark import benchmark_similarity, get_system_info


# ── Helpers ──────────────────────────────────────────────────────────────────

DEFAULT_VERSIONS = {
    'movielens': '32m',
    'yambda': '50m',
}


def parse_block_size(raw):
    """Parse the --block-size CLI value.

    Returns
    -------
    int or None
        None means disabled, 0 means auto, >0 means explicit size.
    """
    if raw.lower() == 'none':
        return None
    return int(raw)


def format_block_size(block_size):
    """Human-readable block_size string."""
    if block_size is None:
        return 'disabled'
    return 'auto' if block_size == 0 else str(block_size)


def format_time(computation_time, std_time):
    """Format time with optional ± std."""
    if std_time > 0:
        return f"{computation_time:.2f} ± {std_time:.2f}"
    return f"{computation_time:.2f}"


def build_report_stem(sys_info, config):
    """Build descriptive base filename (without extension) for reports.

    Format: result_{version}_{datasets}_{timestamp}[_{note}]
    Example: result_0.6.0_movielens_32m_20260224_144649
    Example: result_0.6.0_movielens_32m_20260224_144649_my_note
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets_str = "_".join(f"{d}_{v}" for d, v in config['datasets'])
    parts = ["result", sys_info['similaripy_version'], datasets_str, timestamp]
    if config.get('note'):
        safe = '_'.join(config['note'].lower().split())
        safe = ''.join(c for c in safe if c.isalnum() or c == '_')
        if safe:
            parts.append(safe)
    return "_".join(parts)


# ── Core benchmark runner ────────────────────────────────────────────────────

def run_benchmarks(config, verbose=True):
    """
    Run benchmarks on multiple datasets and similarity types.

    Parameters
    ----------
    config : dict
        Benchmark configuration with keys: datasets, similarities, k, shrink,
        threshold, num_threads, block_size, rounds, dataset_kwargs.
    verbose : bool
        Print progress.

    Returns
    -------
    tuple
        (all_results, dataset_info) where all_results is a nested dictionary
        and dataset_info contains URM statistics.
    """
    all_results = {}
    dataset_info = {}

    rounds = config['rounds']

    for dataset_name, version in config['datasets']:
        # Print loading header to terminal
        if verbose:
            print(f"\n{'='*70}")
            print(f"Loading dataset: {dataset_name.upper()} (version: {version})")
            print(f"{'='*70}")

        # Load dataset (not timed)
        load_start = time.perf_counter()
        URM, _metadata = load_URM(
            dataset_name,
            version=version,
            verbose=verbose,
            **config.get('dataset_kwargs', {}).get(dataset_name, {}),
        )
        load_time = time.perf_counter() - load_start

        # Store dataset info for report
        density = URM.nnz / (URM.shape[0] * URM.shape[1])
        dataset_info[(dataset_name, version)] = {
            'shape': URM.shape,
            'nnz': URM.nnz,
            'density': density,
        }

        # Print loading summary to terminal
        if verbose:
            print(f"Dataset loaded in {load_time:.2f}s")
            print(f"URM shape: {URM.shape}")
            print(f"URM nnz: {URM.nnz:,}")
            print(f"URM density: {density:.6%}")

        dataset_results = {}

        # Benchmark each similarity type
        for sim_type in config['similarities']:
            # Print similarity header to terminal
            if verbose:
                print(f"\n{'-'*70}")
                print(f"Benchmarking {sim_type.upper()}")
                print(f"{'-'*70}")

            # Run multiple rounds and collect results
            round_results = []
            for round_num in range(rounds):
                results = benchmark_similarity(
                    URM,
                    similarity_type=sim_type,
                    k=config['k'],
                    shrink=config['shrink'],
                    threshold=config['threshold'],
                    num_threads=config['num_threads'],
                    verbose=verbose,
                    block_size=config['block_size'],
                )
                round_results.append(results)

            # Aggregate results across rounds
            all_times = [r['computation_time'] for r in round_results]
            avg_time = sum(all_times) / rounds
            std_time = (
                (sum((t - avg_time) ** 2 for t in all_times) / rounds) ** 0.5
                if rounds > 1 else 0.0
            )
            avg_throughput = sum(r['throughput'] for r in round_results) / rounds

            dataset_results[sim_type] = {
                'similarity_matrix': round_results[-1]['similarity_matrix'],
                'computation_time': avg_time,
                'std_time': std_time,
                'n_items': round_results[0]['n_items'],
                'nnz': round_results[0]['nnz'],
                'density': round_results[0]['density'],
                'avg_neighbors': round_results[0]['avg_neighbors'],
                'throughput': avg_throughput,
                'rounds': rounds,
                'all_times': all_times,
            }

        all_results[(dataset_name, version)] = dataset_results

    return all_results, dataset_info


# ── Formatting ───────────────────────────────────────────────────────────────

def format_summary_table(all_results, sys_info, block_size):
    """
    Build the benchmark summary table as a string.

    Used by both terminal output and text report writing.

    Parameters
    ----------
    all_results : dict
        Results from run_benchmarks().
    sys_info : dict
        System info from get_system_info().
    block_size : int or None
        Block size parameter for display.

    Returns
    -------
    str
        Formatted summary table.
    """
    lines = []
    w = 130

    lines.append('=' * w)
    lines.append('BENCHMARK SUMMARY')
    lines.append('=' * w)
    lines.append(f"Date: {sys_info['timestamp']}")
    lines.append(f"SimilariPy version: {sys_info['similaripy_version']}")
    lines.append(f"NumPy version: {sys_info['numpy_version']}")
    lines.append(f"SciPy version: {sys_info['scipy_version']}")
    lines.append(f"Git commit: {sys_info['git_hash']}")
    lines.append(f"Python: {sys_info['python_version']}")
    lines.append(f"CPU: {sys_info['cpu_model']}")
    lines.append(f"Architecture: {sys_info['system']} {sys_info['arch']}")
    lines.append(f"CPU cores available: {sys_info['cpu_count']}")
    lines.append(f"Block size: {format_block_size(block_size)}")
    lines.append('=' * w)

    # Data rows, grouped by dataset
    sorted_results = sorted(all_results.items(), key=lambda x: (x[0][0], x[0][1]))

    for idx, ((dataset_name, version), dataset_results) in enumerate(sorted_results):
        # Dataset sub-header
        lines.append(f"\nDataset: {dataset_name} (version: {version})")

        header = (
            f"{'Similarity':<20} "
            f"{'Time (s)':<18} "
            f"{'Throughput':<15} "
            f"{'Output nnz':<15} "
            f"{'Avg Neighbors':<15} "
            f"{'Rounds':<10}"
        )
        lines.append(header)
        lines.append('-' * w)

        for sim_type in sorted(dataset_results.keys()):
            result = dataset_results[sim_type]
            time_str = format_time(result['computation_time'], result['std_time'])

            row = (
                f"{sim_type:<20} "
                f"{time_str:<18} "
                f"{result['throughput']:<15.1f} "
                f"{result['nnz']:<15,} "
                f"{result['avg_neighbors']:<15.1f} "
                f"{result['rounds']:<10}"
            )
            lines.append(row)

    lines.append('=' * w)
    return '\n'.join(lines)


def write_text_report(output_path, config, dataset_info, all_results, sys_info):
    """
    Write a human-readable text report.

    Parameters
    ----------
    output_path : Path
        Path to output file.
    config : dict
        Benchmark configuration.
    dataset_info : dict
        Dataset statistics.
    all_results : dict
        Benchmark results.
    sys_info : dict
        System info from get_system_info().
    """
    with open(output_path, 'w') as f:
        # Section 1: Configuration header
        f.write("=" * 70 + "\n")
        f.write("SIMILARIPY BENCHMARK SUITE\n")
        f.write("=" * 70 + "\n")
        f.write(f"Date: {sys_info['timestamp']}\n")
        ds, ver = config['datasets'][0]
        f.write(f"Dataset: {ds}:{ver}\n")
        f.write(f"Similarities: {', '.join(config['similarities'])}\n")
        f.write(f"Parameters: k={config['k']}, shrink={config['shrink']}, threshold={config['threshold']}\n")
        f.write(f"Threads: {config['num_threads'] if config['num_threads'] > 0 else 'auto'}\n")
        f.write(f"Block size: {format_block_size(config['block_size'])}\n")
        f.write(f"Rounds: {config['rounds']}\n")
        if config.get('note'):
            f.write(f"Note: {config['note']}\n")
        f.write("=" * 70 + "\n")

        # Section 2: Dataset statistics
        for (dataset_name, version), info in dataset_info.items():
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Loading dataset: {dataset_name.upper()} (version: {version})\n")
            f.write(f"{'=' * 70}\n")
            f.write(f"URM shape: {info['shape']}\n")
            f.write(f"URM nnz: {info['nnz']:,}\n")
            f.write(f"URM density: {info['density']:.6%}\n")

        # Section 3: Summary table (reuses the shared formatter)
        f.write('\n')
        f.write(format_summary_table(all_results, sys_info, config['block_size']))
        f.write('\n')


def write_json_report(output_path, config, dataset_info, all_results, sys_info):
    """
    Write benchmark results as a JSON file for machine-readable comparison.

    Parameters
    ----------
    output_path : Path
        Path to output JSON file.
    config : dict
        Benchmark configuration.
    dataset_info : dict
        Dataset statistics.
    all_results : dict
        Benchmark results.
    sys_info : dict
        System info from get_system_info().
    """
    report = {
        "metadata": {
            **sys_info,
            "note": config.get('note') or "",
        },
        "config": {
            "datasets": [[d, v] for d, v in config['datasets']],
            "similarities": config['similarities'],
            "k": config['k'],
            "shrink": config['shrink'],
            "threshold": config['threshold'],
            "num_threads": config['num_threads'],
            "block_size": format_block_size(config['block_size']),
            "rounds": config['rounds'],
        },
        "datasets": {},
        "results": {},
    }

    for (dataset_name, version), info in dataset_info.items():
        key = f"{dataset_name}:{version}"
        report["datasets"][key] = {
            "shape": list(info["shape"]),
            "nnz": info["nnz"],
            "density": info["density"],
        }

    for (dataset_name, version), dataset_results in all_results.items():
        key = f"{dataset_name}:{version}"
        report["results"][key] = {}
        for sim_type, result in dataset_results.items():
            report["results"][key][sim_type] = {
                "computation_time": round(result["computation_time"], 4),
                "std_time": round(result["std_time"], 4),
                "throughput": round(result["throughput"], 1),
                "nnz": result["nnz"],
                "avg_neighbors": round(result["avg_neighbors"], 1),
                "rounds": result["rounds"],
                "all_times": [round(t, 4) for t in result["all_times"]],
            }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)



# ── CLI ──────────────────────────────────────────────────────────────────────

def build_config(args):
    """
    Build a benchmark config dict from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Benchmark configuration dictionary.
    """
    version = args.version or DEFAULT_VERSIONS.get(args.dataset)
    datasets = [(args.dataset, version)]

    return {
        'datasets': datasets,
        'similarities': args.similarities,
        'k': args.k,
        'shrink': args.shrink,
        'threshold': args.threshold,
        'num_threads': args.threads,
        'block_size': parse_block_size(args.block_size),
        'rounds': args.rounds,
        'note': args.note,
        'dataset_kwargs': {
            'yambda': {'event_type': args.event_type},
        },
    }


def print_config_header(config):
    """Print the benchmark configuration banner to terminal."""
    print("=" * 70)
    print("SIMILARIPY BENCHMARK SUITE")
    print("=" * 70)
    ds, ver = config['datasets'][0]
    print(f"Dataset: {ds}:{ver}")
    print(f"Similarities: {', '.join(config['similarities'])}")
    print(f"Parameters: k={config['k']}, shrink={config['shrink']}, threshold={config['threshold']}")
    print(f"Threads: {config['num_threads'] if config['num_threads'] > 0 else 'auto'}")
    print(f"Block size: {format_block_size(config['block_size'])}")
    print(f"Rounds: {config['rounds']}")
    if config.get('note'):
        print(f"Note: {config['note']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmark script for similaripy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark on MovieLens 32M with default similarities
  python run_benchmarks.py --dataset movielens --version 32m

  # Benchmark specific similarities
  python run_benchmarks.py --dataset movielens --similarities cosine rp3beta

  # Custom parameters with note
  python run_benchmarks.py --dataset movielens --k 200 --shrink 10 --note "baseline"
        """
    )

    parser.add_argument('--dataset', default='movielens',
                        choices=['movielens', 'yambda'],
                        help='Dataset to benchmark (default: movielens)')
    parser.add_argument('--version', type=str, default=None,
                        help='Dataset version (e.g., "25m", "32m" for MovieLens). Default: dataset default')
    parser.add_argument('--similarities', nargs='+',
                        default=['dot_product', 'cosine', 'rp3beta'],
                        help='Similarity types to benchmark (default: dot_product cosine rp3beta)')
    parser.add_argument('--k', type=int, default=100,
                        help='Number of top similar items to keep (default: 100)')
    parser.add_argument('--shrink', type=float, default=0,
                        help='Shrinkage parameter (default: 0)')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Minimum similarity threshold (default: 0)')
    parser.add_argument('--threads', type=int, default=0,
                        help='Number of threads (0 = auto) (default: 0)')
    parser.add_argument('--rounds', type=int, default=1,
                        help='Number of rounds to run each config (results averaged) (default: 1)')
    parser.add_argument('--block-size', type=str, default='0',
                        help='Block size: 0=auto, none=disabled, or int>0 (default: 0)')
    parser.add_argument('--event-type', type=str, default='multi_event',
                        choices=['likes', 'listens', 'multi_event'],
                        help='Yambda event type (default: multi_event)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output during benchmarking')
    parser.add_argument('--output-dir', type=str, default='bench_results',
                        help='Output directory for benchmark results (default: bench_results)')
    parser.add_argument('--note', type=str, default=None,
                        help='Optional note/comment to attach to the benchmark report')

    args = parser.parse_args()
    config = build_config(args)

    # Print header & run
    print_config_header(config)

    all_results, dataset_info = run_benchmarks(config, verbose=not args.quiet)

    sys_info = get_system_info()

    # Print summary to terminal
    print()
    print(format_summary_table(all_results, sys_info, config['block_size']))

    # Write reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_filename = build_report_stem(sys_info, config)
    txt_path = output_dir / f"{base_filename}.txt"
    json_path = output_dir / f"{base_filename}.json"

    write_text_report(txt_path, config, dataset_info, all_results, sys_info)
    write_json_report(json_path, config, dataset_info, all_results, sys_info)

    print(f"\nResults saved to:")
    print(f"  Text:  {txt_path}")
    print(f"  JSON:  {json_path}")


if __name__ == "__main__":
    main()