"""
Small-workload benchmark preset for similaripy.

Calls on small inputs take milliseconds, so fixed per-call costs (Python prep,
binding, thread start-up, output build) matter here and stay hidden on
MovieLens/Yambda. Each case runs on a seeded synthetic matrix; one timing is the
median of --reps calls, and rounds are interleaved across cases (ABAB).

Cases cover parameters the large datasets don't: k=10/1000, binary=True,
block_size=None, a small explicit block_size (blocked path), array- and
matrix-mode filter_cols/target_cols, and two concurrent calls
(num_threads=2 each) vs the same two calls run serially. Run the concurrency
case on a free-threaded build (3.14t/3.15t) to get a meaningful number;
the report records gil_enabled.

Usage:
    python run_small_benchmarks.py                 # smoke set, 3 rounds
    python run_small_benchmarks.py --cases all     # every case, incl. concurrency
    python run_small_benchmarks.py --kernel cpp rust
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import similaripy as sim
from similaripy.cython_code import s_plus as _s_plus_module
from benchmark import get_system_info
from run_benchmarks import (
    aggregate_rounds,
    build_report_stem,
    format_summary_table,
    result_key,
    write_json_report,
    write_text_report,
)


DATASET = ('synthetic', '10k')

# Default smoke set: catches big (>20%) regressions; the rest via --cases all
SMOKE_CASES = ['dot_product', 'cosine', 'cosine_k1000', 'cosine_filter_matrix', 'cosine_block2048']


def make_urm(n_users=10_000, n_items=10_000, nnz=1_000_000, zipf_a=0.8, seed=42):
    """Seeded users x items matrix with Zipf-like item popularity, ratings 1..5."""
    rng = np.random.default_rng(seed)
    popularity = 1.0 / np.arange(1, n_items + 1) ** zipf_a
    items = rng.choice(n_items, size=nnz, p=popularity / popularity.sum())
    # Shuffle item ids so popularity isn't already sorted by column
    items = rng.permutation(n_items)[items]
    users = rng.integers(0, n_users, size=nnz)
    ratings = rng.integers(1, 6, size=nnz).astype(np.float32)
    urm = sp.csr_array((ratings, (users, items)), shape=(n_users, n_items))
    urm.sum_duplicates()
    # int32 indices: 0.7.0 array-mode filter_cols/target_cols crash on int64 (separate fix)
    urm.indices = urm.indices.astype(np.int32)
    urm.indptr = urm.indptr.astype(np.int32)
    return urm


def random_col_matrix(n_rows, n_cols, per_row, seed):
    """Sparse 0/1 matrix with per_row random columns per row (filter/target selector)."""
    rng = np.random.default_rng(seed)
    rows = np.repeat(np.arange(n_rows), per_row)
    cols = rng.integers(0, n_cols, size=n_rows * per_row)
    m = sp.csr_array((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n_rows, n_cols))
    m.sum_duplicates()
    m.data[:] = 1
    return m


def build_cases(X):
    """(name, function, kwargs) for each case. X is items x users."""
    n = X.shape[0]
    rng = np.random.default_rng(7)
    return [
        ('dot_product', sim.dot_product, {}),
        ('cosine', sim.cosine, {}),
        ('rp3beta', sim.rp3beta, {}),
        ('cosine_k10', sim.cosine, {'k': 10}),
        ('cosine_k1000', sim.cosine, {'k': 1000}),
        ('cosine_binary', sim.cosine, {'binary': True}),
        ('cosine_noblock', sim.cosine, {'block_size': None}),
        ('cosine_block2048', sim.cosine, {'block_size': 2048}),
        ('cosine_filter_array', sim.cosine,
         {'filter_cols': np.sort(rng.choice(n, n // 10, replace=False))}),
        ('cosine_filter_matrix', sim.cosine,
         {'filter_cols': random_col_matrix(n, n, 50, seed=1)}),
        ('cosine_target_matrix', sim.cosine,
         {'target_cols': random_col_matrix(n, n, 500, seed=2)}),
    ]


def time_call(func, X, kwargs, kernel, num_threads):
    """One serial call: (total seconds, kernel seconds, output)."""
    kw = {'k': 100, 'verbose': False, 'num_threads': num_threads, **kwargs}
    kw['_backend'] = 'cython' if kernel == 'cpp' else kernel
    start = time.perf_counter()
    out = func(X, **kw)
    total = time.perf_counter() - start
    return total, _s_plus_module._last_kernel_time, out


def measure(func, X, kwargs, kernel, num_threads, reps):
    """Median of reps serial calls, shaped like benchmark_similarity's result."""
    totals, kernels = [], []
    for _ in range(reps):
        total, kern, out = time_call(func, X, kwargs, kernel, num_threads)
        totals.append(total)
        kernels.append(kern)
    n_items = out.shape[0]
    return {
        'computation_time': statistics.median(totals),
        'kernel_time': statistics.median(kernels),
        'n_items': n_items,
        'nnz': out.nnz,
        'density': out.nnz / (n_items * out.shape[1]),
        'avg_neighbors': out.nnz / n_items,
    }


def measure_concurrent(X, kernel, reps, concurrent):
    """Median wall time of two cosine calls (num_threads=2 each), concurrent or serial."""
    kw = {'k': 100, 'verbose': False, 'num_threads': 2}
    kw['_backend'] = 'cython' if kernel == 'cpp' else kernel
    times = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(reps):
            start = time.perf_counter()
            if concurrent:
                outs = list(pool.map(lambda _: sim.cosine(X, **kw), range(2)))
            else:
                outs = [sim.cosine(X, **kw) for _ in range(2)]
            times.append(time.perf_counter() - start)
    out = outs[0]
    n_items = out.shape[0]
    return {
        'computation_time': statistics.median(times),
        'kernel_time': float('nan'),  # per-call global is racy under concurrency
        'n_items': n_items,
        'nnz': out.nnz,
        'density': out.nnz / (n_items * out.shape[1]),
        'avg_neighbors': out.nnz / n_items,
    }


def main():
    parser = argparse.ArgumentParser(description='Small-workload benchmark preset for similaripy')
    parser.add_argument('--rounds', type=int, default=3, help='Timed rounds (default: 3)')
    parser.add_argument('--cases', nargs='+', default=SMOKE_CASES,
                        help='Case names, or "all" (default: smoke set)')
    parser.add_argument('--warmup', type=int, default=1, help='Warm-up rounds (default: 1)')
    parser.add_argument('--reps', type=int, default=21,
                        help='Calls per timing, median taken (default: 21)')
    parser.add_argument('--threads', type=int, default=0, help='Threads for serial cases, 0=auto')
    parser.add_argument('--kernel', nargs='+', default=['cpp'], choices=['cpp', 'rust'],
                        help='Kernel backend(s), interleaved per round (default: cpp)')
    parser.add_argument('--tag', type=str, default=None, help='Report filename tag')
    parser.add_argument('--note', type=str, default=None, help='Note attached to the report')
    parser.add_argument('--output-dir', type=str, default='bench_results')
    args = parser.parse_args()

    X = make_urm().T.tocsr()  # items x users, as in benchmark_similarity
    assert X.indices.dtype == np.int32
    cases = build_cases(X)
    concurrency = ['cosine_2x2_serial', 'cosine_2x2_concurrent']
    if args.cases != ['all']:
        unknown = set(args.cases) - {name for name, _, _ in cases} - set(concurrency)
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c[0] in args.cases]
        concurrency = [c for c in concurrency if c in args.cases]

    measurements = []
    for name, func, kwargs in cases:
        for kernel in args.kernel:
            measurements.append((result_key(name, kernel),
                                 lambda f=func, kw=kwargs, kern=kernel:
                                 measure(f, X, kw, kern, args.threads, args.reps)))
    for kernel in args.kernel:
        for concurrent, name in ((False, 'cosine_2x2_serial'), (True, 'cosine_2x2_concurrent')):
            if name not in concurrency:
                continue
            measurements.append((result_key(name, kernel),
                                 lambda kern=kernel, c=concurrent:
                                 measure_concurrent(X, kern, args.reps, c)))

    round_results = {key: [] for key, _ in measurements}
    for round_num in range(args.warmup + args.rounds):
        is_warmup = round_num < args.warmup
        label = 'warmup' if is_warmup else f"round {round_num - args.warmup + 1}/{args.rounds}"
        for key, run in measurements:
            r = run()
            print(f"[{label}] {key:<28} total {r['computation_time'] * 1e3:8.2f} ms  "
                  f"kernel {r['kernel_time'] * 1e3:8.2f} ms")
            if not is_warmup:
                round_results[key].append(r)

    all_results = {DATASET: {key: aggregate_rounds(rs) for key, rs in round_results.items()}}
    dataset_info = {DATASET: {'shape': X.shape, 'nnz': X.nnz, 'density': X.nnz / (X.shape[0] * X.shape[1])}}
    config = {
        'datasets': [DATASET],
        'similarities': [name for name, _, _ in cases] + concurrency,
        'kernels': args.kernel,
        'k': 100,
        'shrink': 0,
        'threshold': 0,
        'num_threads': args.threads,
        'block_size': 0,
        'rounds': args.rounds,
        'warmup': args.warmup,
        'note': args.note,
        'tag': args.tag,
    }
    sys_info = get_system_info()

    print()
    print(format_summary_table(all_results, sys_info, config['block_size']))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = build_report_stem(sys_info, config)
    write_text_report(output_dir / f"{stem}.txt", config, dataset_info, all_results, sys_info)
    write_json_report(output_dir / f"{stem}.json", config, dataset_info, all_results, sys_info)
    print(f"\nResults saved to:\n  Text:  {output_dir / stem}.txt\n  JSON:  {output_dir / stem}.json")


if __name__ == "__main__":
    main()
