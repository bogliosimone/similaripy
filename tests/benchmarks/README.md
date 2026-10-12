# Similaripy Benchmarking

Unified benchmarking system for testing similaripy performance across datasets and similarity algorithms.

## Quick Start

### Installation

```bash
pip install -e ".[bench]"
```

### Run Benchmarks

```bash
# Benchmark MovieLens 32M with default similarities (dot_product, cosine, rp3beta)
python tests/benchmarks/run_benchmarks.py --dataset movielens --version 32m

# Benchmark specific similarities
python tests/benchmarks/run_benchmarks.py --dataset movielens --version 32m --similarities cosine rp3beta

# Multiple datasets
python tests/benchmarks/run_benchmarks.py --dataset movielens yambda --similarities cosine rp3beta

# Stable timing: 1 untimed warm-up round (default) + 5 timed rounds, median reported
python tests/benchmarks/run_benchmarks.py --dataset movielens --version 32m --rounds 5

# Compare kernels in one process, interleaved per round (ABAB); rust needs the spike build
python tests/benchmarks/run_benchmarks.py --dataset movielens --kernel cpp rust --rounds 5 --tag spike

# Small-workload preset (synthetic 10k x 10k, millisecond calls): smoke set, or every case
python tests/benchmarks/run_small_benchmarks.py
python tests/benchmarks/run_small_benchmarks.py --cases all

# Custom parameters
python tests/benchmarks/run_benchmarks.py --dataset movielens --k 200 --shrink 10 --threads 4

# Add a note to the benchmark report
python tests/benchmarks/run_benchmarks.py --dataset movielens --note "baseline before optimization"
```

### Compare Benchmarks

```bash
# List all available benchmark reports (JSON)
python tests/benchmarks/compare_benchmarks.py --list

# Compare the latest 2 runs
python tests/benchmarks/compare_benchmarks.py --latest 2

# Compare two specific report files
python tests/benchmarks/compare_benchmarks.py result_0.5.0_20260224.json result_0.6.0_20260226.json

# Compare all runs for a specific version
python tests/benchmarks/compare_benchmarks.py --filter 0.6.0

# Filter comparison to a specific similarity
python tests/benchmarks/compare_benchmarks.py --latest 3 --similarity cosine
```

## Components

### `run_benchmarks.py`
Main benchmark runner with CLI interface. Supports:
- Multiple datasets in a single run
- Multiple similarity algorithms
- Comprehensive summary table with system info
- Dataset loading time excluded from results
- Outputs both text and JSON reports
- Optional `--note` for annotating runs

### `run_small_benchmarks.py`
Small-workload preset on a seeded synthetic 10k x 10k matrix (about 1M nnz,
Zipf-like item popularity). Exercises what the large datasets don't: k=10/1000,
`binary=True`, `block_size=None`, a small explicit block size (blocked path),
array- and matrix-mode `filter_cols`/`target_cols`, and two concurrent calls
(`num_threads=2` each) vs the same calls run serially. By default only a 5-case
smoke set runs (its noise is 10-20%, so it catches big regressions only);
`--cases all` runs everything. Each timing is the median
of `--reps` calls (default 21). Run it with a free-threaded Python (3.14t/3.15t)
for a meaningful concurrency number; the report records `gil_enabled`.

### `compare_benchmarks.py`
Benchmark comparison tool. Supports:
- Listing all available JSON reports
- Comparing 2+ runs side-by-side
- Auto-scanning the bench directory
- Filtering by version pattern or similarity type
- Shows timing differences and percentage changes
- Uses the median when the report has one, else the mean (older reports)

### `dataset_loaders.py`
Unified dataset loaders for:
- **MovieLens** (`25m`, `32m`) - Auto-downloads to `tests/datasets/`
- **Yambda** (`50m`, `500m`) - Loads from HuggingFace

### `benchmark.py`
Core benchmarking functions:
- `benchmark_similarity()` - Benchmark single similarity
- `get_system_info()` - Collect system/environment info for reports

## Supported Datasets

| Dataset | Versions | Size | Requirements |
|---------|----------|------|--------------|
| MovieLens | 25m, 32m | 25-32M ratings | pandas |
| Yambda | 50m, 500m | 50-500M interactions | pandas, datasets, pyarrow |

## Supported Similarities

`dot_product`, `cosine`, `asymmetric_cosine`, `jaccard`, `dice`, `tversky`, `p3alpha`, `rp3beta`, `splus`

## CLI Options

```
--dataset DATASET [DATASET ...]
                        Dataset(s) to benchmark (default: movielens)
--version VERSION      Dataset version (default: dataset default)
--similarities SIM [SIM ...]
                        Similarity types (default: dot_product cosine rp3beta)
--k K                  Top-k items to keep (default: 100)
--shrink SHRINK        Shrinkage parameter (default: 0)
--threshold THRESHOLD  Minimum threshold (default: 0)
--threads THREADS      Number of threads, 0=auto (default: 0)
--block-size SIZE      Block size: 0=auto, none=disabled, or int>0 (default: 0)
--rounds ROUNDS        Number of timed rounds per config (default: 1)
--warmup N             Untimed warm-up rounds before the timed ones (default: 1)
--kernel K [K ...]     Kernel backend(s): cpp, rust; interleaved per round (default: cpp)
--tag TAG              Report filename tag (default: similaripy version)
--output-dir DIR       Output directory for results (default: bench_results)
--note NOTE            Optional note/comment to attach to the report
--event-type TYPE      Yambda event type: likes/listens/multi_event (default: multi_event)
--quiet                Suppress harness progress output
```

## Output Format

The benchmark produces a comprehensive summary table:

```
========================================================================================================================
BENCHMARK SUMMARY
========================================================================================================================
Date: 2026-02-24 18:30:00
Similaripy version: 0.5.0
Git commit: 53cfcc4
Python: 3.14.0
CPU: Apple M2 Pro
Architecture: Darwin arm64
CPU cores available: 10
Block size: auto
========================================================================================================================
Dataset              Version    Similarity           Time (s)           Throughput      Output nnz      Avg Neighbors   Rounds
------------------------------------------------------------------------------------------------------------------------
movielens            32m        cosine               12.45 ± 0.23       2534.5          3153400         100.0           3
                                dot_product          8.23 ± 0.15        3825.4          3153400         100.0           3
                                rp3beta              15.67 ± 0.31       2010.3          3153400         100.0           3
========================================================================================================================
```

- **Median (s)**: median wall time of the public call across timed rounds (the primary number).
- **Kernel (s)**: median time inside the compute kernel only (excludes Python prep and output build).
- **Spread**: (max − min) / median across rounds, a quick noise indicator.
- **Mean ± std (s)**: kept for comparison with older reports.

The similarity call always runs with `verbose=False`, so the progress bar never
falls inside the timed region. Configs are interleaved within each round, so slow
drift (thermals, background load) affects all of them equally.

## Python API

```python
from dataset_loaders import load_URM
from benchmark import benchmark_similarity

# Load dataset (not timed)
URM, meta = load_URM("movielens", version="32m")

# Single similarity
results = benchmark_similarity(URM, similarity_type="cosine", k=100)
```

## Directory Structure

```
tests/benchmarks/
├── run_benchmarks.py      # Main benchmark runner (CLI)
├── compare_benchmarks.py  # Benchmark comparison tool (CLI)
├── dataset_loaders.py     # Dataset loaders
├── run_small_benchmarks.py # Small-workload preset (CLI)
├── benchmark.py           # Core benchmark functions
└── README.md              # This file

tests/datasets/            # Downloaded datasets (gitignored)
├── ml-25m/
└── ml-32m/
```

## Output Files

Benchmark results are automatically saved in two formats:
- **Text report**: Human-readable summary table (`result_{version}_{timestamp}.txt`)
- **JSON report**: Machine-readable data for comparisons (`result_{version}_{timestamp}.json`)

**Default location**: `bench_results/` directory

### JSON Report Structure

```json
{
  "metadata": {
    "similaripy_version": "0.6.0",
    "timestamp": "2026-02-26 21:27:28",
    "note": "baseline before optimization",
    ...
  },
  "config": {
    "datasets": [["movielens", "32m"]],
    "similarities": ["cosine", "rp3beta"],
    "k": 100, "shrink": 0, "rounds": 2,
    ...
  },
  "datasets": {
    "movielens:32m": { "shape": [200948, 84432], "nnz": 32000204, "density": 0.00188609 }
  },
  "results": {
    "movielens:32m": {
      "cosine": {
        "computation_time": 4.92,
        "std_time": 0.04,
        "throughput": 17147.5,
        "nnz": 8443200,
        "avg_neighbors": 100.0,
        "rounds": 2,
        "median_time": 4.92,
        "kernel_median_time": 4.55,
        "spread": 0.0163,
        "all_times": [4.96, 4.88],
        "kernel_times": [4.58, 4.52]
      }
    }
  }
}
```

Custom output directory:
```bash
python tests/benchmarks/run_benchmarks.py --dataset movielens --output-dir my_results
```

## Notes

- Dataset loading time is **excluded** from benchmark results
- Only similarity computation time is measured
- First run downloads datasets automatically
- Results include system info (version, architecture, CPU model, Python version, git commit)
- All benchmark runs are saved to timestamped files automatically

## Codegen-sensitive hot loops (Rust core)

Two speedups in `crates/similaripy-core/src/kernel.rs` depend on what LLVM emits, and no unit test can see them:

- `lower_bound` must stay **branchy**. LLVM likes to turn it into a branchless `csel`/`cmov` loop, which serializes cache misses once the data no longer fits in cache. An empty `asm!` barrier prevents that today.
- `accumulate` must keep `sums` and the `touched` cursor in registers. As a `&mut self` method, LLVM reloaded them from the stack on every element.

Only Yambda 500M exposes a regression here; ML-32M and Yambda 50M mostly fit in cache. When you upgrade rustc or edit either function, rerun the 500M comparison on an idle machine, and check the cpp time against earlier runs to catch background load:

```bash
python run_benchmarks.py --dataset yambda --version 500m --similarities dot_product cosine \
    --rounds 1 --warmup 0 --kernel cpp rust
```

Until the Python bindings land (v1 Step 2), `--kernel rust` needs the spike hook, which lives only in the commits of the `v1/rust-spike` PR (`git fetch origin pull/<N>/head`).

When the Rust kernel became slower than cpp at 500M, the cause was these two loops (+13..16% before the fix, −7..−11% after).
