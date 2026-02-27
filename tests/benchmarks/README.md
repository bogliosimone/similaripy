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

# Run multiple rounds and average results (useful for stable benchmarks)
python tests/benchmarks/run_benchmarks.py --dataset movielens --version 32m --rounds 3

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

### `compare_benchmarks.py`
Benchmark comparison tool. Supports:
- Listing all available JSON reports
- Comparing 2+ runs side-by-side
- Auto-scanning the bench directory
- Filtering by version pattern or similarity type
- Shows timing differences and percentage changes

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
--rounds ROUNDS        Number of times to run each config, results averaged (default: 1)
--output-dir DIR       Output directory for results (default: bench_results)
--note NOTE            Optional note/comment to attach to the report
--event-type TYPE      Yambda event type: likes/listens/multi_event (default: multi_event)
--quiet                Suppress verbose output
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

When `--rounds` > 1, the time column shows mean ± standard deviation across all rounds.

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
        "all_times": [4.96, 4.88]
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
