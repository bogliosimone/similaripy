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
```

## Components

### `run_benchmarks.py`
Main benchmark runner with CLI interface. Supports:
- Multiple datasets in a single run
- Multiple similarity algorithms
- Comprehensive summary table with system info
- Dataset loading time excluded from results

### `dataset_loaders.py`
Unified dataset loaders for:
- **MovieLens** (`25m`, `32m`) - Auto-downloads to `tests/datasets/`
- **Yambda** (`50m`, `500m`) - Loads from HuggingFace

### `benchmark.py`
Core benchmarking functions:
- `benchmark_similarity()` - Benchmark single similarity
- `profile_similarities()` - Profile multiple similarities

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
--rounds ROUNDS        Number of times to run each config, results averaged (default: 1)
--output-dir DIR       Output directory for results (default: bench_results)
--event-type TYPE      Yambda event type: likes/listens/multi_event (default: multi_event)
--quiet                Suppress verbose output
```

## Output Format

The benchmark produces a comprehensive summary table:

```
========================================================================================================================
BENCHMARK SUMMARY
========================================================================================================================
Similaripy version: 0.1.0
Architecture: Darwin arm64
CPU cores available: 10
========================================================================================================================
Dataset              Version    Similarity           Time (s)           Throughput      Avg Neighbors   Rounds
------------------------------------------------------------------------------------------------------------------------
movielens            32m        cosine               12.45 ± 0.23       2534.5          100.0           3
                                dot_product          8.23 ± 0.15        3825.4          100.0           3
                                rp3beta              15.67 ± 0.31       2010.3          100.0           3
========================================================================================================================
```

When `--rounds` > 1, the time column shows mean ± standard deviation across all rounds.

## Python API

```python
from dataset_loaders import load_URM
from benchmark import benchmark_similarity, profile_similarities

# Load dataset (not timed)
URM, meta = load_URM("movielens", version="32m")

# Single similarity
results = benchmark_similarity(URM, similarity_type="cosine", k=100)

# Multiple similarities
results = profile_similarities(URM, similarity_types=("dot_product", "cosine", "rp3beta"), k=100)
```

## Directory Structure

```
tests/benchmarks/
├── run_benchmarks.py      # Main benchmark runner (CLI)
├── dataset_loaders.py     # Dataset loaders
├── benchmark.py           # Core benchmark functions
├── benchmarks_rnd_seed.py # Random seed benchmarks
└── README.md              # This file

tests/datasets/            # Downloaded datasets (gitignored)
├── ml-25m/
└── ml-32m/
```

## Output Files

Benchmark results are automatically saved to text files:
- **Default location**: `bench_results/` directory
- **Filename format**: `result_{version}_{timestamp}.txt`
  - Example: `result_0.1.0_20250126_143025.txt`
- **Contents**: Complete benchmark output including:
  - SIMILARIPY BENCHMARK SUITE section
  - Dataset loading information
  - BENCHMARK SUMMARY table

Custom output directory:
```bash
python tests/benchmarks/run_benchmarks.py --dataset movielens --output-dir my_results
```

## Notes

- Dataset loading time is **excluded** from benchmark results
- Only similarity computation time is measured
- First run downloads datasets automatically
- Results include system info (version, architecture, CPU cores)
- All benchmark runs are saved to timestamped files automatically
