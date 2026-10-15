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

# Compare two builds (e.g. before/after a change): run in each checkout, then compare the JSONs
python tests/benchmarks/run_benchmarks.py --dataset movielens --rounds 5 --tag before
python tests/benchmarks/compare_benchmarks.py bench_results/result_before_*.json bench_results/result_after_*.json

# Small-workload preset (synthetic 10k x 10k, millisecond calls): smoke set, or every case
python tests/benchmarks/run_small_benchmarks.py
python tests/benchmarks/run_small_benchmarks.py --cases all

# Recommendation step (URM x item model), with and without filter_cols=URM
python tests/benchmarks/run_recommend_benchmarks.py --dataset movielens --version 32m

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

### `run_recommend_benchmarks.py`
The recommendation step: `dot_product(URM, S)` with S a cosine item model
(`--model-k`, built once, untimed), with and without `filter_cols=URM` ("exclude
seen items"). The item-item benchmarks have no realistic matrix-mode filter
(their output is items x items); this is where one lives, and it is the
per-request work of a server. Prints median total / prep / kernel / output;
`--json` writes them.

Every runner reports the time per phase (summary table and JSON): **kernel** (scoring, top-K, per-row
un-permute + sort, compaction) and **output** (matrix wrap) come from `similaripy._s_plus._last_timings`, **prep** is the rest
(wrapper normalization + numpy prep).

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
--format-output FMT    Output format csr/coo passed to every call (default: csr; also in
                        run_small_benchmarks.py / run_recommend_benchmarks.py). Builds whose
                        library default differs (0.7.0: coo) time the same output.
--rounds ROUNDS        Number of timed rounds per config (default: 1)
--warmup N             Untimed warm-up rounds before the timed ones (default: 1)
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
├── run_recommend_benchmarks.py # Recommendation step, filter_cols=URM (CLI)
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

These speedups in `crates/similaripy-core/src/kernel.rs` and `topk.rs` depend on what LLVM emits, and no unit test can see them (the output stays the same either way):

- `lower_bound` must stay **branchy**. LLVM likes to turn it into a branchless `csel`/`cmov` loop, which serializes cache misses once the data no longer fits in cache. An empty `asm!` barrier prevents that today. With the per-row block cursor, it only finds where a block ends inside the rest of an m2 row.
- `accumulate` must keep `sums` and the `touched` cursor in registers. As a `&mut self` method, LLVM reloaded them from the stack on every element.
- `accumulate`'s first-touch push must stay **branchless**: always write the slot, then `n += (old == 0.0)`. On Yambda ~39% of elements are first touches, so a branch there is mispredicted all the time; replacing it cut kernel time by ~20% on Yambda 50M. Check for `cinc` (aarch64) / `cmpeqss` + `and` + `add` (x86_64) and no branch on the old sum.
- The heap's sift-down (`PeekMut` drop) must stay **out of line** (a `bl`/`call` in every drain copy). When a change got it inlined into the drain loops, no run showed a gain and some were up to ~8% slower (likely register pressure in the hottest loop); replacements are only ~4% of scored columns, so the call is cheap.
- `drain` branches once per drain on `a1 != 1.0` (`POW`) and on the scoring mode (`DOT` / `COSINE` / `FULL`), so the loop has no `powf` call and no per-column weight tests for dot and cosine. Inside the loop LLVM speculates `powf` and calls it on every column; a bayesian-shrink test in the small `COSINE` loop gets if-converted into a second `fdiv` on every column, which is why any bayesian shrink goes to `FULL`. Check: no `bl powf` in the a1 == 1 copies, one `fdiv`/`divss` in the cosine copy.
- `Scratch::drain` is `#[inline(never)]` and its loop is a free function over slices and locals (`drain_impl`), like `accumulate`: inlined into the row functions, or as a `&mut self` method, LLVM reloaded the `sums` pointer, the row's norm and the heap's length/root on every column. The row's norm terms (`RowTerms`) and the lowest acceptable score (`TopK::floor`) live in registers. A cosine column rejected below the floor is ~18 instructions with 3 loads (touched index, sum, column norm); check that no load of `sums`' pointer or of the heap precedes the reject compare.
- `TopK::push` rejects below the root with one compare (`!(score >= root)`: `fcmp; b.lt` / `ucomiss; jb`); only an equal score takes the column tie test.

The first two only show on Yambda 500M (ML-32M and Yambda 50M mostly fit in cache); the first-touch branch and the drain shape show at both Yambda sizes. When you upgrade rustc or edit these loops, rerun Yambda 50M and the 500M comparison on an idle machine against the previous build (a `git worktree` of the last good commit, its own venv), alternating the two processes:

```bash
python run_benchmarks.py --dataset yambda --version 500m --similarities dot_product cosine \
    --rounds 1 --warmup 0
```

When the Rust kernel became slower than cpp at 500M, the cause was `lower_bound` and the `accumulate` register reloads (+13..16% before the fix, −7..−11% after). The block cursor (each m2 row read once, front to back, per output row) took it to −13..−17%. The scoring modes and the drain over locals took a further −13..−18% (cosine) and −9..−12% (dot) of 500M kernel time; the one-compare root reject measured as noise.

## Profiling the kernel (macOS)

Per-phase shares (accumulate, drain/score, top-K, row setup) come from `/usr/bin/sample` on a **stripped plain-release** build: the `[profile.release]` in `Cargo.toml`, which has no debug info, plus `strip -x` to drop the local symbols. With debug info (even after `strip -x`, since `sample` finds it by UUID) or with local symbols, `sample` collapses the inlined kernel into one symbol. Stripped, it reports raw offsets, one per instruction.

1. Build the release extension (`make install-dev`) and keep an unstripped copy for disassembly. Then `strip -x similaripy/_core.cpython-*-darwin.so`. Run Python from `.venv/bin/python` directly: `uv run` rebuilds over the stripped file.
2. In a script: load the dataset, start `/usr/bin/sample <pid> <seconds> -file out.txt` **before** the call, and give it more seconds than the call takes. Sample the whole call: row order isn't random (popularity-ordered rows), so a time window biases the shares.
3. Read the "Sort by top of stack" section. It gives samples per `_core` offset. Bucket the offsets into phases with address ranges read from `objdump -d --start-address=.. --stop-address=..` of the unstripped copy (symbols from `nm -n`). `single_block_row`, `blocked_row` and `PeekMut` drop are own symbols; accumulate, drain and `lower_bound` are inlined into them. Leave rayon's park/steal code out of the denominator.
4. Samples land on the instruction *after* a stall: read the pattern around a branch or a load, not one address.

`sample` slows the run by ~25–35%, so take seconds from an unsampled run. For work counts that hold on every CPU (elements accumulated, first touches, heap replacements, root rejections), a throwaway patch with per-worker counters is more reliable than shares; don't time that build. To check x86_64 codegen from a Mac: `rustup target add x86_64-unknown-linux-gnu`, then `cargo rustc -p similaripy-core --release --lib --target x86_64-unknown-linux-gnu -- --emit asm` (no linker needed; LTO into the extension can still change inlining).

## 0.7.0 reference (Cython)

Cython was removed in 1.0 development; two references remain:

- **Outputs (parity):** `check_kernel_parity.py` and `check_normalization_parity.py` run in two steps: `--dump` under the reference install, `--compare` here (usage in their docstrings). The quick reference is the 0.7.0 wheel from PyPI (`uv run --isolated --no-project --with similaripy==0.7.0 ...`). Run the scripts by path: `python -c` from the repo root imports the checkout instead.
- **Bit-level and timing:** commit `608cc06` (local tag `v1-cython-ref`; from a fresh clone: `git fetch origin pull/29/head`), the last commit with both kernels in one process (`--kernel cpp rust`, interleaved ABAB). Build it in a `git worktree` with its own venv (`make install-dev`, needs Homebrew GCC on macOS).

The PyPI wheel was compiled by a different GCC (CI runner, `-O3 -ffast-math`), so some scores differ by 1 float32 ulp from the local build the Rust core was matched against. Parity still passes, but near-ties at the k-th score become one-sided entries: ML-32M rp3beta has ~627k (7%) against PyPI and 0 against the tag build; synthetic p3alpha/rp3beta 2/26. For a bit-level check, dump from the tag build.
