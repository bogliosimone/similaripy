# Profiling Tools

This directory contains profiling scripts for benchmarking and analyzing the performance of SimilariPy's similarity algorithms.

## Reference Dataset

All profiling scripts use **Yamba 50M** as the reference dataset for consistent benchmarking.

## Available Scripts

### 1. `benchmark_wallclock.py`
**Purpose:** Quick wall-clock time benchmark for measuring real-world performance.

**What it measures:**
- Wall-clock execution time (real elapsed time)
- Throughput (items processed per second)

**Usage:**
```bash
cd tests/profiling
uv run python benchmark_wallclock.py
```

**When to use:**
- ✅ Comparing performance between code versions
- ✅ Measuring real-world speedup after optimizations
- ✅ Quick regression testing for performance
- ✅ **This is your main benchmarking tool**

---

### 2. `profile_cprofile.py`
**Purpose:** Python-level profiling with cProfile to identify bottlenecks in preprocessing.

**What it measures:**
- ✅ Python function call times (cumulative and internal)
- ✅ Preprocessing steps (validation, conversions, etc.)
- ✅ Time spent in Cython code before `with nogil:`
- ❌ **CANNOT see into C++ code inside `with nogil:` blocks**

**Usage:**
```bash
cd tests/profiling
uv run python profile_cprofile.py
```

**Output:**
- Console output with top 30 functions by time
- `profile_cosine_yamba50m.prof` - Binary profile data
- `profile_cosine_yamba50m.txt` - Human-readable report

**When to use:**
- ✅ Debugging slow preprocessing (matrix conversions, validation)
- ✅ Finding Python/Cython overhead before the hot path
- ✅ Identifying excessive function calls
- ❌ **DO NOT use for C++ hot path profiling** (shows only as "lock acquire")

**Example use cases:**
- "Why does my similarity function take 5 seconds before the progress bar starts?"
- "Is there excessive time in CSR conversion or eliminate_zeros()?"
- "Are we spending too much time in input validation?"

---

### 3. `profile_pyspy.py`
**Purpose:** Native C++ profiling with py-spy for analyzing the similarity computation hot path.

**What it measures:**
- ✅ C++ function execution time
- ✅ Native code performance inside `with nogil:` blocks
- ✅ Full stack traces including C++/Cython code
- ✅ Multi-threaded OpenMP performance

**Usage:**
```bash
cd tests/profiling
sudo py-spy record -o flamegraph.svg --native -- uv run python profile_pyspy.py
```

**Output:**
- Flamegraph visualization (`flamegraph.svg`) - open in browser

**When to use:**
- ✅ Profiling C++ hot path (accumulation, foreach, output writing)
- ✅ Quick overview of where time is spent in native code
- ✅ Visual analysis with flamegraphs
- ✅ Multi-threaded performance analysis
- ✅ **Best for laptops/machines where you have root access**

**Requirements:**
- `py-spy` installed: `pip install py-spy`
- `sudo` access (required for native profiling)

**Limitations:**
- Less precise than manual C++ instrumentation (with `std::chrono`)
- Requires root access (not available in all environments)

**Alternative:**
For more precise C++ profiling, add manual instrumentation:
```cpp
#include <chrono>
auto start = std::chrono::high_resolution_clock::now();
// ... code to profile ...
auto end = std::chrono::high_resolution_clock::now();
std::cerr << "Time: " << std::chrono::duration<double>(end - start).count() << "s\n";
```

---

## Quick Reference: Which Tool to Use?

| Scenario | Tool |
|----------|------|
| Compare performance between versions | `benchmark_wallclock.py` |
| Slow startup before progress bar | `profile_cprofile.py` |
| Python/preprocessing bottleneck | `profile_cprofile.py` |
| C++ hot path bottleneck (with root) | `profile_pyspy.py` |
| C++ hot path bottleneck (no root) | Manual C++ instrumentation |
| Quick visual overview of C++ | `profile_pyspy.py` |

---

## Benchmark Parameters

All scripts use consistent parameters:
- **Dataset:** Yamba 50M
- **k:** 100 (top-100 neighbors)
- **shrink:** 0 (no shrinkage)
- **threshold:** 0 (no threshold filtering)
- **num_threads:** 0 (use all available cores)
- **format_output:** 'csr' (sparse CSR matrix output)

---

## Profiling Workflow

### For Performance Optimization:

1. **Baseline Benchmark**
   ```bash
   uv run python benchmark_wallclock.py > baseline.txt
   ```

2. **Apply Code Changes**
   - Modify C++ code in `similaripy/cython_code/`
   - Rebuild: `make test-dev`

3. **Optimized Benchmark**
   ```bash
   uv run python benchmark_wallclock.py > optimized.txt
   ```

4. **Compare Results**
   ```bash
   diff baseline.txt optimized.txt
   ```

### For Deep Analysis:

1. **Identify bottleneck location** (Python vs C++)
   ```bash
   uv run python profile_cprofile.py
   ```
   - If time is in preprocessing → optimize Python/Cython code
   - If time is in "lock acquire" → bottleneck is in C++ hot path

2. **Profile C++ hot path** (if you have root access)
   ```bash
   sudo py-spy record -o flamegraph.svg --native -- uv run python profile_pyspy.py
   ```
   - View flamegraph: Open `flamegraph.svg` in browser
   - Identify which C++ functions are taking the most time

3. **Detailed C++ profiling** (if you need precise measurements)
   - Add manual `std::chrono` instrumentation to C++ code
   - Rebuild and run: `make test-dev && uv run python benchmark_wallclock.py`
   - Remove instrumentation after analysis

---

## Notes

- **Wall-clock vs CPU time:** Benchmark measures wall-clock time (real elapsed time), while profilers may show cumulative CPU time across all threads.
- **Multi-threading:** With OpenMP parallel execution, CPU time can exceed wall-clock time by the number of cores (e.g., 12x on 12-core system).
- **Cache effects:** First run may be slower due to cold cache. Run multiple times for consistent measurements.
- **System load:** Close other applications to avoid interference with benchmarks.
- **cProfile limitation:** Cannot see into `with nogil:` blocks - use py-spy or manual instrumentation for C++ profiling.
