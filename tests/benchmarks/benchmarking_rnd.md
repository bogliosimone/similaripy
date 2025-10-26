# Performance Benchmarking Guide

This document describes how to benchmark similaripy and track performance across versions.

## Quick Start

### Run All Benchmarks
```bash
# Using tox (recommended for CI)
make benchmark

# Or run locally
make benchmark-local
```

### Run Specific Benchmarks
```bash
# Run only normalization benchmarks (20 tests)
make benchmark-norm

# Run only similarity benchmarks (4 tests)
make benchmark-similarity

# Or use pytest directly for more control
uv run pytest tests/benchmarks.py::TestNormalizationPerformance --benchmark-only
uv run pytest tests/benchmarks.py::TestSPlusPerformance --benchmark-only

# Run specific size (e.g., only small matrices)
uv run pytest tests/benchmarks.py -k "small" --benchmark-only
```

## Comparing Versions

### Method 1: pytest-benchmark (Recommended)

Save a baseline and compare:

```bash
# 1. Save baseline from current version
uv run pytest tests/benchmarks.py --benchmark-only --benchmark-save=v0.2.4

# 2. Make your changes...

# 3. Compare against baseline
uv run pytest tests/benchmarks.py --benchmark-only --benchmark-compare=v0.2.4

# 4. Fail if performance degrades >10%
uv run pytest tests/benchmarks.py --benchmark-only \
  --benchmark-compare=v0.2.4 \
  --benchmark-compare-fail=mean:10%
```

View all saved benchmarks:
```bash
ls -la .benchmarks/
```

### Quick Comparison (Small Matrices Only)

For faster comparison, test only small matrices:

```bash
# Save baseline with small matrices only
uv run pytest tests/benchmarks.py -k "small" --benchmark-only --benchmark-save=baseline

# Compare after changes
uv run pytest tests/benchmarks.py -k "small" --benchmark-only --benchmark-compare=baseline

# Fail if >15% slower
uv run pytest tests/benchmarks.py -k "small" --benchmark-only \
  --benchmark-compare=baseline \
  --benchmark-compare-fail=mean:15%
```

## Benchmark Structure

### Test Organization

```
tests/
├── benchmarks.py              # Main benchmark suite (24 tests total)
├── test_normalization.py      # Unit tests for normalization
├── test_similarity.py         # Unit tests for similarity
└── conftest.py               # pytest configuration (registers perf marker)
```

### Benchmark Categories

- **Normalization Benchmarks** (20 tests) - L1, L2, max, TF-IDF, BM25 across 3 matrix sizes
- **Similarity Benchmarks** (4 tests) - s_plus computations with realistic datasets

    - Basic s_plus (2 tests): Default parameters on MovieLens-like datasets
    - Complex s_plus (2 tests): Full normalization parameters (l1, l2, l3, depopularization, bayesian shrinkage)

### Matrix Sizes

- **Small**: 1,000 × 500 (0.05 density) - ~25K non-zeros
- **Medium**: 10,000 × 5,000 (0.01 density) - ~500K non-zeros
- **Large**: 50,000 × 10,000 (0.005 density) - ~2.5M non-zeros

## Interpreting Results

### pytest-benchmark Output

```
Name (time in ms)           Mean    StdDev  Min     Max     Rounds
test_l2_norm[small]        12.5    0.3     12.1    13.2    50
test_l2_norm[medium]       245.2   5.1     238.4   255.3   20
```

### What to Look For

- **Mean time**: Primary metric
- **Std Dev**: Low is better (more consistent)
- **Regression**: >10% slowdown should be investigated
- **Improvement**: >20% speedup is significant

## Best Practices

1. **Consistent Environment**
    - Close other applications
    - Disable CPU frequency scaling
    - Run on same hardware for comparisons

2. **Warmup Runs**
    - First run is always slower (compilation, caching)
    - Benchmarks include warmup rounds

3. **Multiple Runs**
    - Default: 5-10 rounds per benchmark
    - More rounds = more reliable statistics

4. **Realistic Data**
    - Use matrix sizes matching your use case
    - Test with actual data sparsity patterns

## Advanced Usage

### Custom Benchmark

```python
@pytest.mark.perf
def test_my_custom_benchmark(benchmark):
    mat = generate_sparse_matrix(20_000, 10_000, 0.01)
    result = benchmark(my_function, mat, my_param=42)
    assert result is not None
```

### Profiling

For detailed profiling:

```bash
# Using pytest-benchmark
uv run pytest tests/benchmarks.py::test_specific_benchmark \
  --benchmark-only \
  --benchmark-cprofile

# Using Python's profiler
uv run python -m cProfile -o profile.stats your_script.py
uv run python -m pstats profile.stats
```

## Troubleshooting

### "Benchmark runs too fast"
- Increase matrix size
- Use `rounds` parameter to force more iterations

### "Results vary too much"
- Ensure no background processes
- Increase warmup rounds
- Check CPU throttling

### "Out of memory"
- Reduce matrix size
- Use smaller `k` parameter for similarity

## Resources

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [Python profiling](https://docs.python.org/3/library/profile.html)
- [Cython profiling](https://cython.readthedocs.io/en/latest/src/tutorial/profiling_tutorial.html)
