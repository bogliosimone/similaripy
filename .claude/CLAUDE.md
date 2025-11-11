# SimilariPy - Claude Context

## Project Overview

SimilariPy is a high-performance KNN similarity library for Python, optimized for sparse matrices. Primarily designed for Recommender Systems and Information Retrieval tasks.

**Key Technologies:**
- Cython/C++ implementation with OpenMP parallelization
- Optimized for sparse CSR matrices
- Multi-threaded, GIL-free computation

**Links:** [GitHub](https://github.com/bogliosimone/similaripy) | [Docs](https://bogliosimone.github.io/similaripy/)

---

## Code Architecture

### Design Principles
- **Public APIs**: Python interface with validation, wraps Cython implementations
  - [similarity.py](similaripy/similarity.py) - Similarity functions (cosine, jaccard, s_plus, etc.)
  - [normalization.py](similaripy/normalization.py) - Normalization functions (BM25, TF-IDF, etc.)

- **Cython Core**: C++ compiled code with direct CSR matrix memory access, OpenMP parallel execution
  - [s_plus.pyx](similaripy/cython_code/s_plus.pyx) - S-Plus algorithm (reference implementation)
  - [normalization.pyx](similaripy/cython_code/normalization.pyx) - Normalization implementations
  - [utils.pyx](similaripy/cython_code/utils.pyx) - Common utilities

**Config:**
- [pyproject.toml](pyproject.toml) - Dependencies, build config
- [Makefile](Makefile) - Development commands

---

## Performance Architecture

**Key Optimizations:**
- **Float32 precision** + **Pre-allocated buffers** + **GIL-free computation** (`with nogil:`)
- **Top-K filtering** with efficient heap structures (pre-allocated), **Lazy normalization** (only when needed)
- **In-place operations** for zero-copy modifications

**Reference:** See [s_plus.pyx](similaripy/cython_code/s_plus.pyx) for implementation patterns.

---

## Development Workflow

### Package Manager & Build System
Uses **uv** for package management. Built with **scikit-build-core** + CMake. Requires GCC with OpenMP support.

### Setup
```bash
# First time setup (install dependencies)
make install-dev

# Development mode (editable install, auto-recompiles on test)
make install-dev-editable
```

### Testing
```bash
# Local tests (fast, correctness only)
make test-dev
```

### Benchmarking
```bash
# Small dataset for quick regression test (Movielens 32M)
make benchmark-similarity
```

**Test files:** [tests/test_similarity.py](tests/test_similarity.py), [tests/test_normalization.py](tests/test_normalization.py)

---

## Tasks for Claude

### Performance Optimization
**IMPORTANT - Always do this BEFORE implementing:**
1. Explain expected benefit (e.g., "reduces allocations by X", "enables vectorization")
2. Identify trade-offs (e.g., "uses more memory", "increases complexity")
3. Wait for approval

### Bug Fixes
- Reproduce with minimal test case
- Identify root cause before fixing
- Ensure existing tests still pass
- Add regression test if needed

### Documentation
- Update docstrings when changing function signatures
- Document performance characteristics of new algorithms

### Before Submitting Changes
1. Run `make test-dev` to verify compilation and unit tests
2. Suggest test updates for new features or uncovered bugs

---

## Common Gotchas

### Sparse Matrices
- All functions expect **CSR format** (converted automatically)
- **Always call `eliminate_zeros()`** in the init phase to maintain sparsity and avoid binary flag issues

### Cython Development
- Changes to `.pyx` files require recompilation
- **Editable mode auto-recompiles on test execution** (`make test-dev`)
- Use `cdef` for C variables, `def` for Python-callable functions
- Memory views (`float[:]`) faster than NumPy arrays in loops
- Always release GIL (`with nogil:`) for parallel C++ code
- OpenMP: `num_threads=0` = use all cores; set `OMP_NUM_THREADS` to limit
- Ensure no shared mutable state in parallel regions

---

## Version & License
- **License:** MIT
- **Python Support:** 3.10+
