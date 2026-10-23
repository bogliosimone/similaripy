# SimilariPy - Claude Context

## Project Overview

SimilariPy is a high-performance KNN similarity library for Python, optimized for sparse matrices. Primarily designed for Recommender Systems and Information Retrieval tasks.

**Key Technologies:**
- Rust core (PyO3 + rayon), built with maturin
- Optimized for sparse CSR matrices
- Multi-threaded, GIL-free computation

**Links:** [GitHub](https://github.com/bogliosimone/similaripy) | [Docs](https://bogliosimone.github.io/similaripy/)

---

## Code Architecture

### Design Principles
- **Public APIs**: Python interface with validation, calls the Rust extension `similaripy._core`
  - [similarity.py](similaripy/similarity.py) - Similarity functions (cosine, jaccard, s_plus, etc.)
  - [_s_plus.py](similaripy/_s_plus.py) - numpy prep (copies, sorting, column filters), kernel call, output build
  - [normalization.py](similaripy/normalization.py) - Normalization functions (BM25, TF-IDF, etc.)
  - [_serve.py](similaripy/_serve.py) - `ItemKNN`: prepared item-item model (layout chosen here), `view` masks, `recommend` / `recommend_batch`

- **Rust core** ([crates/similaripy-core](crates/similaripy-core)): pure Rust, no Python
  - [kernel.rs](crates/similaripy-core/src/kernel.rs) - S-Plus kernel (single-block and blocked paths, rayon)
  - [normalize.rs](crates/similaripy-core/src/normalize.rs) - Normalizations, in place over CSR data
  - [csr.rs](crates/similaripy-core/src/csr.rs), [topk.rs](crates/similaripy-core/src/topk.rs) - CSR views, top-K heap
  - [serve.rs](crates/similaripy-core/src/serve.rs) - Serving engine: owned model, one history per call, batch = rayon over the same function
  - [store.rs](crates/similaripy-core/src/store.rs) - Saved models: `.npy` reader/writer, mapped (`memmap2`) or owned arrays, directory replace
- **Bindings** ([crates/similaripy-py](crates/similaripy-py)): PyO3 module `similaripy._core`
  - [lib.rs](crates/similaripy-py/src/lib.rs) - kernel + prep bindings, thread pools; [normalize.rs](crates/similaripy-py/src/normalize.rs) - dtype dispatch; [progress.rs](crates/similaripy-py/src/progress.rs) - progress bar; [serve.rs](crates/similaripy-py/src/serve.rs) - `ServeModel` / `ServeMask`

**Config:**
- [pyproject.toml](pyproject.toml) - Dependencies, build config (`[tool.maturin]`)
- [Cargo.toml](Cargo.toml) + [rust-toolchain.toml](rust-toolchain.toml) - Rust workspace, pinned compiler
- [Makefile](Makefile) - Development commands

---

## Performance Architecture

**Key Optimizations:**
- **Float32 precision** + **Pre-allocated per-thread buffers** + **GIL-free computation** (`py.detach`)
- **Top-K filtering** with efficient heap structures (pre-allocated), **Lazy normalization** (only when needed)
- **In-place operations** for zero-copy modifications

**Reference:** See [kernel.rs](crates/similaripy-core/src/kernel.rs) for implementation patterns, and in [tests/benchmarks/README.md](tests/benchmarks/README.md) the sections "Codegen-sensitive hot loops" (what a change must not break) and "Profiling the kernel (macOS)" (per-phase shares with `sample`).

**Perf tiers:** the hot path (`accumulate`, drain/score, `TopK`, `lower_bound`) gets a perf assessment on every change; prep, output build and cold core code are correctness first, perf second.

### Reproducibility contract
- **Same version + same machine → identical results, bit for bit**, run to run, whatever `num_threads` or `block_size`. Different versions or platforms may differ by float rounding.
- New code keeps the **addition order within an output row fixed by the data**, never by thread timing: no atomic float adds, no "first finished, first combined". Intra-row parallelism only by column blocks or fixed chunks combined in a fixed order.
- **Parity check on every core change:** `check_kernel_parity.py --dump` under the base, `--compare` under the head. A refactor that claims no change: `--exact` (free, catches real bugs). Any other change: `--rtol 1e-4` + top-k overlap with the previous version, and a migration-notes line if users can see it. Small float differences (float32 vs float64, summation order) are fine: never add code just to keep bits identical.
- **Normal use cases first:** extreme values (e.g. data ×1e-30 or ×1e19, past float32 range) are best effort; a cheap check on a realistic failure is fine, dedicated code paths are not.

### Index widths
- Inputs and ids are **int32** (each input's nnz ≤ int32 max, output column ids int32). Only output counts and offsets are 64-bit: past 2^31 results the CSR output gets int64 indptr and indices.

---

## Development Workflow

### Package Manager & Build System
Uses **uv** for package management. Built with **maturin**. Requires the Rust toolchain (rustup; `rust-toolchain.toml` pins the version).

### Setup
```bash
# Editable install with every dependency group; `uv run` rebuilds it after Rust edits
make install-dev
uv run pre-commit install   # optional: ruff + cargo fmt on each commit
```

### Testing
```bash
# Local tests (fast, correctness only)
make test-dev

# Everything CI's lint, docs, Rust and test jobs run: ruff, ty, cargo fmt/clippy/test, pytest, strict docs build
make check
```

### Benchmarking
```bash
# Small dataset for quick regression test (Movielens 32M)
make benchmark-similarity-small
```

**Test files:**
- [tests/test_similarity.py](tests/test_similarity.py) - every similarity × shrink type against one dense float64 reference (`_dense_scores`), selectors, signed input
- [tests/test_s_plus_prep.py](tests/test_s_plus_prep.py) - validation, canonicalization, layout, output build, regressions
- [tests/test_normalization.py](tests/test_normalization.py), [tests/test_progress_bar.py](tests/test_progress_bar.py)
- [tests/test_free_threading.py](tests/test_free_threading.py) - concurrent calls, identical bits across threads and blocking
- [tests/test_serve.py](tests/test_serve.py) - `ItemKNN` vs a dense float64 reference, masks and views, unknown ids, batch == single bits, pickle
- [crates/similaripy-core/tests/serve.rs](crates/similaripy-core/tests/serve.rs) - serving proptest vs an f64 reference, bit-identity claims
- [crates/similaripy-core/tests/reference.rs](crates/similaripy-core/tests/reference.rs) - kernel proptest vs an f64 reference, `validate` errors; unit tests in each module

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
1. Run `make check` to verify lint, types, compilation and unit tests
2. Suggest test updates for new features or uncovered bugs

---

## Common Gotchas

### Sparse Matrices
- All functions expect **CSR format** (converted automatically)
- Inputs are canonicalized before use (summed duplicates, no explicit zeros: `_canonical_csr` in `_s_plus.py`, also used by `normalization.py`); never mutate the caller's matrices
- **Prep owns its data:** `_canonical_csr` always copies each input once (CSR, float32 for similarities / the input's float type for normalizations, int32, sorted, summed, no zeros); every later prep step assumes that form and may change the arrays in place. Prefer one copy over aliasing tricks in prep (cold code: clarity first)

### Rust Development
- `uv run` (e.g. `make test-dev`) rebuilds the extension after `.rs` edits (`[tool.uv] cache-keys`)
- The Python prep guarantees the kernel's invariants (int32, sorted rows, no explicit zeros); the core still validates once per call, the hot loops don't
- Release the GIL (`py.detach`) around kernel calls; free-threaded builds (3.14t+) are supported
- Threads: `num_threads=0` = rayon global pool (`RAYON_NUM_THREADS` to limit), `n` = a pool cached per calling thread
- Ensure no shared mutable state across rayon workers
- 0.7.0 (the last Cython release) is a tolerance reference, not a contract: `check_kernel_parity.py` / `check_normalization_parity.py` (`--dump` / `--compare`, see [tests/benchmarks/README.md](tests/benchmarks/README.md))

---

## Version & License
- **License:** MIT
- **Python Support:** 3.11+
