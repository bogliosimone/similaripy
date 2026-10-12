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

- **Rust core** ([crates/similaripy-core](crates/similaripy-core)): pure Rust, no Python
  - [kernel.rs](crates/similaripy-core/src/kernel.rs) - S-Plus kernel (single-block and blocked paths, rayon)
  - [normalize.rs](crates/similaripy-core/src/normalize.rs) - Normalizations, in place over CSR data
  - [csr.rs](crates/similaripy-core/src/csr.rs), [topk.rs](crates/similaripy-core/src/topk.rs) - CSR views, top-K heap
- **Bindings** ([crates/similaripy-py](crates/similaripy-py)): PyO3 module `similaripy._core`, thread pools, progress bar

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

**Reference:** See [kernel.rs](crates/similaripy-core/src/kernel.rs) for implementation patterns, and the codegen-sensitive loops in [tests/benchmarks/README.md](tests/benchmarks/README.md).

---

## Development Workflow

### Package Manager & Build System
Uses **uv** for package management. Built with **maturin**. Requires the Rust toolchain (rustup; `rust-toolchain.toml` pins the version).

### Setup
```bash
# Editable install with dependencies; `uv run` rebuilds it after Rust edits
make install-dev
```

### Testing
```bash
# Local tests (fast, correctness only)
make test-dev

# Rust core
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

### Benchmarking
```bash
# Small dataset for quick regression test (Movielens 32M)
make benchmark-similarity-small
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
1. Run `make test-dev` (and the cargo checks for Rust changes) to verify compilation and unit tests
2. Suggest test updates for new features or uncovered bugs

---

## Common Gotchas

### Sparse Matrices
- All functions expect **CSR format** (converted automatically)
- **Always call `eliminate_zeros()`** in the init phase to maintain sparsity and avoid binary flag issues

### Rust Development
- `uv run` (e.g. `make test-dev`) rebuilds the extension after `.rs` edits (`[tool.uv] cache-keys`)
- The Python prep guarantees the kernel's invariants (int32, sorted m2 when blocking, no explicit zeros); the core still validates once per call, the hot loops don't
- Release the GIL (`py.detach`) around kernel calls; free-threaded builds (3.14t+) are supported
- Threads: `num_threads=0` = rayon global pool (`RAYON_NUM_THREADS` to limit), `n` = a pool cached per calling thread
- Ensure no shared mutable state across rayon workers
- Parity with 0.7.0 (Cython): `check_kernel_parity.py` / `check_normalization_parity.py` (`--dump` / `--compare`, see [tests/benchmarks/README.md](tests/benchmarks/README.md))

---

## Version & License
- **License:** MIT
- **Python Support:** 3.11+
