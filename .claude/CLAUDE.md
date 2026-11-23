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
  - [_s_plus.py](similaripy/_s_plus.py) - glue, no numpy prep: argument checks, `Sparse` / `Params` for the one `_core.similarity` call (canonical copies + the whole pipeline), output wrap; `_canonical_csr` (the door the normalizations and serving use)
  - [normalization.py](similaripy/normalization.py) - Normalization functions (BM25, TF-IDF, etc.): canonical copy + one core op (`axis` 1 or 0, no transposes) + wrap
  - [preprocessing.py](similaripy/preprocessing.py) - Interaction logs → URM + id maps (`interactions_to_urm`), histories for `recommend_batch` from a log or a URM (`interactions_to_histories`, `urm_to_histories`), event-level `time_decay` / `rank_decay` (one weight per event, offline and online); one import path
  - [serving.py](similaripy/serving.py) - `SparseKNN` (one import path: `similaripy.serving`): thin wrapper (the Rust core builds the model), `view` masks, `recommend` / `recommend_batch`, `save` / `load`

- **Rust core** ([crates/similaripy-core](crates/similaripy-core)): pure Rust, no Python
  - [kernel.rs](crates/similaripy-core/src/kernel.rs) - S-Plus kernel (single-block and blocked paths, rayon)
  - [similarity.rs](crates/similaripy-core/src/similarity.rs) - The S-Plus driver (in [similarity/](crates/similaripy-core/src/similarity)): `Options`, data rules, norms / depop / walk (p3alpha, rp3beta) / prescale, selectors, popularity layout, auto block ([block.rs](crates/similaripy-core/src/block.rs)), `compute`; canonical `Csr<f32>` in, top-K out
  - [normalize.rs](crates/similaripy-core/src/normalize.rs) - Normalizations over canonical float32 CSR data, in place, `Axis` rows / cols (column stats: one serial pass in row order; sums, IDF, logs in float64); `Decay` (the one decay formula, for a future server too) + `user_ages`
  - [prep.rs](crates/similaripy-core/src/prep.rs) - `canonical` (CSR / CSC / COO, f32 or f64 values, i32 or i64 indices → one owned canonical float32 CSR), canonical rows (`canonical_row`, shared with the serving build), renumber + sort, `group_interactions` (logs → CSR: counting sort by user, rows in parallel)
  - [csr.rs](crates/similaripy-core/src/csr.rs), [topk.rs](crates/similaripy-core/src/topk.rs) - CSR views, top-K buffer (cut by select)
  - [serve.rs](crates/similaripy-core/src/serve.rs) - Serving engine, in [serve/](crates/similaripy-core/src/serve): `model.rs` (build: trim to `candidate_items`, then `ItemOrder` (clustered by default, D115/D116), the one check `Core::from_arrays`, save/load, views), `ids.rs` (id maps, masks), `request.rs` (`Worker`: one serving thread's scratch, block size resolved once, stats, reused buffers; the free functions = a thread-local worker; batch = rayon over the same function). Block size = a runtime setting, never saved (W1)
  - [store.rs](crates/similaripy-core/src/store.rs) - Saved models: `.npy` reader/writer, mapped (`memmap2`) or owned arrays, saves into a new or empty directory only
- **Bindings** ([crates/similaripy-py](crates/similaripy-py)): PyO3 module `similaripy._core`
  - [lib.rs](crates/similaripy-py/src/lib.rs) - `similarity`, `canonical` and prep bindings, thread pools; [normalize.rs](crates/similaripy-py/src/normalize.rs) - dtype dispatch; [progress.rs](crates/similaripy-py/src/progress.rs) - progress bar; [serve.rs](crates/similaripy-py/src/serve.rs) - `ServeModel` (model + its block size: resolved once for single calls, per call for batches)
- **Server** ([crates/similaripy-server](crates/similaripy-server), `publish = false`, not in the wheel): experimental, released as its own artifact (`server-v*` tags: binaries + GHCR image, [server-release.yml](.github/workflows/server-release.yml)); user docs [docs/server.md](docs/server.md), [CHANGELOG](crates/similaripy-server/CHANGELOG.md)
  - [lib.rs](crates/similaripy-server/src/lib.rs) - axum router (`POST /v1/recommend`: items + context (model name, request id from `x-request-id`, k, sorted, view, history counts), `Server-Timing`; `GET /metrics`, `GET /health`, `GET /config`), N threads (current-thread tokio runtime + its own warmed `Worker` each), round-robin acceptor; JSON weights parsed as f64 then f32 (numpy's rounding); [metrics.rs](crates/similaripy-server/src/metrics.rs) - per-thread single-writer atomics, Prometheus text; [config.rs](crates/similaripy-server/src/config.rs) - `Config` (defaults < TOML file < `SIMILARIPY_<SECTION>__<KEY>` env < `--section.key` flags, one deserialize with `deny_unknown_fields`; V1), old flags as aliases; [main.rs](crates/similaripy-server/src/main.rs) - CLI; [bin/loadgen.rs](crates/similaripy-server/src/bin/loadgen.rs) - open-loop load generator (rps at p99 <= budget, `--check` vs Python's answers)

**Config:**
- [pyproject.toml](pyproject.toml) - Dependencies, build config (`[tool.maturin]`)
- [Cargo.toml](Cargo.toml) + [rust-toolchain.toml](rust-toolchain.toml) - Rust workspace, pinned compiler
- [.cargo/config.toml](.cargo/config.toml) - 64-byte function alignment for every build (D121, D124)
- [Makefile](Makefile) - Development commands

---

## Performance Architecture

**Key Optimizations:**
- **Float32 precision** + **Pre-allocated per-thread buffers** + **GIL-free computation** (`py.detach`)
- **Top-K filtering** with a pre-allocated buffer cut by `select_nth_unstable`, **Lazy normalization** (only when needed)
- **In-place operations** for zero-copy modifications

**Reference:** See [kernel.rs](crates/similaripy-core/src/kernel.rs) for implementation patterns, and in [benchmarks/README.md](benchmarks/README.md) the sections "Codegen-sensitive hot loops" (what a change must not break) and "Profiling the kernel (macOS)" (per-phase shares with `sample`).

**Perf tiers:** the hot path (`accumulate`, `accumulate_below`, drain/score, `TopK`) gets a perf assessment on every change; prep, output build and cold core code are correctness first, perf second.

### Reproducibility contract
- **Same version + same machine → identical results, bit for bit**, run to run, whatever `num_threads` or `block_size`. Different versions or platforms may differ by float rounding.
- New code keeps the **addition order within an output row fixed by the data**, never by thread timing: no atomic float adds, no "first finished, first combined". Intra-row parallelism only by column blocks or fixed chunks combined in a fixed order.
- **Parity check on every core change:** `benchmarks/parity_kernel.py --dump` under the base, `--compare` under the head. A refactor that claims no change: `--exact` (free, catches real bugs). Any other change: the default `--rtol 1e-4` + top-k overlap with the previous version, and a migration-notes line if users can see it. Small float differences (float32 vs float64, summation order) are fine: never add code just to keep bits identical.
- **Normal use cases first:** extreme values (e.g. data ×1e-30 or ×1e19, past float32 range) are best effort; a cheap check on a realistic failure is fine, dedicated code paths are not.

### Index widths
- Column ids are **int32** (output column ids too). Offsets: the Rust core reads `indptr` as i64 only, so the prep's `Csr` copy is always int64; outputs keep int32 while nnz ≤ int32 max and int64 past it (`_indptr_dtype`, scipy's rule, which makes a scipy matrix's indices int64 too).

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
Harness in [benchmarks/](benchmarks/README.md): `run.py` (similarity, recommend, small, serve; `--rss`), `abba.py` (two builds, alternating processes), `compare.py`, parity scripts, `server_prep.py` (the server's loadgen inputs); datasets cached in `~/.cache/similaripy-bench`. Its README has the protocol (which runs for which change).

**Test files:**
- [tests/test_similarity.py](tests/test_similarity.py) - every similarity × shrink type against one dense float64 reference (`_dense_scores`), selectors, signed input
- [tests/test_s_plus_prep.py](tests/test_s_plus_prep.py) - validation, canonicalization, layout, output build, regressions
- [tests/test_preprocessing.py](tests/test_preprocessing.py) - `interactions_to_urm` vs a dict reference (duplicates × timestamps × vocabularies), id dtypes, both histories paths (batch == single, log == URM scores)
- [tests/test_decay.py](tests/test_decay.py) - `time_decay` / `rank_decay` vs a Python reference, offline-weighted URM == online-weighted history
- [tests/test_normalization.py](tests/test_normalization.py), [tests/test_progress_bar.py](tests/test_progress_bar.py)
- [tests/test_free_threading.py](tests/test_free_threading.py) - concurrent calls, identical bits across threads and blocking
- [tests/test_benchmarks.py](tests/test_benchmarks.py) - the benchmark harness on the synthetic dataset (every workload, params recorded == passed, ABBA + compare)
- [tests/test_serve.py](tests/test_serve.py) - `SparseKNN` vs a dense float64 reference, masks and views, unknown ids, batch == single bits, pickle, save/load round trips + corrupted files + a committed format-1 model (`tests/data/sparseknn-format1`)
- [crates/similaripy-server/tests/http.rs](crates/similaripy-server/tests/http.rs) - the server over loopback HTTP: same ids + score bits as core `recommend` as Python calls it, JSON f32 rounding, edge rejections, a panicking connection, metrics, config (sorted default, history reject, `/config`)
- [crates/similaripy-core/tests/serve.rs](crates/similaripy-core/tests/serve.rs) - serving proptest vs an f64 reference, bit-identity claims, every `ItemOrder` and trim vs a reference order, one `Worker` across models / views / blocks, stats (panic mid-request: unit test in `request.rs`)
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
- Inputs are canonicalized before use (summed duplicates, no explicit zeros: `_core.canonical`, one copy; the similarities do it inside `_core.similarity`, the normalizations and serving through `_canonical_csr`); never mutate the caller's matrices
- **Prep owns its data:** the core copies each input once into canonical arrays (float32 for every caller, int32 indices, int64 indptr, sorted, summed, no zeros; values cast before duplicates are summed); every later step assumes that form and may change the arrays in place. Prefer one copy over aliasing tricks in prep (cold code: clarity first)

### Saved models
- **Format 1 is frozen at 1.0:** 1.x loads every older 1.x format. Changing what a saved file means bumps `FORMAT` (`serve/model.rs`) and keeps a reader for the old one; `tests/data/sparseknn-format1` is never regenerated after 1.0.
- Pickles are same-version only (`_PICKLE_FORMAT` in `serving.py`): a change to `parts()` bumps it.

### Rust Development
- `uv run` (e.g. `make test-dev`) rebuilds the extension after `.rs` edits (`[tool.uv] cache-keys`)
- The Python prep guarantees the kernel's invariants (int32 indices, sorted rows, no explicit zeros); the core still validates once per call, the hot loops don't
- Release the GIL (`py.detach`) around kernel calls; free-threaded builds (3.14t+) are supported
- Threads: `num_threads=0` = rayon global pool (`RAYON_NUM_THREADS` to limit), `n` = a pool cached per calling thread; every rayon-running binding goes through `with_pool`, which in a forked child swaps the global pool for one per-process pool and rebuilds cached pools (pools don't survive `fork()`; `tests/test_fork.py`)
- Ensure no shared mutable state across rayon workers
- 0.7.0 (the last Cython release) is a tolerance reference, not a contract: `parity_kernel.py` / `parity_normalization.py` (`--dump` / `--compare`, see [benchmarks/README.md](benchmarks/README.md))

---

## Version & License
- **License:** MIT
- **Python Support:** 3.11+
