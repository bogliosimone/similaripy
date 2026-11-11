# SimilariPy benchmarks

The benchmark harness: timing, peak memory and A/B comparisons of similaripy builds, plus the parity checks against a previous version. Scripts run by path from any checkout and against any install, 0.7.0 (the last Cython release, from PyPI) included.

| File | What it does |
|---|---|
| `run.py` | Benchmark one build: one workload on one dataset, a JSON report |
| `abba.py` | Two builds, alternating processes (ABBA), the change per pair; several candidates share one base (`--cand NAME=PY`, rotated order) |
| `gate.py` | The core's gate: `abba.py` on a fixed list of steps (quick / full), never idle, one verdict per case |
| `compare.py` | Reports side by side, the first as the reference |
| `parity_kernel.py`, `parity_normalization.py` | Outputs of this install vs a reference install's (`--dump` / `--compare`) |
| `loaders.py` | Datasets (synthetic, MovieLens, Yambda, H&M, OTTO), cached as `.npz` |
| `workloads.py` | The cases of each workload and the exact kwargs of each call |
| `libraries.py` | Other libraries' item-item similarity for the `crosslib` workload (see "Other libraries") |
| `crosslib-requirements.txt`, `crosslib-lock-*.txt` | Their pins, and every dependency pinned per platform |
| `compat.py` | Everything that differs between builds (0.7.0 vs v1) |
| `report.py` | System info, preflight, the round loop, peak memory, the JSON format |

`tests/test_benchmarks.py` runs every workload on the synthetic dataset in `make check` and CI, so the harness can't rot unnoticed.

## Quick start

```bash
make install-dev   # the bench dependency group: pandas

# Item-item similarity on MovieLens 32M (dot_product, cosine, rp3beta), 1 warm-up + 1 timed round
uv run --group bench python benchmarks/run.py similarity --dataset movielens

# Other similarities, more rounds, a tag and a note in the report
uv run --group bench python benchmarks/run.py similarity --dataset yambda --version 50m \
    --similarities cosine rp3beta_tuned --rounds 3 --tag before --note "baseline"

# Recommendation step: dot_product(URM, S) with and without filter_cols=URM
uv run --group bench python benchmarks/run.py recommend --dataset movielens

# Millisecond calls on the synthetic matrix: smoke set (3 rounds), or every case
uv run --group bench python benchmarks/run.py small
uv run --group bench python benchmarks/run.py small --cases all

# Serving (ItemKNN): request latency, batch vs kernel, block size, model build
uv run --group bench python benchmarks/run.py serve --dataset movielens --mode latency block build

# Peak memory per case: each case's first call in a fresh process, after the timed rounds
uv run --group bench python benchmarks/run.py similarity --dataset movielens --rss

# Two builds, ABBA (here: a worktree of the previous commit vs this checkout)
python benchmarks/abba.py --base ~/similaripy-wt/base/.venv/bin/python --cand .venv/bin/python \
    --tag my-change -- similarity --dataset yambda --version 50m --similarities cosine

# Compare reports
python benchmarks/compare.py benchmarks/bench_results/before_*.json benchmarks/bench_results/after_*.json
```

`python benchmarks/run.py <workload> --help` lists every option.

## Workloads

- **similarity**: `sim.<name>(URM.T, k=100, ...)` for every similarity (`dot_product`, `cosine`, `asymmetric_cosine`, `jaccard`, `dice`, `tversky`, `p3alpha`, `rp3beta`, `splus`) and fixed-parameter variants: `dot_product_binary`, `cosine_binary`, `cosine_bayesian`, `cosine_stabilized`, `dot_product_bayesian`, `p3alpha_tuned`, `p3alpha_bayesian`, `p3alpha_binary`, `rp3beta_tuned`, `rp3beta_binary`, `rp3beta_stabilized`, `rp3beta_bayesian` (placeholder values: shrink 10, alpha 0.8, beta 0.6, in `workloads.py`). Options: `--k`, `--shrink` / `--threshold` (passed only when set), `--block-size auto|none|N`, `--column-order popularity|original` (private kernel switch), `--threads`, `--format-output`.
- **recommend**: `dot_product(URM, S, k)` as one library batch call over all users, `recommend` and `recommend_filter_seen` (`filter_cols=URM`); S = `cosine(URM.T, k=--model-k)`, built once, untimed.
- **small**: millisecond calls on the seeded synthetic 10k x 10k matrix (~1M nnz, Zipf-like popularity), where fixed per-call costs (prep, binding, thread start-up, output build) show. Cases: k=10/1000, `binary=True`, `block_size=None` / 2048, array and matrix `filter_cols` / `target_cols`, and two concurrent cosine calls (`num_threads=2` each) vs the same calls serially (meaningful on a free-threaded build, 3.14t). One measurement = the median of `--reps` calls (default 21). The default smoke set catches big (>20%) regressions only.
- **serve**: `similaripy.serving.ItemKNN` on S = cosine or rp3beta (`--model`, `--model-k`). Requests are URM rows (user histories), `--requests` resampled with replacement in random order (`--seed`); a history cap keeps that many random items of the row; every request excludes its own items. Modes:
  - `latency`: `recommend` end to end per request, µs p50/p99/mean (p99.9 from 100k requests) per history cap (`--caps`), warm (back to back) and cold (`--evict-mb` read before each request);
  - `block`: the core batch call over every user (prep outside the timing) with the default block size vs one block per request, on `--block-threads` (default 1 and all cores, `info.block_threads`; concurrency changes the answer, D65). `info.block_control`: the catalog fits one block, both run the same. (The Step 4 batch-vs-kernel `gate` mode was removed after D64 settled it.)
  - `build`: `ItemKNN(S)` time and peak memory in a fresh process (S saved to a temp file first). Peak memory is `phys_footprint` (see below), not the `ru_maxrss` of reports before D75: not comparable with them.

  `--item-order` builds every model in that `item_order` (default: `SIMILARIPY_BENCH_ITEM_ORDER`, else not passed: the library's, so a base without it still runs; set on such a base: an error); `--transpose` serves `ItemKNN(S.T)`; `--candidates N` trims every model to the N most popular items (`candidate_items`: a smaller catalogue on real data).

  Serving times the core batch call through private API (`compat.py`, the one place to edit if it changes).

  `--model-cache` (off by default): S is built by the first run and saved in the bench cache (`models/{dataset}-{model}-k{model_k}.npz`, a plain scipy CSR, plus the build that made it in a `.json`; the report's `info.model_cache` names both). Every later process and every build serves that same S: no ~1 min Y500M build per process, and a serving A/B isn't moved by kernel changes. Each build still builds its own `ItemKNN(S)`, so no serving format is shared. Delete `models/` when S's definition changes.

- **crosslib**: another library's item-item similarity on the same URM (`--library`, one per process; `--check` compares the output with similaripy). See "Other libraries".

Every call runs with `verbose=False` (no progress bar in the timed region) and the same `--format-output` (default csr; 0.7.0 defaults to coo).

## Datasets

| Dataset | Versions | Source |
|---|---|---|
| `synthetic` | `10k` | seeded, built on every run (`loaders.make_urm`) |
| `movielens` | `32m` (default), `25m` | downloaded from grouplens.org; ratings as values |
| `yambda` | `50m` (default), `500m`; `--event-type likes / listens / multi_event` | Hugging Face `yandex/yambda` (local HF cache first); implicit, repeated events summed. The first build needs `--with datasets` (not in the bench group: it pulls pyarrow) |
| `hm` | `full` | Kaggle "H&M Personalized Fashion Recommendations" `transactions_train.csv`: download it yourself (account + competition rules) into `$CACHE/h-and-m/`; purchase counts. Internal only until the competition rules are checked for publishing |
| `otto` | `full` | Kaggle `otto/recsys-dataset` (CC-BY 4.0) `otto-recsys-train.jsonl` (11.3 GB; the zip of both files is 12 GB unpacked): download it yourself (account) and unzip into `$CACHE/otto/`; sessions as users, clicks + carts + orders counted alike. First build reads ~217M events (a few minutes) |

Real datasets are built once and cached as `.npz` in **one directory shared by every checkout and worktree**: `~/.cache/similaripy-bench`, or `SIMILARIPY_BENCH_DATA`. Every side of an A/B then loads the same bytes, and 0.7.0 needs no pandas. Each cache is checked against the shape and nnz in `loaders.EXPECTED`: a changed source or loader fails the load instead of timing other data (add a new dataset's numbers there after its first load). Indices are int32 whenever they fit.

## Reports

One JSON format for every workload (`schema_version` 1), in `benchmarks/bench_results/` (gitignored) or `--output-dir`, named `{tag}_{workload}_{dataset}_{timestamp}.json`:

- `harness.git`: the commit of these scripts (the harness is frozen before publishable numbers; a later change bumps the plan's Deviations, and the schema version if fields change);
- `build`: the timed similaripy: version, path, editable source tree or wheel, the git state of *that* tree (v1 builds report `__version__` 0.7.0 until 1.0: use `build.path` / `build.git`);
- `system`: Python + GIL, numpy, scipy, OS, CPU, cores, RAM, rustc, `RAYON_NUM_THREADS`;
- `preflight`: 1-min load before and after, power source, the busiest processes;
- `config`: every option of the run; `dataset`: key, shape, nnz; `info`: facts about the setup (S nnz, active items, ...);
- `cases.<name>`: `params` (**exactly the kwargs passed**; arrays and matrices described), `median` per phase, `spread` ((max − min) / median of the total), extras (output nnz, latency percentiles, `rss_base_mb` / `rss_peak_mb`), and every round's raw numbers.

Phases: **total** wall time of the public call; **kernel** (scoring, top-K, per-row un-permute + sort, compaction) and **output** (matrix wrap) from `similaripy._s_plus._last_timings`; **prep** = the rest (wrapper normalization + numpy prep). 0.7.0 has no phase timer: only total.

Reports from before the rebuild (`tests/benchmarks/`, no `schema_version`) aren't read; the old harness is in git history.

## Method

- **Rounds:** `--warmup` untimed rounds (default 1), then `--rounds` timed ones (default 1; small: 3). Within a process the cases interleave, order reversed every other round (ABBA), so drift hits all equally. The median is the number; check `spread` before trusting a difference.
- **Params recorded = params passed.** A harness option reaches a call only when it is set, and an option that a fixed-parameter entry also sets is an error. (The old harness passed `shrink=0` to every call, which silently overrode a `functools.partial`'s shrink and timed plain cosine.)
- **Preflight:** a load above `--max-load` (default 3.0) or running on battery prints a warning and stores it in the report; `--strict` refuses to run. Check before timing: background apps, video calls and post-build Spotlight / `mediaanalysisd` runs move results by 5–25%.
- **Peak memory (`--rss`):** each case's first call in a fresh spawned process, sampling every 1 ms. On macOS `phys_footprint` (Activity Monitor's "Memory"; RSS keeps freed pages, +1 MB for an ML-32M call), elsewhere psutil RSS (`pip install psutil`). A repeated call reuses the allocator's cached regions without counting them again: only a first call shows its peak.
- **ABBA across builds (`abba.py`):** one process per (side, pair), order base-cand then cand-base. Both sides run this checkout's scripts. A worktree side is refused when its extension is older than its Rust sources (`.venv/bin/python` doesn't rebuild the way `uv run` does). `pypi:X` sides pin numpy/scipy to the running interpreter's versions. The dataset is cached first, by the running interpreter. Several candidates (`--cand NAME=SPEC` repeated) share one base run per pair, order rotated (base, c1..ck, then ck..c1, base): k comparisons for k+1 runs per pair instead of 2k; one `abba_{tag}-{NAME}.json` each.

## The gate (`gate.py`)

```bash
uv run --group bench python benchmarks/gate.py quick --base ~/wt/base/.venv/bin/python \
    --cand .venv/bin/python --tag my-change --until 07:30
```

- **Priorities (D90):** 1. online serving (`ItemKNN.recommend`: most requests/s at the least CPU, p99 under load), 2. batch serving (`recommend_batch`: throughput), 3. offline similarity (Y500M decides; small-dataset losses OK for simpler prep). Shared code (kernel, `TopK`, block size) is judged in that order: a tier-1 loss blocks, a few % tier-3 loss is acceptable for a tier-1 win. `dot_product` for recommendations is discouraged: a control, no perf tier.
- **quick** (~55 min): Y50M then Y500M, kernel (one workload per scoring mode: `cosine` = DOT, as `rp3beta` and `dot_product`; `cosine_stabilized` = COSINE; `rp3beta_bayesian` = FULL) and serving batch (`--model-cache`), `--threads` (default 6, the M2 Pro's performance cores: E-cores add noise), then concurrent callers (Y50M, Y500M) and Y500M single-request latency. **full** (~85 min, once per PR): + Y500M serving at 1 thread, ML-32M kernel / serving / latency, model build, Y50M `dot_product(URM, S)` (control). `--extra STEP ...` adds some full-tier steps to a quick run (e.g. `--extra ml32m-kernel y500m-lat`: what a change targets), after the tier's steps.
- **Never idle:** no waiting for a quiet machine. A sampler records the load and the CPU used by every process outside the gate every 10 s (CPU time differences, the same on macOS and Linux; a VM's steal time isn't seen) in `timeline.tsv`; a step with > 100% such CPU in > 10% of its samples is rerun once at the end. `--until HH:MM`: a step starts only if its estimate (scaled by `--pairs`, ≥ 2) ends before. A failed or skipped step is a row of the verdict (exit 1 on a failure); an output dir that already holds a run is refused.
- **Output** (`--output-dir`, default `bench_results/gate-{tag}`): abba.py's reports per step, `steps.tsv` (time, load, burst share per step), `verdict.md`: per case and metric (the kernel phase, else total; plus `p99_us` and `cpu_us_per_request` where recorded), the change in every pair and win / loss (same sign, each ≥ 3%), small (same sign, some pair below), neutral (mixed), noisy.
- macOS and Linux (the sampler reads `ps` and the load average; on Windows it's down and the verdict says so).
- `--dry`: every step on the synthetic dataset, to check the plumbing (`tests/test_benchmarks.py` runs it).

## Which runs for which change

| Change | Run |
|---|---|
| Iterating | `run.py similarity --dataset movielens --similarities cosine` |
| Kernel / serving hot path (`accumulate`, `accumulate_below`, drain, `TopK`, the blocked loops) | `gate.py quick` vs the previous commit (~55 min; Y500M decides); `gate.py full` once per PR. A change counts when every pair shows it with the same sign, ≥ 3% (D40 P6) |
| Prep, output build, Python wrappers | ML-32M + Y50M total, `small --cases all` |
| Memory | `--rss` (fresh process per case) |
| Serving latency, block size | `serve --mode latency` on ML-32M and Y500M (in `gate.py full`, or `--extra`); `serve --mode block` at 1 thread and the core count (D65) |
| rustc upgrade | see "Codegen-sensitive hot loops" |
| Other libraries (adapter, pin bump) | `make bench-crosslib-test`, then `crosslib --check` on ML-32M per library (see "Other libraries") |
| Release history / other versions | `abba.py --base pypi:0.7.0 ...` (total only; the PyPI wheel is not the `v1-cython-ref` local build behind older "vs cpp" numbers: different compiler and flags, not comparable) |

## Other libraries

The `crosslib` workload times other libraries' exact sparse top-k item-item similarity next to similaripy (D78). The libraries come from a pinned overlay, never a dependency group (LensKit pulls torch and needs numpy >= 2; the minimum-versions CI job runs numpy 1.23.5):

```bash
# One library per process: OpenMP runtimes (implicit, sklearn, torch) clash in one process on macOS
uv run --group bench --with-requirements benchmarks/crosslib-lock-darwin-arm64.txt \
    python benchmarks/run.py crosslib --dataset movielens --library sparse_dot_topn --threads 6 --check
# The similaripy rows: the similarity workload, under the same lock (same numpy / scipy as every library)
uv run --group bench --with-requirements benchmarks/crosslib-lock-darwin-arm64.txt \
    python benchmarks/run.py similarity --dataset movielens --similarities cosine dot_product cosine_binary --threads 6
# The adapters' tests under the pins (nothing in CI runs them): at every pin bump
make bench-crosslib-test
```

One shared environment for every library, isolated per process: the shared dependencies resolved to their latest releases on 2026-11-02, so a venv per library would install the same versions. At a pin bump, compare `info.dependencies` of the new reports with each library's own resolve; a library that holds a shared dependency back gets its own lock. Locks: `crosslib-requirements.txt` holds the top-level pins and the recompile commands (macOS with `MACOSX_DEPLOYMENT_TARGET=15.0`: uv's default target resolves torch 2.9.1). sp_matmul_rs has no 3.14 wheel and builds from the sdist (Rust toolchain at install). The Linux lock has the PyPI (CUDA) torch: a large download, only LensKit imports it.

| Library | `cosine` | `dot_product` | `cosine_binary` | Self in top-k | Threads | Traps |
|---|---|---|---|---|---|---|
| sparse_dot_topn 1.2.0 | rows L2-normalized by the caller (timed) | plain | yes | yes | `n_threads` | rejects `csr_array` (the transpose is built as a `csr_matrix`, timed); `n_threads=0` means 1 |
| sp_matmul_rs 0.2.1 | as sparse_dot_topn | plain | yes | yes | `n_threads` (rayon) | as sparse_dot_topn |
| implicit 0.7.3 | `CosineRecommender` | `ItemItemRecommender` on float64 (cast timed) | yes | yes (`K=k`) | `num_threads`; **the macOS wheel has no OpenMP** (single-threaded) | float64 output; `fit` also builds its scorer |
| scikit-learn 1.9.1 | `NearestNeighbors(metric="cosine", algorithm="brute").kneighbors(X)` | – | yes | yes | `n_jobs` (0 raises; only the distance chunks run in parallel) | rows always hold k entries (zero fillers dropped for the check); float32 `1 − d`: check atol 1e-6 |
| LensKit 2026.4.0 | – | – | `ItemKNNScorer(feedback="implicit")` | **no**: `save_nbrs = k − 1` | rayon, set once per process | no CSR constructor: Dataset built from the COO arrays, timed and shown as `dataset_s`; `min_sim` 1e-6 |
| scipy (baseline) | `cosine_similarity(X, dense_output=False)` + per-row top-k (RecPack's ItemKNN) | `X @ X.T` + top-k | yes | yes | none: `--threads 1` only | full product in memory (OOM on large catalogs is the result) |

Rules:
- **Same input:** every library starts from the loader's users x items CSR (float32, int32); `cosine_binary` binarizes it before the timer (`params.input: binarized`).
- **Total time** = from that CSR to the library's own output object, its prep (transpose, normalize, casts, Dataset) included; conversions for the check are outside. No kernel column across libraries (each library's "kernel" is a different cut); the timer is `untimed()`, never `timed()` (similaripy's phase timer would be stale).
- **Threads:** `--threads 0` becomes `os.cpu_count()` once, and that explicit count goes to every library in its own spelling (`info.threads`). Counts: 1 and the machine's P-cores (D78 Q8). `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` unset.
- **Check (`--check`):** the first call's output (warm-up included) against similaripy on the same input, outside the timer, with the parity tie rule (`parity_kernel.compare`) at **rtol 1e-3** (`libraries.RTOL`; the parity scripts keep 1e-4): two float32 libraries summing ~1e5-term rows differ by up to ~6e-4 (on ML-32M similaripy's own cosine diagonal is off by up to 5e-4 vs float64), while a mapping mistake shows at 1e-2 and more; every round carries the result as `check = {overlap, shared, one_sided, failures, first_failures}`. Pass = zero failures (exit 1 otherwise). Timings to publish come from runs without `--check` (the reference build sits between rounds). Not on Y500M (two outputs in memory). `tests/test_benchmarks.py` also checks every library against a dense float64 reference.
- **Peak memory (`--rss`):** unchanged; the children never run the check.
- **Report:** `info` = library, version, threads, shared dependency versions; the default tag is the library name.

## Codegen-sensitive hot loops (Rust core)

These speedups in `crates/similaripy-core/src/kernel.rs` and `topk.rs` depend on what LLVM emits, and no unit test can see them (the output stays the same either way):

- The block end has no search: a row crossing it stops inside `accumulate_below` (same body as `accumulate` plus one compare per entry), and a row with nothing in the block is skipped on its cached next column (kernel and serving, perf gates 2026-11-01). A search brought back here must stay **branchy**: the old `lower_bound` needed an empty `asm!` barrier, since LLVM turned it into a branchless `csel`/`cmov` loop that serializes cache misses once the data no longer fits in cache.
- `accumulate` must keep `sums` and the `touched` cursor in registers. As a `&mut self` method, LLVM reloaded them from the stack on every element.
- `accumulate`'s first-touch push must stay **branchless**: always write the slot, then `n += (old == 0.0)`. On Yambda ~39% of elements are first touches, so a branch there is mispredicted all the time; replacing it cut kernel time by ~20% on Yambda 50M. Check for `cinc` (aarch64) / `cmpeqss` + `and` + `add` (x86_64) and no branch on the old sum.
- `TopK::cut_to_k` (the `select_nth_unstable` cut, once per k accepted pushes) must stay **out of line** (`#[cold] #[inline(never)]`: one symbol, a `bl`/`call` in every drain copy, 11 sites on aarch64 and x86_64 at D89). Its predecessor, the heap's sift-down, showed why: inlined into the drain loops, no run gained and some were up to ~8% slower (likely register pressure in the hottest loop).
- `drain` branches once per drain on `a1 != 1.0` (`POW`) and on the scoring mode (`DOT` / `COSINE` / `FULL`), so the loop has no `powf` call and no per-column weight tests for dot and cosine. Inside the loop LLVM speculates `powf` and calls it on every column; a bayesian-shrink test in the small `COSINE` loop gets if-converted into a second `fdiv` on every column, which is why any bayesian shrink goes to `FULL`. Check: no `bl powf` in the a1 == 1 copies, one `fdiv`/`divss` in the cosine copy.
- `Scratch::drain` is `#[inline(never)]` and its loop is a free function over slices and locals (`drain_impl`), like `accumulate`: inlined into the row functions, or as a `&mut self` method, LLVM reloaded the `sums` pointer, the row's norm and the top-K's state on every column. The row's norm terms (`RowTerms`) and the lowest acceptable score (`TopK::floor`) live in registers. A cosine column rejected below the floor is ~18 instructions with 3 loads (touched index, sum, column norm); check that no load of `sums`' pointer or of the top-K precedes the reject compare.
- Serving's history-row loop prefetches the row `PREFETCH_ROWS` (4) ahead (`prfm` / `prefetcht0`, one hint per line, up to 16 lines), and each row's accumulate is the out-of-line `accumulate_row`: with the prefetch inlined into the row loop, x86_64 spilled the block end and re-read it from the stack on every element of the bounded loop (aarch64 unchanged). Check: no `(%rsp)` / `[sp` operand in `accumulate_row`'s two loops; one `prfm` / `prefetcht0` in `recommend_internal`.
- The kernel's row loops (`single_block_row`, `blocked_row`) prefetch the m2 row `PREFETCH_ROWS` (8) ahead, up to 8 lines of each array (`prefetch_span`), with the same `kernel::prefetch` as serving. Long rows stream on the hardware prefetcher, so short rows gain most (ML-32M). Check: the accumulate loops keep their stack operands (x86_64: 0 in the whole-row loops, 3 in the bounded one, as before the prefetch); two hints per line in each row function.
- `TopK::push` is inlined: it rejects below the floor with one compare (`!(score >= min)`: `fcmp; b.lt` / `ucomiss; jb`), else the entry's order key (4 integer instructions: `fmov`/`asr`/`eor`/`eor` + `bfi` on aarch64), one 8-byte store and a length compare; ties are decided at the cut, by column. The buffer holds u64 keys (`topk::key`: score in `total_cmp` order, inverted, above the column), so the cut and the final sorts are integer `select_nth_unstable` / `sort_unstable`: serving latency −6..−15% at caps 10-50 vs (score, col) pairs with a `total_cmp` comparator (D2, 2026-11-06). The gain is the compare itself (`total_cmp` was integer bit-twiddling plus a column tie-break per step, never `fcmp`). Check: the reject loops unchanged; `cut_to_k` and the drains call the u64 sort/select (mangled `...y`), no `Entry` comparator.

The first two only show on Yambda 500M (ML-32M and Yambda 50M mostly fit in cache); the first-touch branch and the drain shape show at both Yambda sizes. When you upgrade rustc or edit these loops, rerun Yambda 50M and the 500M comparison on an idle machine against the previous build (a `git worktree` of the last good commit, its own venv), alternating the two processes:

```bash
python benchmarks/abba.py --base ~/similaripy-wt/base/.venv/bin/python --cand .venv/bin/python --tag rustc-1.x \
    -- similarity --dataset yambda --version 500m --similarities dot_product cosine --warmup 0 --strict
```

One exception: a rustc point release (`1.x.y`) whose kernel and top-K disassembly is identical to the previous compiler's, on aarch64 and x86_64, once addresses and mangled-name hashes are masked, runs the same instructions. Record that check instead of timing.

When the Rust kernel became slower than cpp at 500M, the cause was `lower_bound` and the `accumulate` register reloads (+13..16% before the fix, −7..−11% after). The block cursor (each m2 row read once, front to back, per output row) took it to −13..−17%. The scoring modes and the drain over locals took a further −13..−18% (cosine) and −9..−12% (dot) of 500M kernel time; the one-compare root reject measured as noise.

## Profiling the kernel (macOS)

Per-phase shares (accumulate, drain/score, top-K, row setup) come from `/usr/bin/sample` on a **stripped plain-release** build: the `[profile.release]` in `Cargo.toml`, which has no debug info, plus `strip -x` to drop the local symbols. With debug info (even after `strip -x`, since `sample` finds it by UUID) or with local symbols, `sample` collapses the inlined kernel into one symbol. Stripped, it reports raw offsets, one per instruction.

1. Build the release extension (`make install-dev`) and keep an unstripped copy for disassembly. Then `strip -x similaripy/_core.cpython-*-darwin.so`. Run Python from `.venv/bin/python` directly: `uv run` rebuilds over the stripped file.
2. In a script: load the dataset, start `/usr/bin/sample <pid> <seconds> -file out.txt` **before** the call, and give it more seconds than the call takes. Sample the whole call: row order isn't random (popularity-ordered rows), so a time window biases the shares.
3. Read the "Sort by top of stack" section. It gives samples per `_core` offset. Bucket the offsets into phases with address ranges read from `objdump -d --start-address=.. --stop-address=..` of the unstripped copy (symbols from `nm -n`). `single_block_row`, `blocked_row` and `TopK::cut_to_k` are own symbols; accumulate, `accumulate_below` and drain are inlined into them. Leave rayon's park/steal code out of the denominator.
4. Samples land on the instruction *after* a stall: read the pattern around a branch or a load, not one address.

`sample` slows the run by ~25–35%, so take seconds from an unsampled run. For work counts that hold on every CPU (elements accumulated, first touches, top-K pushes and cuts, floor rejections), a throwaway patch with per-worker counters is more reliable than shares; don't time that build. To check x86_64 codegen from a Mac: `rustup target add x86_64-unknown-linux-gnu`, then `cargo rustc -p similaripy-core --release --lib --target x86_64-unknown-linux-gnu -- --emit asm` (no linker needed; LTO into the extension can still change inlining).


## Parity vs a previous version

`parity_kernel.py` and `parity_normalization.py` compare this install's outputs with a reference install's, in two steps: `--dump` under the reference, `--compare` here (usage in their docstrings). Run the current script under both installs (the case lists must match), and by path: `python -c` from the repo root imports the checkout instead.

```bash
uv run --isolated --no-project --python 3.14 --with similaripy==0.7.0 \
    python benchmarks/parity_kernel.py --dump ref.npz
uv run python benchmarks/parity_kernel.py --compare ref.npz [--exact]
```

`parity_kernel.py --compare` checks each case within tolerance (rtol 1e-4, the contract for changes that move bits; a column on only one side passes only at a near-tie with the row's k-th score or the threshold) and always reports:

- `exact`: data, indices and indptr identical byte for byte. Same version + same machine must give `exact=yes`; `--exact` makes any bit difference a failure, for changes that claim exactness. For a change expected to move bits (e.g. a different float order), the tolerance check plus the dense float64 reference in `tests/test_similarity.py` is the quality check.
- `overlap`: shared entries / reference entries (top-k overlap with the previous version), reported without a threshold: tie-heavy data moves entries at the k-th score with 1-ulp changes.

References:

- **0.7.0 (the last Cython release), from PyPI.** Expected differences: `dot_bayesian` and `p3alpha_bayesian` differ by design (in 0.7.0 a bayesian shrink with no other normalization returned an empty output; fixed in 1.0); the cosine family differs by more than 1e-5 (1.0 sums pre-scaled inputs, D44). The PyPI wheel was compiled by a different GCC (CI runner, `-O3 -ffast-math`), so some scores differ by 1 float32 ulp from the local build the Rust core was matched against: near-ties at the k-th score become one-sided entries (ML-32M rp3beta: ~627k, 7%, against PyPI, 0 against the tag build below).
- **A commit:** dump from a `git worktree` with its own venv (`make install-dev`).
- **Timing history:** commit `608cc06` (local tag `v1-cython-ref`; from a fresh clone: `git fetch origin pull/29/head`), the last commit with both kernels in one process. Needs Homebrew GCC on macOS.
