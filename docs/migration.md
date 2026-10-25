# 🔁 Migration from 0.7.0

SimilariPy 1.0 replaces the Cython + OpenMP core with a Rust one. Most code
runs unchanged and gets the same top-k, up to float rounding. The changes that
can break your code come first; the full lists are at the end.

## What breaks

- Every parameter after `matrix2` is keyword-only: `sim.cosine(m, None, 50)`
  raises `TypeError`; write `sim.cosine(m, k=50)`.
- `format_output` defaults to `'csr'` (0.7.0: `'coo'`): pass
  `format_output='coo'` for the old format.
- `verbose` defaults to `False`: pass `verbose=True` for the progress bar.
- The normalizations no longer take `inplace` (`TypeError`): use the return
  value, `X = bm25(X)`. They always return a new `csr_array`.
- `block_size=0` (0.7.0's auto) raises `ValueError`; the default is `'auto'`.
- Invalid parameters raise `ValueError` instead of a silent result: negative
  weights or `threshold`, out-of-range ids, an empty `target_cols`, a shrink
  that would do nothing (see [Validation](#details)).
- Install: numpy `>= 1.23.5`, scipy `>= 1.15`; building from source needs
  Rust instead of a C compiler with OpenMP ([Build from source](installation.md#build-from-source));
  the `dev`, `bench` and `notebook` extras are gone.
- Threads: `RAYON_NUM_THREADS` replaces `OMP_NUM_THREADS`. As with OpenMP, the
  thread pool does not survive `fork()`: with `multiprocessing`, use the
  `spawn` or `forkserver` start method if the parent process called SimilariPy.

## What may change in results

- **Float rounding:** scores may differ in the last digits; up to about `1e-4`
  relative on the longest rows for `cosine`, `asymmetric_cosine` and `rp3beta`.
- **Ties** are deterministic: on equal scores the more popular column wins,
  whatever `block_size` or `num_threads`, so which of two tied items makes the
  top-k can differ from 0.7.0.
- **Output layout:** rows are sorted by column; COO output has no padding, and
  exact `0` scores are dropped in both formats.
- **0.7.0 bugs fixed:** CSR inputs with duplicate entries, unsorted rows with
  blocking, bayesian shrink on `dot_product` / `p3alpha`, `binary=True` in
  `p3alpha` / `rp3beta`, `s_plus(alpha=...)` without normalization, idf counts
  past 2^24 documents (see [Bug fixes](#details)).

## Signed input

0.7.0 did not support negative values (it did not check them either). 1.0
accepts them with a few rules per function: see [Signed Input](signed-input.md).

## Details

??? info "Validation: what now raises `ValueError`"
    - Negative weights (e.g. `l1`, `c1`, `beta`), shrinks or `threshold`;
      `alpha <= 0`; custom popularity arrays that are not finite and `>= 0`
      (`> 0` on rows/columns with entries).
    - Out-of-range ids in `target_rows`, and in `target_cols` / `filter_cols`
      lists (0.7.0 ignored them).
    - An empty `target_cols` (list, array or sparse matrix with no entries),
      which used to score every column: pass `None` for all columns. An empty
      `filter_cols` is still accepted (it excludes nothing).
    - A shrink that would do nothing: `additive` without a cosine term,
      `stabilized` on `dot_product` / `p3alpha` (it only rescaled the scores);
      see [Shrinkage Types](similarity.md#shrinkage-types).
    - `tversky` / `s_plus` (`l1 != 0`, not binary) `alpha`/`beta` pairs whose
      denominator can be `<= 0`: when `alpha + beta > 1` on weighted input (e.g.
      `(3, 0.05)`, for which 0.7.0 silently dropped pairs), and when
      `alpha + beta < 1` on signed input. No default is affected, nor binary or
      0/1 data.
    - NaN or inf values in the normalizations (they used to spread NaN through
      the row's or column's norm).
    - Norms out of float32 range in `cosine` and `asymmetric_cosine` without a
      shrink (e.g. data scaled by `1e-25` or `1e19`), in `rp3beta` when its
      `pop^beta` leaves float32 range, and in `s_plus` with a single cosine or
      depop term: rescale the input. With a shrink, such inputs still return
      empty rows.

??? info "Bug fixes that change results"
    - CSR inputs with duplicate entries are summed (scipy's canonical form) in
      every function: 0.7.0 computed wrong norms and binary counts for them.
    - Unsorted CSR rows with column blocking gave wrong results; inputs are
      always sorted now.
    - `shrink_type='bayesian'` on `dot_product` and `p3alpha` returned no
      results; it now applies the bayesian factor.
    - `binary=True` in `p3alpha` / `rp3beta` returned plain binary
      `dot_product` scores (the walk's normalization was erased); the input is
      now binarized first, then normalized. RP3β's popularity is then the item
      degree.
    - `s_plus(alpha=...)` was ignored when `l1 = l2 = l3 = 0`; it now always
      applies.
    - `tfidf` / `bm25` / `bm25plus` document counts no longer stop growing at
      2^24 documents (a float32 counter).

??? info "Other changes"
    - Exact ties on binary or integer data in `cosine`, `asymmetric_cosine` and
      `rp3beta` can differ in the last bit (they run as a dot product of
      pre-scaled inputs); `p3alpha` / `rp3beta` normalize in float32, so
      float64 input may move by float rounding.
    - The order within a COO row differs from 0.7.0.
    - Outputs past 2^31 results are allowed (CSR then gets int64 `indptr` and
      `indices`); inputs stay int32 (up to 2^31 - 1 entries each).
    - Duplicate `target_rows` are computed once.
    - The normalizations return a `csr_array` also for `spmatrix` input (use
      `@` for products), with int32 indices (float32/float64 values kept), and
      raise a `ValueError` past 2^31 - 1 entries, as the similarities.
    - New: `similaripy.binarize(X)`, the `binary=True` of the similarities as a
      separate step.
    - New: `similaripy.serving.ItemKNN`, an online engine over a similarity
      matrix (save / load with mmap, include / exclude rules): see
      [Serving](serving.md).
    - `num_threads=0` (the default) runs on one shared pool: concurrent calls
      queue on it instead of oversubscribing the cores as OpenMP did.
      `num_threads=n` uses a pool cached per calling thread, so concurrent
      callers don't wait for each other.
    - Free-threaded Python (3.14t, 3.15t) is supported.
    - Wheels reach older macOS than 0.7.0 (Apple Silicon 12+, Intel 10.15+;
      0.7.0 needed macOS 15).
