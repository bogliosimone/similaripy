# 🔁 Migration from 0.7.0

SimilariPy 1.0 replaces the Cython + OpenMP core with a Rust one. The functions
and their parameters stay the same; most code runs unchanged and gets the same
top-k, up to float rounding. This page lists every change you can see, with what
to do about it.

## Install and build

- Wheels cover the same platforms, on older macOS versions too (Apple Silicon
  12+, Intel 10.15+; 0.7.0 needed macOS 15); building from source now needs a Rust
  toolchain instead of a C compiler with OpenMP (no `brew install gcc`). See
  [Build from source](installation.md#build-from-source).
- Minimum versions: numpy `>= 1.23.5` (was 1.22.4), scipy `>= 1.15` (was 1.12).
  Python stays `>= 3.11`.
- The `dev`, `bench` and `notebook` extras are gone (`pip install
  similaripy[notebook]` installs no extra packages): they are development
  dependency groups of the repository now.

## API

- Every parameter after `matrix2` is keyword-only: `sim.cosine(m, None, 50)`
  raises `TypeError`; write `sim.cosine(m, k=50)`.
- `verbose` defaults to `False`: pass `verbose=True` for the progress bar.
- `format_output` defaults to `'csr'` (0.7.0: `'coo'`): pass
  `format_output='coo'` to keep the old output format.
- `block_size` defaults to `'auto'`; `0` (0.7.0's auto) raises `ValueError`.
- `normalize`, `tfidf`, `bm25`, `bm25plus` no longer take `inplace` (a
  `TypeError`): use the return value, `X = bm25(X)`. They always return a new
  `csr_array` (also for `spmatrix` input: use `@` for products) with int32
  indices (float32/float64 values kept), and raise a `ValueError` past
  2^31 - 1 entries, as the similarities.
- New: `similaripy.binarize(X)`, the `binary=True` of the similarities as a
  separate step.
- Parameters are validated, with a `ValueError` instead of a silent result:
  - negative weights (e.g. `l1`, `c1`, `beta`), shrinks or `threshold`,
    `alpha <= 0`, custom
    popularity arrays that are not finite and `>= 0` (`> 0` on rows/columns
    with entries);
  - out-of-range ids in `target_rows`, and in `target_cols` / `filter_cols`
    lists (0.7.0 ignored them);
  - an empty `target_cols` (list, array or sparse matrix with no entries),
    which used to score every column: pass `None` for all columns. An empty
    `filter_cols` is still accepted (it excludes nothing);
  - a shrink that would do nothing: `additive` without a cosine term,
    `stabilized` on `dot_product` / `p3alpha` (it only rescaled the scores);
    see [Shrinkage Types](similarity.md#shrinkage-types);
  - `tversky` / `s_plus` (`l1 != 0`, not binary) `alpha`/`beta` pairs whose
    denominator can be `<= 0`: when `alpha + beta > 1` on weighted input (e.g.
    `(3, 0.05)`, for which 0.7.0 silently dropped pairs), and when
    `alpha + beta < 1` on signed input. No default is affected, nor binary or
    0/1 data;
  - NaN or inf values in the normalizations (they used to spread NaN through
    the row's or column's norm).
  - Norms out of float32 range in `cosine` and `asymmetric_cosine` without a
    shrink (e.g. data scaled by `1e-25` or `1e19`), in
    `rp3beta` when its `pop^beta` leaves float32 range, and in `s_plus` with a
    single cosine or depop term, raise instead of returning empty rows: rescale
    the input. With a shrink, such inputs still return empty rows.

## Results

- **Float rounding:** scores may differ from 0.7.0 in the last digits. For
  `cosine`, `asymmetric_cosine` and `rp3beta` (computed as a dot product of
  pre-scaled inputs) up to about `1e-4` relative on the longest rows, also on
  integer data. Values stay float32: on the 500 longest MovieLens 32M rows,
  99.996% of the top-100 slots equal an exact float64 computation.
- **Ties** are deterministic: on equal scores the more popular column wins,
  whatever `block_size` or `num_threads`. 0.7.0 broke ties by heap order, so
  which of two tied items makes the top-k can differ. On binary or integer
  data, exact ties in `cosine`, `asymmetric_cosine` and `rp3beta` can now
  differ in the last bit.
- **Output layout:** rows are sorted by column (canonical CSR). COO output has
  no padding: its `nnz` is the number of results, and a score of exactly `0` is
  dropped in both formats (0.7.0 kept it in COO). The order within a COO row
  differs from 0.7.0.
- **Outputs past 2^31 results** are allowed (CSR then gets int64 `indptr` and
  `indices`); inputs stay int32 (up to 2^31 - 1 entries each).
- **Duplicate `target_rows`** are computed once.

Bug fixes that change results:

- CSR inputs with duplicate entries are summed (scipy's canonical form) in
  every function: 0.7.0 computed wrong norms and binary counts for them.
- Unsorted CSR rows with column blocking gave wrong results in 0.7.0; inputs
  are always sorted now.
- `shrink_type='bayesian'` on `dot_product` and `p3alpha` returned no results;
  it now applies the bayesian factor.
- `binary=True` in `p3alpha` / `rp3beta` returned plain binary `dot_product`
  scores (the walk's normalization was erased); the input is now binarized
  first, then normalized. RP3β's popularity is then the item degree.
- `s_plus(alpha=...)` was ignored when `l1 = l2 = l3 = 0`; it now always
  applies.
- `tfidf` / `bm25` / `bm25plus` document counts no longer stop growing at
  2^24 documents (a float32 counter).

## Signed input

0.7.0 assumed non-negative values without checking. 1.0 accepts negative
values wherever the result stays meaningful and raises a `ValueError` where it
would be wrong; the rules are in [Signed Input](guide.md#signed-input). The
changes from 0.7.0 on signed data:

- Only positive scores are results: negative scores and scores that cancel to
  exactly `0` are dropped. Pairs that should cancel can leave tiny scores
  (`~1e-8`) in `cosine`, `asymmetric_cosine` (no shrink or additive) and
  `rp3beta` (no shrink), even on integer data: a small `threshold` drops them.
- Raises: bayesian shrink, `alpha != 1` in `s_plus`, and `tfidf`, `bm25`,
  `bm25plus` on negative values.
- `normalize(norm='max')` divides by the largest `|x|`: mixed-sign and
  all-negative rows change (non-negative rows don't).
- Popularity `'sum'` in `s_plus` and the `rp3beta` popularity are sums of `|x|`;
  P3α / RP3β raise signed weights to `alpha` keeping their sign.
- `p3alpha` / `rp3beta` normalize in float32: float64 input may move by float
  rounding.
- A `filter_cols` / `target_cols` matrix with duplicate entries that cancel
  (e.g. `+1` and `-1`) no longer selects that column.
- `binary=True` still treats every non-zero (negative too) as `1`.

## Threads

- `RAYON_NUM_THREADS` replaces `OMP_NUM_THREADS` to cap the default pool.
- `num_threads=0` (the default) runs on one shared pool: concurrent calls
  queue on it instead of oversubscribing the cores as OpenMP did (same total
  time, the first call finishes earlier).
- `num_threads=n` uses a pool cached per calling thread, so concurrent callers
  don't wait for each other.
- Free-threaded Python (3.14t, 3.15t) is supported.
- As with OpenMP (libgomp), the thread pool does not survive `fork()`: in a
  forked child process, call SimilariPy only if the parent never did, or use
  the `spawn` / `forkserver` start methods of `multiprocessing`.
