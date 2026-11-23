"""
s_plus on the Rust core: argument checks, one `_core.similarity` call
(canonical copies + the S-Plus pipeline), output wrap.
"""

import time
from typing import Any, Literal, NamedTuple

import numpy as np
import scipy.sparse as sp

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)

# A caller's sparse input before `_canonical_csr`: any scipy sparse array or
# matrix, checked at runtime (`sp.issparse`). `Any` for type checkers:
# scipy-stubs gives sparray/spmatrix no attributes (.shape, .nnz, ...); after
# the canonical copy the prep works on `Csr` arrays.
SparseInput = Any

# Accumulator block in float32 entries (~1 MB), defined in the Rust core.
DEFAULT_BLOCK_SIZE: int = _core.DEFAULT_BLOCK_SIZE

PROGRESS_BAR_REFRESH_RATE = 3  # Hz
PROGRESS_BAR_WIDTH = 25

# Wall time (s) of the last call's kernel (scoring, top-K, per-row un-permute
# + sort, compaction) and output build (matrix wrap). Read by the benchmark
# harness (benchmarks/workloads.py), which takes prep as total - kernel -
# output, so prep includes the wrappers' normalization. Racy under concurrent
# calls: read it only after serial calls.
_last_timings = {"kernel": 0.0, "output": 0.0}

# Kernel column order: 'popularity' (default, see `_s_plus`) or 'original'
# (no renumbering: ties go to the lower original column). Private switch for
# benchmarks and tests, not an API; don't flip it under concurrent calls.
_COLUMN_ORDER = "popularity"

# Largest nnz with an int32 CSR indptr; past it indptr is int64 (and a scipy
# matrix's indices too: scipy keeps one index dtype). A module constant so
# tests can lower it.
_MAX_INT32_NNZ = np.iinfo(np.int32).max


def _indptr_dtype(nnz: int) -> type[np.signedinteger]:
    """int32 while `nnz` fits (as scipy), else int64."""
    return np.int32 if nnz <= _MAX_INT32_NNZ else np.int64


class Sparse(NamedTuple):
    """A caller's matrix as scipy stores it, for `_core.canonical` /
    `_core.similarity` (read by attribute): `major` = indptr (COO: rows),
    `minor` = indices (COO: cols), both int32 or both int64; float32 or
    float64 data; `canonical` = scipy's `has_canonical_format` (a hint)."""

    format: str
    shape: tuple[int, int]
    major: np.ndarray
    minor: np.ndarray
    data: np.ndarray
    canonical: bool


class Csr(NamedTuple):
    """A canonical CSR matrix's arrays (float32 or float64 data, int32
    indices, int64 indptr: the core's offsets; outputs follow `_indptr_dtype`)."""

    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    n_cols: int

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.indptr) - 1, self.n_cols


class Params(NamedTuple):
    """The scoring weights (`s_plus` arguments of the same names), read by
    attribute by `_core.similarity`."""

    a1: float
    l1: float
    l2: float
    l3: float
    t1: float
    t2: float
    stabilized_shrink: float
    bayesian_shrink: float
    threshold: float


def _check_args(
    matrix1: SparseInput,
    matrix2: SparseInput,
    weight_depop_matrix1: str | np.ndarray,
    weight_depop_matrix2: str | np.ndarray,
    k: int,
    filter_cols: list | np.ndarray | SparseInput | None,
    target_cols: list | np.ndarray | SparseInput | None,
    format_output: str,
    weights: dict,
    a1: float,
    num_threads: int,
    block_size: int | Literal["auto"] | None,
) -> None:
    """
    Check the `s_plus` arguments that don't need the values, before the copy
    (fail fast): types, shapes and scalars. The rules on the values (id
    ranges, depop weights, finite data, signed input, tversky) are the core's.

    `weights`: name -> value of every scalar that must be finite and >= 0.

    Raises:
        TypeError: If matrix1 or matrix2 are not sparse matrices.
        ValueError: If any parameter has invalid value or incompatible dimensions.
    """
    # Check matrix types
    if not sp.issparse(matrix1):
        raise TypeError("matrix1 must be a sparse matrix")
    if not sp.issparse(matrix2):
        raise TypeError("matrix2 must be a sparse matrix")

    # Check matrix dimensions
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(
            f"Incompatible matrix shapes: matrix1.shape[1]={matrix1.shape[1]} "
            f"must equal matrix2.shape[0]={matrix2.shape[0]}"
        )

    if not _is_int(k) or k < 1:
        raise ValueError(f"k must be an integer >= 1, got {k!r}")

    _check_depop("pop1", weight_depop_matrix1, matrix1.shape[0])
    _check_depop("pop2", weight_depop_matrix2, matrix2.shape[1])

    # Column selectors: a list/array is one set of column ids for every row, a
    # sparse matrix one row of columns per output row (the core's `Csr` has no
    # width: the shape is checked here).
    expected_shape = (matrix1.shape[0], matrix2.shape[1])
    for name, cols in (("filter_cols", filter_cols), ("target_cols", target_cols)):
        if cols is None:
            continue
        if isinstance(cols, (list, np.ndarray)):
            _ids(name, cols)
        elif not sp.issparse(cols):
            raise TypeError(f"{name} must be a sparse matrix, list, numpy array, or None")
        elif cols.nnz != 0 and cols.shape != expected_shape:
            raise ValueError(f"{name} shape {cols.shape} does not match expected shape {expected_shape}")
    # An empty target_cols would score nothing: None is "all columns". An empty
    # filter_cols excludes nothing, the same as None.
    if (
        target_cols is not None
        and (np.size(target_cols) if isinstance(target_cols, (list, np.ndarray)) else target_cols.nnz) == 0
    ):
        raise ValueError("target_cols is empty: pass None to score all columns")

    # Check format_output
    if format_output not in ("coo", "csr"):
        raise ValueError(f"format_output must be 'coo' or 'csr', got '{format_output}'")

    # Every weight >= 0: the norms (sums of x^2), popularities (sums of |x|) and
    # cosine and depop terms are then >= 0 on any finite input, and tversky's
    # denominator is kept > 0 by its rule in the core (with the rules for
    # signed input). Negative weights would flip signs or make NaN (a negative
    # base to a fractional power); alpha <= 0 would flatten or invert the ranking.
    for name, value in weights.items():
        _check_scalar(name, value)
    _check_scalar("a1 (alpha)", a1, strict=True)

    if not (
        block_size is None
        or (isinstance(block_size, str) and block_size == "auto")
        or (_is_int(block_size) and block_size > 0)
    ):
        raise ValueError(f"block_size must be 'auto', None (no blocking) or an integer > 0, got {block_size!r}")
    if not (_is_int(num_threads) and num_threads >= 0):
        raise ValueError(f"num_threads must be an integer >= 0, got {num_threads!r}")


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _check_scalar(name: str, value, strict: bool = False) -> None:
    """Finite and >= 0 (> 0 if `strict`)."""
    if not (np.isfinite(value) and (value > 0 if strict else value >= 0)):
        raise ValueError(f"{name} must be finite and {'>' if strict else '>='} 0, got {value}")


def _ids(name: str, ids) -> np.ndarray:
    """`ids` as a flat int64 array (the core checks the range)."""
    ids = np.asarray(ids)
    if ids.size != 0 and not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f"{name} must hold integer ids, got dtype {ids.dtype}")
    return np.ascontiguousarray(ids.ravel(), dtype=np.int64)


def _check_depop(name: str, weights, n: int) -> None:
    """'none', 'sum' or n weights (the core checks the values: finite, >= 0,
    > 0 where used)."""
    if isinstance(weights, str):
        if weights not in ("none", "sum"):
            raise ValueError(f'{name} must be "none", "sum" or an array of length {n}, got {weights!r}')
        return
    w = np.asarray(weights)
    if w.shape != (n,):
        raise ValueError(f'{name} must be "none", "sum" or an array of length {n}, got shape {w.shape}')


def _sparse(matrix: SparseInput, dtype=np.float32) -> Sparse:
    """
    `matrix`'s arrays as the core's one door in (`prep::canonical`) reads them:
    CSR / CSC / COO as they are, other formats via `tocsr()`, other value
    types (int, bool, float16, ...) one `astype(dtype)`.
    """
    if matrix.format not in ("csr", "csc", "coo"):
        matrix = matrix.tocsr()
    if matrix.format == "coo":
        major, minor, data = matrix.row, matrix.col, matrix.data
    else:
        nnz = matrix.indptr[-1]  # scipy allows longer arrays past the last row
        major, minor, data = matrix.indptr, matrix.indices[:nnz], matrix.data[:nnz]
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(dtype)
    if major.dtype != minor.dtype or major.dtype not in (np.int32, np.int64):  # scipy never does this
        major, minor = major.astype(np.int64), minor.astype(np.int64)
    return Sparse(
        matrix.format,
        matrix.shape,
        np.ascontiguousarray(major),  # COO coords may be strided views
        np.ascontiguousarray(minor),
        np.ascontiguousarray(data),
        bool(matrix.has_canonical_format),
    )


def _canonical_csr(matrix: SparseInput, dtype, num_threads: int) -> Csr:
    """
    The call's own canonical copy of `matrix`: CSR arrays, `dtype` values (float32
    or float64), int32 indices, rows sorted, duplicates summed, no explicit zeros.
    Arrays, not a scipy matrix: the indptr is int64 (as the core reads it), and
    scipy would copy the indices to int64 too.
    Always a copy: the caller's matrix never changes, and every later step may
    change the arrays in place. Used by the normalizations (the input's float
    type) and serving; the similarities copy inside `_core.similarity`.

    Explicit zeros are wrong in binary mode (they would become 1.0), and
    unsummed duplicates give wrong norms and binary counts. The Rust core
    (`_core.canonical`) reads CSR / CSC / COO arrays as they are and copies them
    once: values cast before duplicates are summed (zeros are dropped after the
    cast, so a float64 value below float32's range goes too when `dtype` is
    float32), CSC and COO through one counting sort by row (no scipy `tocsr`
    temp). Canonical input (scipy's `has_canonical_format`, e.g. our own output
    fed back) only drops zeros.
    """
    s = _sparse(matrix, dtype)
    indptr, indices, data = _core.canonical(**s._asdict(), dtype=np.dtype(dtype).name, num_threads=num_threads)
    return Csr(data, indices, indptr, matrix.shape[1])


def _selector(name: str, cols) -> np.ndarray | Sparse | None:
    """A column selector for the core: int64 ids, a matrix's arrays, or None
    (no selector, or an empty matrix: an empty filter excludes nothing)."""
    if cols is None or (sp.issparse(cols) and cols.nnz == 0):
        return None
    return _sparse(cols) if sp.issparse(cols) else _ids(name, cols)


# ---- Entry point ----


def s_plus(
    matrix1: SparseInput,
    matrix2: SparseInput | None = None,
    weight_depop_matrix1: str | np.ndarray = "none",
    weight_depop_matrix2: str | np.ndarray = "none",
    p1: float = 0,
    p2: float = 0,
    a1: float = 1,
    l1: float = 0,
    l2: float = 0,
    l3: float = 0,
    t1: float = 1,
    t2: float = 1,
    c1: float = 0.5,
    c2: float = 0.5,
    k: int = 100,
    stabilized_shrink: float = 0,
    bayesian_shrink: float = 0,
    additive_shrink: float = 0,
    threshold: float = 0,
    binary: bool = False,
    target_rows: list | np.ndarray | None = None,
    filter_cols: list | np.ndarray | SparseInput | None = None,
    target_cols: list | np.ndarray | SparseInput | None = None,
    verbose: bool = False,
    format_output: str = "csr",
    num_threads: int = 0,
    block_size: int | Literal["auto"] | None = "auto",
    walk: tuple[float, bool] | None = None,
) -> sp.csr_array | sp.coo_array:
    """
    Compute top-K similarity between rows of two sparse matrices using the S_Plus algorithm.

    The S_Plus algorithm combines multiple similarity metrics (Tversky, Cosine, Depopularization)
    with configurable weights and supports various normalization strategies.

    Args:
        matrix1: First sparse matrix (typically item-user interactions); values finite
            in float32, negative values allowed (signed input: negative and 0.0
            scores are never results; docs/guide.md, "Signed Input").
        matrix2: Optional second matrix. If None, uses matrix1.T (self-similarity).
        weight_depop_matrix1: Depopularization weights for matrix1: 'none', 'sum', or custom array.
        weight_depop_matrix2: Depopularization weights for matrix2: 'none', 'sum', or custom array.
        p1: Power for depopularization weights on matrix1.
        p2: Power for depopularization weights on matrix2.
        a1: Power applied to dot product values (with or without normalization); must be > 0,
            and 1 on signed input.
        l1: Weight for Tversky similarity term.
        l2: Weight for Cosine similarity term.
        l3: Weight for Depopularization term.
        t1: Tversky parameter for matrix1.
        t2: Tversky parameter for matrix2.
        c1: Cosine power exponent for matrix1.
        c2: Cosine power exponent for matrix2.
        k: Number of top similar items to keep per row.
        stabilized_shrink: Stabilized shrinkage parameter.
        bayesian_shrink: Bayesian shrinkage parameter; 0 on signed input.
        additive_shrink: Additive shrinkage for cosine normalization.
        threshold: Minimum similarity threshold.
        binary: If True, treat all non-zero values as 1.0 (set theory).
        target_rows: Specific rows to compute (duplicates ignored). If None, compute all rows.
        filter_cols: Columns to exclude from results (e.g., already seen items).
        target_cols: Columns to include in results (only compute these); None = all,
            an empty one raises ValueError.
        verbose: Show progress bar during computation.
        format_output: Output matrix format: 'csr' or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation. Controls the trade-off
            between cache efficiency and multi-pass overhead.
            'auto' = the library's choice, from the output width, the L2 cache and the
            threads; may change in future versions (results are the same at any size).
            None = disabled (no blocking).
            int > 0 = explicit size in number of float32 entries.
        walk: Private (p3alpha, rp3beta): (alpha, popularity). Both inputs'
            rows l1-normalized (after `binary`), then raised to alpha keeping
            signs; `popularity` replaces m2's depop weights by its column
            sums of |x| before the walk.

    Returns:
        A sparse matrix of shape (n_rows, n_cols) in the specified format,
        containing the top-k similarity scores: at most k entries per row, rows
        sorted by column, canonical (`has_canonical_format`).

    Raises:
        ValueError: invalid shapes or parameters (see `_check_args`),
            a block_size other than 'auto', None or an int > 0, input values
            not finite in float32, a1 != 1 or bayesian_shrink on signed input
            (values < 0, not binary), or l1 != 0 with t1, t2 whose tversky
            term can be <= 0 on weighted input, values out of float32 range
            for a score computed from pre-scaled inputs (the core's
            `similarity` rules), ids out of range.
    """
    if matrix2 is None:
        matrix2 = matrix1.T

    _check_args(
        matrix1=matrix1,
        matrix2=matrix2,
        weight_depop_matrix1=weight_depop_matrix1,
        weight_depop_matrix2=weight_depop_matrix2,
        k=k,
        filter_cols=filter_cols,
        target_cols=target_cols,
        format_output=format_output,
        weights={
            "l1": l1,
            "l2": l2,
            "l3": l3,
            "t1 (tversky alpha)": t1,
            "t2 (tversky beta)": t2,
            "c1": c1,
            "c2": c2,
            "p1 (beta1)": p1,
            "p2 (beta2, rp3beta beta)": p2,
            "threshold": threshold,
            "stabilized_shrink": stabilized_shrink,
            "bayesian_shrink": bayesian_shrink,
            "additive_shrink": additive_shrink,
        },
        a1=a1,
        num_threads=num_threads,
        block_size=block_size,
    )

    if target_rows is not None:  # any order, duplicates ignored (the core sorts)
        target_rows = _ids("target_rows", target_rows)

    bar = None
    if verbose:
        n_targets = matrix1.shape[0] if target_rows is None else len(np.unique(target_rows))
        bar = _core.ProgressBar(n_targets, PROGRESS_BAR_REFRESH_RATE, PROGRESS_BAR_WIDTH)
        bar.set_description("Preprocessing")
    try:
        indptr, cols, vals, kernel_seconds = _core.similarity(
            m1=_sparse(matrix1),
            m2=_sparse(matrix2),
            k=k,
            params=Params(a1, l1, l2, l3, t1, t2, stabilized_shrink, bayesian_shrink, threshold),
            c1=c1,
            c2=c2,
            p1=p1,
            p2=p2,
            additive_shrink=additive_shrink,
            pop1=_pop(weight_depop_matrix1),
            pop2=_pop(weight_depop_matrix2),
            binary=binary,
            walk=walk,
            target_rows=target_rows,
            filter_cols=_selector("filter_cols", filter_cols),
            target_cols=_selector("target_cols", target_cols),
            block_size=block_size,
            column_order=_COLUMN_ORDER,
            num_threads=num_threads,
            progress=bar,
        )
        if bar is not None:
            bar.set_description(f"Building {format_output} matrix")
        output_start = time.perf_counter()
        out = _build_output(indptr, cols, vals, (matrix1.shape[0], matrix2.shape[1]), format_output)
        _last_timings.update(kernel=kernel_seconds, output=time.perf_counter() - output_start)
    except BaseException:
        if bar is not None:
            bar.close("Failed")
        raise
    if bar is not None:
        bar.close("Done")
    return out


def _pop(weights):
    """'none' / 'sum' as they are, custom weights as float32."""
    return weights if isinstance(weights, str) else np.ascontiguousarray(weights, dtype=np.float32)


def _build_output(indptr, cols, vals, shape, format_output):
    """
    The core's CSR (int64 indptr, exact rows sorted by column, canonical: no
    duplicates, no zeros) wrapped without a copy: indptr narrowed to int32
    while it fits; COO rows from the indptr.
    """
    if format_output == "coo":
        rows = np.repeat(np.arange(shape[0], dtype=np.int32), np.diff(indptr))
        out = sp.coo_array((vals, (rows, cols)), shape=shape)
    else:
        indptr = indptr.astype(_indptr_dtype(indptr[-1]), copy=False)
        if indptr.dtype == np.int64:  # scipy wants one index dtype: it would upcast cols anyway
            cols = cols.astype(np.int64)
        out = sp.csr_array((vals, cols, indptr), shape=shape)
    out.has_canonical_format = True
    return out
