"""
    s_plus on the Rust kernel: numpy prep, kernel call, output build (D19).

    Same results as 0.7.0's Cython `s_plus`, minus its bugs:
    never mutates the caller's matrices, always sorts m2 when blocking,
    accepts int64-indexed inputs, closes the progress bar on errors.
"""

import time
from typing import Union, Optional, List

import numpy as np
import scipy.sparse as sp

from . import _core

# Accumulator block in float32 entries (~1 MB), as 0.7.0 (`s_plus.h`).
DEFAULT_BLOCK_SIZE = 262144

PROGRESS_BAR_REFRESH_RATE = 3  # Hz
PROGRESS_BAR_WIDTH = 25

# Column selector modes (the kernel's `filter_mode` / `target_mode`)
MODE_NONE = 0    # use all columns
MODE_ARRAY = 1   # list/array, pre-filtered here: the kernel treats it as NONE
MODE_MATRIX = 2  # sparse matrix, per-row check in the kernel

# Wall time (s) of the last call's kernel (scoring, top-K, per-row un-permute
# + sort, compaction) and output build (matrix wrap); the harness takes prep as total - kernel - output, so it includes
# the wrappers' normalization. Racy under concurrent calls: read it only after
# serial calls.
_last_timings = {'kernel': 0.0, 'output': 0.0}

# Kernel column order: 'popularity' (default, see `_s_plus`) or 'original'
# (no renumbering: ties go to the lower original column). Private switch for
# benchmarks and tests, not an API; don't flip it under concurrent calls.
_COLUMN_ORDER = 'popularity'

# Largest output nnz with an int32 CSR indptr; past it indptr and indices
# are int64 (D42). A module constant so tests can lower it.
_MAX_INT32_NNZ = np.iinfo(np.int32).max

_EMPTY_F32 = np.empty(0, dtype=np.float32)
_EMPTY_I32 = np.empty(0, dtype=np.int32)


def get_num_threads() -> int:
    """Threads used with `num_threads=0` (rayon global pool, `RAYON_NUM_THREADS`)."""
    return _core.get_num_threads()


def _validate_inputs(
    matrix1: sp.sparray,
    matrix2: sp.sparray,
    weight_depop_matrix1: Union[str, np.ndarray],
    weight_depop_matrix2: Union[str, np.ndarray],
    k: int,
    filter_cols: Optional[Union[List, np.ndarray, sp.sparray]],
    target_cols: Optional[Union[List, np.ndarray, sp.sparray]],
    format_output: str,
    weights: dict,
    a1: float,
    num_threads: int,
    block_size: Optional[int],
) -> None:
    """
    Validate the `s_plus` arguments that don't need the data; the rules on the
    canonical values (finite in float32, signed input, used depop weights) are
    checked in `_s_plus`.

    `weights`: name -> value of every scalar that must be finite and >= 0.

    Raises:
        TypeError: If matrix1 or matrix2 are not sparse matrices.
        ValueError: If any parameter has invalid value or incompatible dimensions.
    """
    # Check matrix types
    if not sp.issparse(matrix1):
        raise TypeError('matrix1 must be a sparse matrix')
    if not sp.issparse(matrix2):
        raise TypeError('matrix2 must be a sparse matrix')

    # Check matrix dimensions
    if matrix1.shape[1] != matrix2.shape[0]:
        raise ValueError(
            f'Incompatible matrix shapes: matrix1.shape[1]={matrix1.shape[1]} '
            f'must equal matrix2.shape[0]={matrix2.shape[0]}'
        )

    if not _is_int(k) or k < 1:
        raise ValueError(f'k must be an integer >= 1, got {k!r}')

    _check_depop('weight_depop_matrix1 (pop1)', weight_depop_matrix1, matrix1.shape[0])
    _check_depop('weight_depop_matrix2 (pop2)', weight_depop_matrix2, matrix2.shape[1])

    # Column selectors: a list/array is one set of column ids for every row, a
    # sparse matrix one row of columns per output row.
    expected_shape = (matrix1.shape[0], matrix2.shape[1])
    for name, cols in (('filter_cols', filter_cols), ('target_cols', target_cols)):
        if cols is None:
            continue
        if isinstance(cols, (list, np.ndarray)):
            _ids(name, cols, matrix2.shape[1])
        elif not sp.issparse(cols):
            raise TypeError(f'{name} must be a sparse matrix, list, numpy array, or None')
        elif cols.data.shape[0] != 0 and cols.shape != expected_shape:
            raise ValueError(f'{name} shape {cols.shape} does not match expected shape {expected_shape}')

    # Check format_output
    if format_output not in ('coo', 'csr'):
        raise ValueError(f"format_output must be 'coo' or 'csr', got '{format_output}'")

    # Every weight >= 0: the norms (sums of x^2), popularities (sums of |x|) and
    # cosine and depop terms are then >= 0 on any finite input, and tversky's
    # denominator is kept > 0 by its rule in `_s_plus` (`_check_signed_rules`,
    # with the rules for signed input, D45). Negative weights would flip signs or
    # make NaN (a negative base to a fractional power); alpha <= 0 would flatten
    # or invert the ranking.
    for name, value in weights.items():
        _check_scalar(name, value)
    _check_scalar('a1 (alpha)', a1, strict=True)

    if block_size is not None and not (_is_int(block_size) and block_size >= 0):
        raise ValueError(f'block_size must be None (no blocking), 0 (auto) or > 0, got {block_size!r}')
    if not (_is_int(num_threads) and num_threads >= 0):
        raise ValueError(f'num_threads must be an integer >= 0, got {num_threads!r}')

    # Input indices and indptrs are int32: fail loudly instead of wrapping. The
    # output may pass 2^31 results (D42, see `_MAX_INT32_NNZ`).
    int32_max = np.iinfo(np.int32).max
    if matrix1.nnz > int32_max or matrix2.nnz > int32_max:
        raise ValueError(f"input matrices must have at most {int32_max} non-zeros")


def _is_int(x) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _check_scalar(name: str, value, strict: bool = False) -> None:
    """Finite and >= 0 (> 0 if `strict`)."""
    if not (np.isfinite(value) and (value > 0 if strict else value >= 0)):
        raise ValueError(f'{name} must be finite and {">" if strict else ">="} 0, got {value}')


def _ids(name: str, ids, n: int) -> np.ndarray:
    """`ids` as an integer array, every id in [0, n)."""
    ids = np.asarray(ids)
    if ids.size == 0:
        return ids.astype(np.int64)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f'{name} must hold integer ids, got dtype {ids.dtype}')
    if ids.min() < 0 or ids.max() >= n:
        raise ValueError(f'{name} ids must be in [0, {n}), got min {ids.min()} and max {ids.max()}')
    return ids


def _check_depop(name: str, weights, n: int) -> None:
    """'none', 'sum' or n finite weights >= 0 (a negative one would flip signs); > 0 where
    it is used is checked in `_s_plus` (`_check_used_weights`)."""
    if isinstance(weights, str):
        if weights not in ('none', 'sum'):
            raise ValueError(f'{name} must be "none", "sum" or an array of length {n}, got {weights!r}')
        return
    w = np.asarray(weights)
    if w.shape != (n,):
        raise ValueError(f'{name} must be "none", "sum" or an array of length {n}, got shape {w.shape}')
    if not np.all((w >= 0) & (w < np.inf)):
        raise ValueError(f'{name} weights must be finite and >= 0')


def _check_used_weights(name: str, weights, used: np.ndarray) -> None:
    """
    Custom weights > 0 on the rows/columns with entries: only those reach a
    denominator, where a 0 weight would score 0.0. Empty ones (e.g. items with
    no interactions, pop 0 when computed as counts) are never read.
    """
    if not isinstance(weights, str) and not np.all(np.asarray(weights)[used] > 0):
        raise ValueError(f'{name} weights must be > 0 on every row/column with entries')


# ---- Prep helpers (numpy, D19) ----

def _check_signed_rules(signed: bool, binary: bool, a1, l1, t1, t2, bayesian_shrink,
                        m1_data, m2_data) -> None:
    """
    Parameters that would be wrong on signed input (D45), plus the tversky
    denominator rule, which also binds on non-negative weighted input.
    """
    if signed and bayesian_shrink != 0:
        raise ValueError("bayesian shrink (shrink_type='bayesian', s_plus bayesian_shrink) is not "
                         'supported on signed input (values < 0): xy / (xy + shrink) flips the sign '
                         'of negative scores')
    if signed and a1 != 1:
        raise ValueError(f'a1 (alpha) must be 1 on signed input (values < 0), got {a1}: '
                         'a power of a negative score is NaN or turns it positive')
    # den = t1 |x|^2 + t2 |y|^2 + c xy (c = 1 - t1 - t2), |xy| <= |x| |y|: > 0 on
    # every pair iff c = 0 or 4 t1 t2 > c^2. The worst pair has xy = |x||y| for
    # c < 0 (any weighted input) and xy = -|x||y| for c > 0 (signed input only).
    # Binary, or every value in both matrices the same (0/1 data passed without
    # binary=True): den >= v^2 |X & Y| > 0 always. Only the tversky term is checked:
    # l2/l3/stabilized shrink terms may keep den > 0, but not on every pair.
    c = 1 - t1 - t2
    if (l1 != 0 and not binary and c != 0 and (c < 0 or signed) and not 4 * t1 * t2 > c * c
            and not _one_value(m1_data, m2_data)):
        raise ValueError(
            f't1, t2 (tversky alpha, beta) = ({t1}, {t2}) give a denominator <= 0 on some pairs'
            f'{" of signed input (values < 0)" if c > 0 else ""}: need t1 + t2 = 1 or '
            f'4 t1 t2 > (1 - t1 - t2)^2')


def _one_value(*arrays) -> bool:
    """Every entry of the (non-empty) arrays equal to the same value."""
    vals = {v for a in arrays if a.size for v in (a.min(), a.max())}
    return len(vals) == 1


def _canonical_csr(matrix: sp.sparray, num_threads: int) -> sp.csr_array:
    """
    CSR with summed duplicates and no explicit zeros, without touching the
    caller's matrix.

    Explicit zeros are wrong in binary mode (they would become 1.0), and
    unsummed duplicates give wrong norms and binary counts (a bug in 0.7.0
    for CSR input). Canonical CSR passes through (copied only when it has
    zeros), and so does a canonical COO (indptr from the rows); other COO and
    non-canonical CSR go through the Rust canonicalization,
    which replaces scipy's single-threaded COO -> CSR (2.2 s on a 93M-entry
    COO item model).
    """
    if matrix.format == 'coo' and matrix.has_canonical_format:
        # Row-major, sorted, summed (e.g. our own COO output, fed back): only
        # the indptr to build, then the canonical CSR path below.
        indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int32)
        np.cumsum(np.bincount(matrix.row, minlength=matrix.shape[0]), out=indptr[1:])
        matrix = sp.csr_array((matrix.data, matrix.col, indptr), shape=matrix.shape)
        matrix.has_canonical_format = True
    if matrix.format == 'coo':
        rows, cols, data = matrix.row, matrix.col, matrix.data
        if np.any(rows[1:] < rows[:-1]):
            order = np.argsort(rows, kind='stable')
            rows, cols, data = rows[order], cols[order], data[order]
        indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=matrix.shape[0]), out=indptr[1:])
    else:
        matrix = matrix.tocsr()
        if matrix.has_canonical_format:
            if (matrix.data == 0).any():
                matrix = matrix.copy()
                matrix.eliminate_zeros()
            return matrix
        indptr, cols, data = matrix.indptr.astype(np.int32), matrix.indices, matrix.data
    # private copies: canonicalized in place
    indices, data = cols.astype(np.int32), data.astype(np.float32)
    indptr = _core.canonicalize_rows(indptr, indices, data, num_threads)
    nnz = indptr[-1]
    out = sp.csr_array((data[:nnz], indices[:nnz], indptr), shape=matrix.shape)
    out.has_canonical_format = True  # sorted, summed: skips scipy's re-checks
    return out


def _row_sums(values: np.ndarray, indptr: np.ndarray, dtype=None) -> np.ndarray:
    """
    Per-row sums of a CSR value array, in the values' dtype unless `dtype` is
    given; same summation order as 0.7.0.

    The trailing 0 lets `reduceat` take start == nnz (trailing empty rows,
    an IndexError in 0.7.0); reduceat returns values[start] for empty rows,
    hence the reset.
    """
    starts = indptr[:-1]
    out = np.add.reduceat(np.append(values, values.dtype.type(0)), starts, dtype=dtype)
    out[starts == indptr[1:]] = 0
    return out


def _col_sums(values: np.ndarray, indices: np.ndarray, n_cols: int) -> np.ndarray:
    """Per-column sums of a CSR value array (bincount in float64, cast once to float32)."""
    return np.bincount(indices, weights=values, minlength=n_cols).astype(np.float32, copy=False)


def _depop_weights(weight_spec, sums, p: float, n: int) -> np.ndarray:
    """Depopularization weights: custom array ** p, 'none' (ones) or row/col sums ** p."""
    if not isinstance(weight_spec, str):
        return np.power(np.asarray(weight_spec, dtype=np.float32), p, dtype=np.float32)
    if weight_spec == 'none':
        return np.ones(n, dtype=np.float32)  # 1 ** p == 1
    return np.power(sums(), p, dtype=np.float32)


def _column_selector(cols, num_threads: int):
    """
    (mode, indptr, indices) for filter_cols/target_cols.

    Matrix mode: the canonical pattern (`_canonical_csr`, never the caller's
    matrix changed; 0.7.0 sorted and eliminated zeros in place), so an entry is
    selected iff its summed value is non-zero, the same rule as the inputs:
    COO and non-canonical CSR agree on cancelling duplicates (D45). A matrix
    whose entries all cancel selects nothing (as explicit zeros only); an
    empty one is no selector.
    """
    if sp.issparse(cols) and cols.data.shape[0] != 0:
        cols = _canonical_csr(cols, num_threads)
        return MODE_MATRIX, cols.indptr.astype(np.int32, copy=False), cols.indices.astype(np.int32, copy=False)
    if isinstance(cols, (list, np.ndarray)) and len(cols) != 0:
        return MODE_ARRAY, _EMPTY_I32, _EMPTY_I32
    return MODE_NONE, _EMPTY_I32, _EMPTY_I32


def _array_keep_mask(filter_cols, target_cols, n_cols: int) -> np.ndarray:
    """
    Mask of the columns kept by array-mode selectors: the target array (or all columns)
    minus the filter array (ids validated in range). Only called when at
    least one selector is an array; a matrix selector here counts as "all".
    """
    if isinstance(target_cols, (list, np.ndarray)) and len(target_cols) != 0:
        mask = np.zeros(n_cols, dtype=bool)
        mask[np.asarray(target_cols)] = True
    else:
        mask = np.ones(n_cols, dtype=bool)
    if isinstance(filter_cols, (list, np.ndarray)) and len(filter_cols) != 0:
        mask[np.asarray(filter_cols)] = False
    return mask


def _keep_columns(data, indices, indptr, keep_mask):
    """Drops the entries whose column is not in `keep_mask`; column ids and order unchanged."""
    # ponytail: slower than 0.7.0's Cython loop (two boolean gathers); a Rust
    # pass is the upgrade, tracked for the non-hot-path polish
    sel = keep_mask[indices]
    new_indptr = np.zeros(len(indptr), dtype=np.int32)
    np.cumsum(_row_sums(sel, indptr, dtype=np.int32), out=new_indptr[1:])
    return data[sel], indices[sel], new_indptr


def _sorted_rows(data, indices, indptr, n_cols, forward, num_threads):
    """
    Rows renumbered by `forward` (empty = keep) and sorted by column id, in Rust,
    in parallel; copies first, so the caller's arrays never change. Without a
    renumbering, sorted rows are returned as they are.
    """
    if len(forward) == 0 and sp.csr_array((data, indices, indptr), shape=(len(indptr) - 1, n_cols),
                                          copy=False).has_sorted_indices:
        return data, indices
    data, indices = data.copy(), indices.copy()
    _core.remap_sort_rows(indptr, indices, data, forward, num_threads)
    return data, indices


# ---- Entry point ----

def s_plus(
    matrix1: sp.sparray,
    matrix2: Optional[sp.sparray] = None,
    weight_depop_matrix1: Union[str, np.ndarray] = 'none',
    weight_depop_matrix2: Union[str, np.ndarray] = 'none',
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
    target_rows: Optional[Union[list, np.ndarray]] = None,
    filter_cols: Optional[Union[list, np.ndarray, sp.sparray]] = None,
    target_cols: Optional[Union[list, np.ndarray, sp.sparray]] = None,
    verbose: bool = True,
    format_output: str = 'csr',
    num_threads: int = 0,
    block_size: Optional[int] = 0,
) -> Union[sp.csr_array, sp.coo_array]:
    """
    Compute top-K similarity between rows of two sparse matrices using the S_Plus algorithm.

    The S_Plus algorithm combines multiple similarity metrics (Tversky, Cosine, Depopularization)
    with configurable weights and supports various normalization strategies.

    Args:
        matrix1: First sparse matrix (typically item-user interactions); values finite
            in float32, negative values allowed (signed input, D45: negative and 0.0
            scores are never results).
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
        target_cols: Columns to include in results (only compute these).
        verbose: Show progress bar during computation.
        format_output: Output matrix format: 'csr' or 'coo'.
        num_threads: Number of threads (0 = rayon global pool, all cores or `RAYON_NUM_THREADS`).
        block_size: Block size for column-blocked accumulation. Controls the trade-off
            between cache efficiency and multi-pass overhead.
            0 = auto (DEFAULT_BLOCK_SIZE entries, ~1 MB).
            None = disabled (no blocking).
            int > 0 = explicit size in number of float32 entries.

    Returns:
        A sparse matrix of shape (n_rows, n_cols) in the specified format,
        containing the top-k similarity scores: at most k entries per row, rows
        sorted by column, canonical (`has_canonical_format`).

    Raises:
        ValueError: invalid shapes or parameters (see `_validate_inputs`),
            a negative block_size, more than 2^31 - 1 non-zeros in an input, input
            values not finite in float32, a1 != 1 or bayesian_shrink on signed input
            (values < 0, not binary), or l1 != 0 with t1, t2 whose tversky
            term can be <= 0 on weighted input (see `_check_signed_rules`).
    """
    if matrix2 is None:
        matrix2 = matrix1.T

    _validate_inputs(
        matrix1=matrix1,
        matrix2=matrix2,
        weight_depop_matrix1=weight_depop_matrix1,
        weight_depop_matrix2=weight_depop_matrix2,
        k=k,
        filter_cols=filter_cols,
        target_cols=target_cols,
        format_output=format_output,
        weights={'l1': l1, 'l2': l2, 'l3': l3, 't1 (tversky alpha)': t1, 't2 (tversky beta)': t2,
                 'c1': c1, 'c2': c2, 'p1 (beta1)': p1, 'p2 (beta2, rp3beta beta)': p2,
                 'threshold': threshold, 'stabilized_shrink': stabilized_shrink,
                 'bayesian_shrink': bayesian_shrink, 'additive_shrink': additive_shrink},
        a1=a1,
        num_threads=num_threads,
        block_size=block_size,
    )

    # Sorted, unique: output rows come out in row order (CSR straight from the
    # counts), and a duplicate target would duplicate its row's entries.
    if target_rows is not None:
        target_rows = np.unique(_ids('target_rows', target_rows, matrix1.shape[0])).astype(np.int32)

    targets = np.arange(matrix1.shape[0], dtype=np.int32) if target_rows is None else target_rows
    n_targets = len(targets)

    bar = None
    if verbose:
        bar = _core.ProgressBar(n_targets, PROGRESS_BAR_REFRESH_RATE, PROGRESS_BAR_WIDTH)
        bar.set_description('Preprocessing')
    try:
        res = _s_plus(
            bar, matrix1, matrix2, targets, k,
            weight_depop_matrix1, weight_depop_matrix2, p1, p2, a1, l1, l2, l3, t1, t2, c1, c2,
            stabilized_shrink, bayesian_shrink, additive_shrink, threshold, binary,
            filter_cols, target_cols, format_output, num_threads, block_size,
        )
    except BaseException:
        if bar is not None:
            bar.close('Failed')
        raise
    if bar is not None:
        bar.close('Done')
    return res


def _s_plus(
    bar, matrix1, matrix2, targets, k,
    weight_depop_matrix1, weight_depop_matrix2, p1, p2, a1, l1, l2, l3, t1, t2, c1, c2,
    stabilized_shrink, bayesian_shrink, additive_shrink, threshold, binary,
    filter_cols, target_cols, format_output, num_threads, block_size,
):
    m1 = _canonical_csr(matrix1, num_threads)
    m2 = _canonical_csr(matrix2, num_threads)
    n_output_rows, n_output_cols = m1.shape[0], m2.shape[1]
    m1_data, m2_data = _kernel_values(m1, m2, binary, a1, l1, t1, t2, bayesian_shrink)
    m1_arrays = (m1_data, m1.indices.astype(np.int32, copy=False), m1.indptr.astype(np.int32, copy=False))
    m2_arrays = (m2_data, m2.indices.astype(np.int32, copy=False), m2.indptr.astype(np.int32, copy=False))

    # Block size: 0 = auto, None = disabled (0 for the kernel), int > 0 = explicit
    if block_size is None:
        block_size = 0
    elif block_size == 0:
        block_size = DEFAULT_BLOCK_SIZE
    use_blocking = block_size > 0 and n_output_cols > block_size

    norms = _norms(m1_arrays, m2_arrays, n_output_cols, l1, l2, l3, c1, c2, additive_shrink,
                   weight_depop_matrix1, weight_depop_matrix2, p1, p2)
    filt, targ, m2_arrays = _selectors(filter_cols, target_cols, m2_arrays, n_output_cols, num_threads)
    m2_arrays, norms, filt, targ, back_perm = _popularity_layout(
        m2_arrays, norms, filt, targ, n_output_cols, use_blocking, num_threads)

    if bar is not None:
        bar.set_description('Computing')
    kernel_start = time.perf_counter()
    cols, vals, counts = _core.compute_similarities(
        targets,
        *m1_arrays, m1.shape[1],
        *m2_arrays,
        *norms,
        a1, l1, l2, l3, t1, t2,
        stabilized_shrink, bayesian_shrink, threshold,
        k, n_output_cols,
        *filt,
        *targ,
        _EMPTY_I32 if back_perm is None else back_perm,
        num_threads, block_size, bar,
    )
    kernel_end = time.perf_counter()

    if bar is not None:
        bar.set_description(f'Building {format_output} matrix')
    out = _build_output(cols, vals, counts, targets, (n_output_rows, n_output_cols), format_output)
    _last_timings.update(kernel=kernel_end - kernel_start, output=time.perf_counter() - kernel_end)
    return out


# ---- `_s_plus` steps. CSR arrays travel as (data, indices, indptr), selectors
# as (mode, indptr, indices), norms as the kernel's six arrays (x/y tversky,
# cosine, depop; empty when unused). ----

def _kernel_values(m1, m2, binary, a1, l1, t1, t2, bayesian_shrink):
    """
    Kernel values: float32 (all ones in binary mode). Finite is checked on the
    float32 values in both modes (1e39 is inf there, and binary must not turn
    NaN/inf into 1.0). Signed input (D45) is fine: negative and cancelled (0.0)
    scores are never results. `signed` is read from the values the kernel sees,
    so binary input is never signed.
    """
    def values(name, m):
        with np.errstate(over='ignore'):  # raised below
            data = m.data.astype(np.float32, copy=False)
        if not np.isfinite(data).all():
            raise ValueError(f'{name} values must be finite (in float32)')
        return np.ones(m.nnz, dtype=np.float32) if binary else data
    m1_data, m2_data = values('matrix1', m1), values('matrix2', m2)
    signed = not binary and bool((m1_data < 0).any() or (m2_data < 0).any())
    _check_signed_rules(signed, binary, a1, l1, t1, t2, bayesian_shrink, m1_data, m2_data)
    return m1_data, m2_data


def _norms(m1_arrays, m2_arrays, n_cols, l1, l2, l3, c1, c2, additive_shrink,
           weight_depop_matrix1, weight_depop_matrix2, p1, p2):
    """The normalization terms the weights use (x per m1 row, y per m2 column); the others stay empty."""
    (m1_data, _, m1_indptr), (m2_data, m2_indices, _) = m1_arrays, m2_arrays
    Xtversky = Ytversky = Xcosine = Ycosine = Xdepop = Ydepop = _EMPTY_F32
    if l1 != 0 or l2 != 0:
        m1_sq = _row_sums(np.square(m1_data), m1_indptr)
        m2_sq = _col_sums(np.square(m2_data), m2_indices, n_cols)
    if l1 != 0:
        Xtversky, Ytversky = m1_sq, m2_sq
    if l2 != 0:
        Xcosine = np.power(m1_sq + np.float32(additive_shrink), np.float32(c1), dtype=np.float32)
        Ycosine = np.power(m2_sq + np.float32(additive_shrink), np.float32(c2), dtype=np.float32)
    if l3 != 0:
        _check_used_weights('weight_depop_matrix1 (pop1)', weight_depop_matrix1, np.diff(m1_indptr) > 0)
        _check_used_weights('weight_depop_matrix2 (pop2)', weight_depop_matrix2,
                            np.bincount(m2_indices, minlength=n_cols) > 0)
        # 'sum' popularity = sum of |x| (identical on non-negative input)
        Xdepop = _depop_weights(weight_depop_matrix1, lambda: _row_sums(np.abs(m1_data), m1_indptr),
                                np.float32(p1), len(m1_indptr) - 1)
        Ydepop = _depop_weights(weight_depop_matrix2, lambda: _col_sums(np.abs(m2_data), m2_indices, n_cols),
                                np.float32(p2), n_cols)
    return [Xtversky, Ytversky, Xcosine, Ycosine, Xdepop, Ydepop]


def _selectors(filter_cols, target_cols, m2_arrays, n_cols, num_threads):
    """
    (filter, target) selectors for the kernel. Array selectors drop their
    columns from m2 here (returned m2), so the kernel treats them as NONE.
    """
    filt = _column_selector(filter_cols, num_threads)
    targ = _column_selector(target_cols, num_threads)
    if filt[0] == MODE_ARRAY or targ[0] == MODE_ARRAY:
        keep = _array_keep_mask(filter_cols, target_cols, n_cols)
        m2_arrays = _keep_columns(*m2_arrays, keep)
    return filt, targ, m2_arrays


def _popularity_layout(m2_arrays, norms, filt, targ, n_cols, use_blocking, num_threads):
    """
    Popularity order (default, `_COLUMN_ORDER`): renumber m2 columns by
    descending count, on every call, blocked or not. Popular columns share the
    first, cache-hot blocks, and the kernel's tie rule (lower column wins) means
    the same thing on every path: ties go to the more popular column, then to
    the lower original id (stable sort). The kernel writes original ids
    (`back_perm`, None when the order is already the identity). Blocking also
    needs sorted m2 rows (binary search).
    """
    m2_data, m2_indices, m2_indptr = m2_arrays
    back_perm = None
    forward = _EMPTY_I32
    if _COLUMN_ORDER == 'popularity':
        col_nnz = np.bincount(m2_indices, minlength=n_cols)
        perm = np.argsort(-col_nnz, kind='stable').astype(np.int32)  # new col -> original col
        if not np.array_equal(perm, np.arange(n_cols, dtype=np.int32)):
            back_perm = perm
            forward = np.empty(n_cols, dtype=np.int32)
            forward[perm] = np.arange(n_cols, dtype=np.int32)
            norms = [y[perm] if j % 2 == 1 and len(y) else y for j, y in enumerate(norms)]  # y_* only
            filt, targ = (_renumbered_pattern(sel, n_cols, forward, num_threads) for sel in (filt, targ))
    if use_blocking or back_perm is not None:
        # ponytail: unblocked + permuted sorts too (one Rust pass does both); a
        # remap-only pass would skip the sort if it ever shows in a profile
        m2_data, m2_indices = _sorted_rows(m2_data, m2_indices, m2_indptr, n_cols, forward, num_threads)
    return (m2_data, m2_indices, m2_indptr), norms, filt, targ, back_perm


def _renumbered_pattern(sel, n_cols, forward, num_threads):
    """A matrix selector with its columns renumbered by `forward` and rows re-sorted; others unchanged."""
    mode, indptr, indices = sel
    if mode != MODE_MATRIX or len(indices) == 0:
        return sel
    ones = np.ones(len(indices), dtype=np.float32)
    _, indices = _sorted_rows(ones, indices, indptr, n_cols, forward, num_threads)
    return mode, indptr, indices


def _build_output(cols, vals, counts, targets, shape, format_output):
    """
    Rows come back exact (no padding), sorted by column, in target order
    (ascending, unique), so both formats wrap the kernel's arrays without a
    copy. Canonical as well: no duplicates and no zeros (the kernel drops 0.0
    scores: cancelled sums on signed input, or an f32 underflow).
    """
    if format_output == 'coo':
        out = sp.coo_array((vals, (np.repeat(targets, counts), cols)), shape=shape)
    else:
        indptr = np.zeros(shape[0] + 1, dtype=np.int64)
        indptr[targets + 1] = counts
        np.cumsum(indptr, out=indptr)
        if indptr[-1] <= _MAX_INT32_NNZ:
            indptr = indptr.astype(np.int32)
        else:  # scipy wants one index dtype: it would upcast cols anyway
            cols = cols.astype(np.int64)
        out = sp.csr_array((vals, cols, indptr), shape=shape)
    out.has_canonical_format = True
    return out
