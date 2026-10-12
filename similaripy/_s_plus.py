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

# Wall time (s) of the last kernel call, read by the benchmark harness.
# Racy under concurrent calls: read it only after serial calls.
_last_kernel_time = 0.0

_EMPTY_F32 = np.empty(0, dtype=np.float32)
_EMPTY_I32 = np.empty(0, dtype=np.int32)


def get_num_threads() -> int:
    """Threads used with `num_threads=0` (rayon global pool, `RAYON_NUM_THREADS`)."""
    return _core.get_num_threads()


def validate_s_plus_inputs(
    matrix1: sp.sparray,
    matrix2: sp.sparray,
    weight_depop_matrix1: Union[str, np.ndarray],
    weight_depop_matrix2: Union[str, np.ndarray],
    k: int,
    target_rows: Optional[Union[List, np.ndarray]],
    filter_cols: Optional[Union[List, np.ndarray, sp.sparray]],
    target_cols: Optional[Union[List, np.ndarray, sp.sparray]],
    verbose: bool,
    format_output: str
) -> None:
    """
    Validate input parameters for s_plus function.

    Args:
        matrix1: First input matrix.
        matrix2: Second input matrix.
        weight_depop_matrix1: Depopularization weights for matrix1.
        weight_depop_matrix2: Depopularization weights for matrix2.
        k: Number of top items to keep.
        target_rows: Rows to compute.
        filter_cols: Columns to filter out.
        target_cols: Columns to target.
        verbose: Whether to show progress.
        format_output: Output format ('coo' or 'csr').

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

    # Check k parameter
    if k < 1:
        raise ValueError(f'k must be >= 1, got {k}')

    # Check depopularization weights
    if not (len(weight_depop_matrix1) == matrix1.shape[0] or
            weight_depop_matrix1 in ('none', 'sum')):
        raise ValueError(
            f'weight_depop_matrix1 must be array of length {matrix1.shape[0]} '
            f'or one of ("none", "sum"), got length {len(weight_depop_matrix1)}'
        )

    if not (len(weight_depop_matrix2) == matrix2.shape[1] or
            weight_depop_matrix2 in ('none', 'sum')):
        raise ValueError(
            f'weight_depop_matrix2 must be array of length {matrix2.shape[1]} '
            f'or one of ("none", "sum"), got length {len(weight_depop_matrix2)}'
        )

    # Check target_rows
    if target_rows is not None and len(target_rows) > matrix1.shape[0]:
        raise ValueError(
            f'target_rows length ({len(target_rows)}) cannot exceed '
            f'matrix1.shape[0] ({matrix1.shape[0]})'
        )

    # Check filter_cols format and shape
    if filter_cols is not None:
        if not (sp.issparse(filter_cols) or isinstance(filter_cols, (list, np.ndarray))):
            raise TypeError(
                'filter_cols must be a sparse matrix, list, numpy array, or None'
            )
        # Validate shape for sparse matrices
        if sp.issparse(filter_cols) and filter_cols.data.shape[0] != 0:
            expected_shape = (matrix1.shape[0], matrix2.shape[1])
            if filter_cols.shape != expected_shape:
                raise ValueError(
                    f'filter_cols shape {filter_cols.shape} does not match expected '
                    f'shape {expected_shape}'
                )

    # Check target_cols format and shape
    if target_cols is not None:
        if not (sp.issparse(target_cols) or isinstance(target_cols, (list, np.ndarray))):
            raise TypeError(
                'target_cols must be a sparse matrix, list, numpy array, or None'
            )
        # Validate shape for sparse matrices
        if sp.issparse(target_cols) and target_cols.data.shape[0] != 0:
            expected_shape = (matrix1.shape[0], matrix2.shape[1])
            if target_cols.shape != expected_shape:
                raise ValueError(
                    f'target_cols shape {target_cols.shape} does not match expected '
                    f'shape {expected_shape}'
                )

    # Check verbose
    if not isinstance(verbose, bool):
        raise TypeError(f'verbose must be boolean, got {type(verbose).__name__}')

    # Check format_output
    if format_output not in ('coo', 'csr'):
        raise ValueError(f"format_output must be 'coo' or 'csr', got '{format_output}'")


# ---- Prep helpers (numpy, D19) ----

def _csr_without_zeros(matrix: sp.sparray) -> sp.csr_array:
    """
    CSR with no explicit zeros, without touching the caller's matrix.

    Explicit zeros are wrong in binary mode (they would become 1.0, so counts,
    norms and dot products change). Copies only when there are zeros.
    """
    matrix = matrix.tocsr()
    if (matrix.data == 0).any():
        matrix = matrix.copy()
        matrix.eliminate_zeros()
    return matrix


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
    if isinstance(weight_spec, (list, np.ndarray)):
        return np.power(weight_spec, p, dtype=np.float32)
    if weight_spec == 'none':
        return np.ones(n, dtype=np.float32)  # 1 ** p == 1
    return np.power(sums(), p, dtype=np.float32)


def _column_selector(cols):
    """
    (mode, indptr, indices) for filter_cols/target_cols.

    Matrix mode: sorted int32 CSR pattern built on a copy (0.7.0 sorted and
    eliminated zeros on the caller's matrix).
    """
    if sp.issparse(cols) and cols.data.shape[0] != 0:
        cols = cols.tocsr(copy=True)
        cols.eliminate_zeros()
        cols.sort_indices()
        return MODE_MATRIX, cols.indptr.astype(np.int32, copy=False), cols.indices.astype(np.int32, copy=False)
    if isinstance(cols, (list, np.ndarray)) and len(cols) != 0:
        return MODE_ARRAY, _EMPTY_I32, _EMPTY_I32
    return MODE_NONE, _EMPTY_I32, _EMPTY_I32


def _array_keep_mask(filter_cols, target_cols, n_cols: int) -> np.ndarray:
    """
    Mask of the columns kept by array-mode selectors: the target array (or all columns)
    minus the filter array. Out-of-range ids are ignored. Only called when at
    least one selector is an array; a matrix selector here counts as "all".
    """
    if isinstance(target_cols, (list, np.ndarray)) and len(target_cols) != 0:
        mask = np.zeros(n_cols, dtype=bool)
        idx = np.asarray(target_cols, dtype=np.int64)
        mask[idx[(idx >= 0) & (idx < n_cols)]] = True
    else:
        mask = np.ones(n_cols, dtype=bool)
    if isinstance(filter_cols, (list, np.ndarray)) and len(filter_cols) != 0:
        idx = np.asarray(filter_cols, dtype=np.int64)
        mask[idx[(idx >= 0) & (idx < n_cols)]] = False
    return mask


def _keep_columns(data, indices, indptr, keep_mask):
    """Drops the entries whose column is not in `keep_mask`; column ids and order unchanged."""
    # ponytail: slower than 0.7.0's Cython loop (two boolean gathers); a Rust
    # pass is the upgrade, tracked for the non-hot-path polish
    sel = keep_mask[indices]
    new_indptr = np.zeros(len(indptr), dtype=np.int32)
    np.cumsum(_row_sums(sel, indptr, dtype=np.int32), out=new_indptr[1:])
    return data[sel], indices[sel], new_indptr


def _sorted_rows(data, indices, indptr, n_cols):
    """Rows sorted by column id (scipy, in C); copies first, so the caller's arrays never change."""
    m = sp.csr_array((data, indices, indptr), shape=(len(indptr) - 1, n_cols), copy=False)
    if m.has_sorted_indices:
        return data, indices
    m = sp.csr_array((data.copy(), indices.copy(), indptr), shape=m.shape, copy=False)
    m.sort_indices()
    return m.data, m.indices.astype(np.int32, copy=False)


def _build_csr(rows, cols, values, n_rows: int, n_cols: int) -> sp.csr_array:
    """
    CSR from the kernel's (rows, cols, values) slots, zeros dropped
    (= 0.7.0's coo_to_csr + eliminate_zeros, same order within rows).
    """
    # 0.7.0 picks the index dtype before dropping the zero padding
    idx_dtype = np.int32 if max(len(values), n_cols) <= np.iinfo(np.int32).max else np.int64
    keep = values != 0
    rows, cols, values = rows[keep], cols[keep], values[keep]
    if np.any(rows[1:] < rows[:-1]):  # target_rows not ascending
        order = np.argsort(rows, kind='stable')
        rows, cols, values = rows[order], cols[order], values[order]
    indptr = np.zeros(n_rows + 1, dtype=idx_dtype)
    np.cumsum(np.bincount(rows, minlength=n_rows), out=indptr[1:])
    return sp.csr_array((values, cols.astype(idx_dtype, copy=False), indptr), shape=(n_rows, n_cols))


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
        matrix1: First sparse matrix (typically item-user interactions).
        matrix2: Optional second matrix. If None, uses matrix1.T (self-similarity).
        weight_depop_matrix1: Depopularization weights for matrix1: 'none', 'sum', or custom array.
        weight_depop_matrix2: Depopularization weights for matrix2: 'none', 'sum', or custom array.
        p1: Power for depopularization weights on matrix1.
        p2: Power for depopularization weights on matrix2.
        a1: Power applied to dot product values.
        l1: Weight for Tversky similarity term.
        l2: Weight for Cosine similarity term.
        l3: Weight for Depopularization term.
        t1: Tversky parameter for matrix1.
        t2: Tversky parameter for matrix2.
        c1: Cosine power exponent for matrix1.
        c2: Cosine power exponent for matrix2.
        k: Number of top similar items to keep per row.
        stabilized_shrink: Stabilized shrinkage parameter.
        bayesian_shrink: Bayesian shrinkage parameter.
        additive_shrink: Additive shrinkage for cosine normalization.
        threshold: Minimum similarity threshold.
        binary: If True, treat all non-zero values as 1.0 (set theory).
        target_rows: Specific rows to compute. If None, compute all rows.
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
        containing the top-k similarity scores.

    Raises:
        ValueError: invalid shapes or parameters (see `validate_s_plus_inputs`),
            a negative block_size, or int32 overflow of inputs or output.
    """
    if matrix2 is None:
        matrix2 = matrix1.T

    validate_s_plus_inputs(
        matrix1=matrix1,
        matrix2=matrix2,
        weight_depop_matrix1=weight_depop_matrix1,
        weight_depop_matrix2=weight_depop_matrix2,
        k=k,
        target_rows=target_rows,
        filter_cols=filter_cols,
        target_cols=target_cols,
        verbose=verbose,
        format_output=format_output
    )

    if block_size is not None and block_size < 0:
        raise ValueError(f'block_size must be None (no blocking), 0 (auto) or > 0, got {block_size}')

    k = min(k, matrix2.shape[1])  # do not allocate unnecessary space

    if target_rows is None:
        targets = np.arange(matrix1.shape[0], dtype=np.int32)
    else:
        targets = np.ascontiguousarray(target_rows, dtype=np.int32)
    n_targets = len(targets)

    # Indices, indptrs and the output are int32: fail loudly instead of wrapping
    int32_max = np.iinfo(np.int32).max
    if matrix1.nnz > int32_max or matrix2.nnz > int32_max:
        raise ValueError(f"input matrices must have at most {int32_max} non-zeros")
    if n_targets * k > int32_max:
        raise ValueError(
            f"output too large: {n_targets} target rows * k={k} exceeds {int32_max} entries; "
            "lower k or split target_rows"
        )

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
    global _last_kernel_time

    m1 = _csr_without_zeros(matrix1)
    m2 = _csr_without_zeros(matrix2)
    n_output_rows, n_output_cols = m1.shape[0], m2.shape[1]

    # Block size: 0 = auto, None = disabled (0 for the kernel), int > 0 = explicit
    if block_size is None:
        block_size = 0
    elif block_size == 0:
        block_size = DEFAULT_BLOCK_SIZE
    use_blocking = block_size > 0 and n_output_cols > block_size

    # Kernel inputs: float32 values (all ones in binary mode), int32 indices
    def values(m):
        return np.ones(m.nnz, dtype=np.float32) if binary else m.data.astype(np.float32, copy=False)
    m1_data, m2_data = values(m1), values(m2)
    m1_indices = m1.indices.astype(np.int32, copy=False)
    m1_indptr = m1.indptr.astype(np.int32, copy=False)
    m2_indices = m2.indices.astype(np.int32, copy=False)
    m2_indptr = m2.indptr.astype(np.int32, copy=False)

    # Normalization terms, only the ones the weights use
    Xtversky = Ytversky = Xcosine = Ycosine = Xdepop = Ydepop = _EMPTY_F32
    if l1 != 0 or l2 != 0:
        m1_sq = _row_sums(np.square(m1_data), m1_indptr)
        m2_sq = _col_sums(np.square(m2_data), m2_indices, n_output_cols)
    if l1 != 0:
        Xtversky, Ytversky = m1_sq, m2_sq
    if l2 != 0:
        Xcosine = np.power(m1_sq + np.float32(additive_shrink), np.float32(c1), dtype=np.float32)
        Ycosine = np.power(m2_sq + np.float32(additive_shrink), np.float32(c2), dtype=np.float32)
    if l3 != 0:
        Xdepop = _depop_weights(weight_depop_matrix1, lambda: _row_sums(m1_data, m1_indptr),
                                np.float32(p1), n_output_rows)
        Ydepop = _depop_weights(weight_depop_matrix2, lambda: _col_sums(m2_data, m2_indices, n_output_cols),
                                np.float32(p2), n_output_cols)

    filter_mode, filter_indptr, filter_indices = _column_selector(filter_cols)
    target_mode, target_indptr, target_indices = _column_selector(target_cols)

    # Array selectors: drop the columns from m2 here, the kernel then treats them as NONE
    if filter_mode == MODE_ARRAY or target_mode == MODE_ARRAY:
        keep = _array_keep_mask(filter_cols, target_cols, n_output_cols)
        m2_data, m2_indices, m2_indptr = _keep_columns(m2_data, m2_indices, m2_indptr, keep)

    # Blocking: renumber m2 columns by descending popularity (popular columns
    # share the first, cache-hot blocks; output un-permuted below), and sort
    # m2 rows, which the blocked kernel binary-searches.
    back_perm = None
    if use_blocking:
        col_nnz = np.bincount(m2_indices, minlength=n_output_cols)
        perm = np.argsort(-col_nnz, kind='stable').astype(np.int32)  # new col -> original col
        if not np.array_equal(perm, np.arange(n_output_cols, dtype=np.int32)):
            back_perm = perm
            forward = np.empty(n_output_cols, dtype=np.int32)
            forward[perm] = np.arange(n_output_cols, dtype=np.int32)
            m2_indices = forward[m2_indices]
            if l1 != 0:
                Ytversky = Ytversky[perm]
            if l2 != 0:
                Ycosine = Ycosine[perm]
            if l3 != 0:
                Ydepop = Ydepop[perm]
            if filter_mode == MODE_MATRIX and len(filter_indices) > 0:
                ones = np.ones(len(filter_indices), dtype=np.float32)
                _, filter_indices = _sorted_rows(ones, forward[filter_indices], filter_indptr, n_output_cols)
            if target_mode == MODE_MATRIX and len(target_indices) > 0:
                ones = np.ones(len(target_indices), dtype=np.float32)
                _, target_indices = _sorted_rows(ones, forward[target_indices], target_indptr, n_output_cols)
        m2_data, m2_indices = _sorted_rows(m2_data, m2_indices, m2_indptr, n_output_cols)

    # Output slots: row i owns [i*k, (i+1)*k), the kernel leaves unused slots
    # untouched. With a permutation, cols starts at -1 to tell padding from a
    # real (col 0, 0.0) result; padding goes back to 0 below.
    n_out = len(targets) * k
    rows = np.zeros(n_out, dtype=np.int32)
    cols = np.full(n_out, -1 if back_perm is not None else 0, dtype=np.int32)
    vals = np.zeros(n_out, dtype=np.float32)

    if bar is not None:
        bar.set_description('Computing')
    kernel_start = time.perf_counter()
    _core.compute_similarities(
        targets,
        m1_data, m1_indices, m1_indptr, m1.shape[1],
        m2_data, m2_indices, m2_indptr,
        Xtversky, Ytversky, Xcosine, Ycosine, Xdepop, Ydepop,
        a1, l1, l2, l3, t1, t2,
        stabilized_shrink, bayesian_shrink, threshold,
        k, n_output_cols,
        filter_mode, filter_indptr, filter_indices,
        target_mode, target_indptr, target_indices,
        rows, cols, vals,
        num_threads, block_size, bar,
    )
    _last_kernel_time = time.perf_counter() - kernel_start

    # Un-permute output columns; padding (-1) back to column 0
    if back_perm is not None:
        used = cols >= 0
        cols[used] = back_perm[cols[used]]
        cols[~used] = 0

    if bar is not None:
        bar.set_description(f'Building {format_output} matrix')
    if format_output == 'coo':
        return sp.coo_array((vals, (rows, cols)), shape=(n_output_rows, n_output_cols), dtype=np.float32)
    return _build_csr(rows, cols, vals, n_output_rows, n_output_cols)
