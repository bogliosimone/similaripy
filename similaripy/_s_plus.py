"""
    s_plus on the Rust kernel: numpy prep, kernel call, output build.
"""

import time
from typing import NamedTuple, Union, Optional, List

import numpy as np
import scipy.sparse as sp

from . import _core

# Accumulator block in float32 entries (~1 MB).
DEFAULT_BLOCK_SIZE = 262144

PROGRESS_BAR_REFRESH_RATE = 3  # Hz
PROGRESS_BAR_WIDTH = 25

# Wall time (s) of the last call's kernel (scoring, top-K, per-row un-permute
# + sort, compaction) and output build (matrix wrap). Read by the benchmark
# harness (tests/benchmarks/benchmark.py), which takes prep as total - kernel -
# output, so prep includes the wrappers' normalization. Racy under concurrent
# calls: read it only after serial calls.
_last_timings = {'kernel': 0.0, 'output': 0.0}

# Kernel column order: 'popularity' (default, see `_s_plus`) or 'original'
# (no renumbering: ties go to the lower original column). Private switch for
# benchmarks and tests, not an API; don't flip it under concurrent calls.
_COLUMN_ORDER = 'popularity'

# Largest output nnz with an int32 CSR indptr; past it indptr and indices
# are int64. A module constant so tests can lower it.
_MAX_INT32_NNZ = np.iinfo(np.int32).max


# ---- The kernel's argument groups (`_core.compute_similarities` reads them by
# attribute: kernel.rs `CsrView`, `Norms`, `Params`) ----

class Csr(NamedTuple):
    """A canonical CSR matrix's arrays (float32 data, int32 ids)."""
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    n_cols: int


class Params(NamedTuple):
    """The scoring weights (`s_plus` arguments of the same names)."""
    a1: float
    l1: float
    l2: float
    l3: float
    t1: float
    t2: float
    stabilized_shrink: float
    bayesian_shrink: float
    threshold: float


class NormParams(NamedTuple):
    """The prep-only `s_plus` arguments: norms (`_norms`) and data rules
    (`_check_data`); depop1/depop2 = weight_depop_matrix1/2."""
    c1: float
    c2: float
    p1: float
    p2: float
    additive_shrink: float
    depop1: Union[str, np.ndarray]
    depop2: Union[str, np.ndarray]
    binary: bool


class Norms(NamedTuple):
    """The kernel's normalization terms (kernel.rs `Norms`): x per m1 row, y
    per m2 column, float32; None when the weights don't use them."""
    x_tversky: Optional[np.ndarray] = None
    y_tversky: Optional[np.ndarray] = None
    x_cosine: Optional[np.ndarray] = None
    y_cosine: Optional[np.ndarray] = None
    x_depop: Optional[np.ndarray] = None
    y_depop: Optional[np.ndarray] = None


def _check_args(
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
    Check the `s_plus` arguments that don't need the values, before the copy
    (fail fast); the rules on the canonical values are in `_check_data`.

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
    # denominator is kept > 0 by its rule in `_check_score_rules` (with the
    # rules for signed input). Negative weights would flip signs or
    # make NaN (a negative base to a fractional power); alpha <= 0 would flatten
    # or invert the ranking.
    for name, value in weights.items():
        _check_scalar(name, value)
    _check_scalar('a1 (alpha)', a1, strict=True)

    if block_size is not None and not (_is_int(block_size) and block_size >= 0):
        raise ValueError(f'block_size must be None (no blocking), 0 (auto) or > 0, got {block_size!r}')
    if not (_is_int(num_threads) and num_threads >= 0):
        raise ValueError(f'num_threads must be an integer >= 0, got {num_threads!r}')


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
    it is used is checked in `_check_data` (`_check_used_weights`)."""
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


def _check_data(m1, m2, params: Params, norm_params: NormParams) -> None:
    """
    The rules on the canonical values (`_canonical_csr`): finite in float32 in
    both modes (1e39 is inf there, and binary must not turn NaN/inf into 1.0),
    the score rules (`_check_score_rules`), custom depop weights > 0 where used.
    Signed input is fine: negative and cancelled (0.0) scores are never results.
    `signed` is read from the values the kernel sees, so binary input is never
    signed.
    """
    a1, l1, l3, t1, t2, bayesian_shrink = (params.a1, params.l1, params.l3, params.t1, params.t2,
                                           params.bayesian_shrink)
    binary = norm_params.binary
    for name, m in (('matrix1', m1), ('matrix2', m2)):
        if not np.isfinite(m.data).all():
            raise ValueError(f'{name} values must be finite (in float32)')
    signed = not binary and bool((m1.data < 0).any() or (m2.data < 0).any())
    _check_score_rules(signed, binary, a1, l1, t1, t2, bayesian_shrink, m1.data, m2.data)
    if l3 != 0:
        _check_used_weights('weight_depop_matrix1 (pop1)', norm_params.depop1, np.diff(m1.indptr) > 0)
        _check_used_weights('weight_depop_matrix2 (pop2)', norm_params.depop2,
                            np.bincount(m2.indices, minlength=m2.shape[1]) > 0)


def _check_score_rules(signed: bool, binary: bool, a1, l1, t1, t2, bayesian_shrink,
                        m1_data, m2_data) -> None:
    """
    Parameters that would be wrong on signed input (docs/guide.md, "Signed
    Input"), plus the tversky denominator rule, which also binds on
    non-negative weighted input.
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


# ---- Prep helpers (numpy) ----

def _canonical_csr(matrix: sp.sparray, dtype, num_threads: int) -> sp.csr_array:
    """
    The call's own canonical copy of `matrix`: CSR, `dtype` values (float32 or
    float64), int32 indices, rows sorted, duplicates summed, no explicit zeros.
    Always a copy: the caller's matrix never changes, and every later prep step
    assumes this form and may change the arrays in place. One cleaner for the
    similarities (float32) and the normalizations (the input's float type).

    Explicit zeros are wrong in binary mode (they would become 1.0), and
    unsummed duplicates give wrong norms and binary counts. One route: the
    arrays are copied once into `dtype`/int32 (zeros are dropped after the
    cast, so a float64 value below float32's range goes too when `dtype` is
    float32); canonical input
    (e.g. our own COO output, fed back) then only drops zeros, the rest goes
    through the Rust canonicalization, which replaces scipy's single-threaded
    COO -> CSR (2.2 s on a 93M-entry COO item model).

    Raises:
        ValueError: more than 2^31 - 1 stored entries (indices are int32: fail
            loudly instead of wrapping; outputs may pass 2^31, `_MAX_INT32_NNZ`).
    """
    int32_max = np.iinfo(np.int32).max
    if matrix.nnz > int32_max:
        raise ValueError(f'sparse inputs must have at most {int32_max} non-zeros')
    if matrix.format == 'coo':
        rows, cols, data = matrix.row, matrix.col, matrix.data
        if np.any(rows[1:] < rows[:-1]):
            order = np.argsort(rows, kind='stable')
            rows, cols, data = rows[order], cols[order], data[order]
        indptr = np.zeros(matrix.shape[0] + 1, dtype=np.int32)
        np.cumsum(np.bincount(rows, minlength=matrix.shape[0]), out=indptr[1:])
    else:
        csr = matrix.tocsr()
        indptr, cols, data = csr.indptr.astype(np.int32), csr.indices, csr.data
    canonical = (matrix if matrix.format == 'coo' else csr).has_canonical_format
    indices = cols.astype(np.int32)
    with np.errstate(over='ignore'):  # inf: reported by the similarities' finite check (`_check_data`)
        data = data.astype(dtype)
    if canonical:
        out = sp.csr_array((data, indices, indptr), shape=matrix.shape)
        if (data == 0).any():
            out.eliminate_zeros()
    else:
        indptr = _core.canonicalize_rows(indptr=indptr, indices=indices, data=data, num_threads=num_threads)
        nnz = indptr[-1]
        out = sp.csr_array((data[:nnz], indices[:nnz], indptr), shape=matrix.shape)
    out.has_canonical_format = True  # sorted, summed: skips scipy's re-checks
    return out


def _row_sums(values: np.ndarray, indptr: np.ndarray, dtype=None) -> np.ndarray:
    """
    Per-row sums of a CSR value array, in the values' dtype unless `dtype` is
    given; summed sequentially per row (reduceat).

    The trailing 0 lets `reduceat` take start == nnz (trailing empty rows);
    reduceat returns values[start] for empty rows, hence the reset.
    """
    starts = indptr[:-1]
    out = np.add.reduceat(np.append(values, values.dtype.type(0)), starts, dtype=dtype)
    out[starts == indptr[1:]] = 0
    return out


def _col_sums(values: np.ndarray, indices: np.ndarray, n_cols: int) -> np.ndarray:
    """Per-column sums of a CSR value array (bincount in float64, cast once to float32)."""
    return np.bincount(indices, weights=values, minlength=n_cols).astype(np.float32)


def _depop_weights(weight_spec, sums, p: float, n: int) -> np.ndarray:
    """Depopularization weights (float32): custom array ** p, 'none' (ones) or row/col sums ** p."""
    if not isinstance(weight_spec, str):
        return np.power(np.asarray(weight_spec, dtype=np.float32), p, dtype=np.float32)
    if weight_spec == 'none':
        return np.ones(n, dtype=np.float32)  # 1 ** p == 1
    return np.power(sums(), p, dtype=np.float32)


def _matrix_selector(cols, num_threads: int):
    """
    (indptr, indices) of a sparse-matrix filter_cols/target_cols, checked per
    row in the kernel; None for anything else (no selector, or a list, applied
    in `_selectors`).

    The canonical pattern only (`_canonical_csr`; the values are needed to
    canonicalize, then dropped), so an entry is selected iff its summed value is non-zero,
    the same rule as the inputs: COO and non-canonical CSR agree on cancelling
    duplicates (signed input). A matrix whose entries all cancel selects
    nothing (as explicit zeros only); an empty one is no selector.
    """
    if sp.issparse(cols) and cols.data.shape[0] != 0:
        cols = _canonical_csr(cols, np.float32, num_threads)
        return cols.indptr, cols.indices
    return None


def _is_list(cols) -> bool:
    """A non-empty list/array selector (an empty one is no selector)."""
    return isinstance(cols, (list, np.ndarray)) and len(cols) != 0


def _keep_columns(m: Csr, keep_mask) -> Csr:
    """Drops the entries whose column is not in `keep_mask`; column ids and order unchanged."""
    # ponytail: two boolean gathers over nnz; a Rust pass is the upgrade if a
    # profile shows it
    sel = keep_mask[m.indices]
    new_indptr = np.zeros(len(m.indptr), dtype=np.int32)
    np.cumsum(_row_sums(sel, m.indptr, dtype=np.int32), out=new_indptr[1:])
    return m._replace(data=m.data[sel], indices=m.indices[sel], indptr=new_indptr)


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
    prepare=None,
) -> Union[sp.csr_array, sp.coo_array]:
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
        target_cols: Columns to include in results (only compute these).
        verbose: Show progress bar during computation.
        format_output: Output matrix format: 'csr' or 'coo'.
        num_threads: Number of threads (0 = all cores on the shared rayon pool, or
            `RAYON_NUM_THREADS`; concurrent calls queue on it).
        block_size: Block size for column-blocked accumulation. Controls the trade-off
            between cache efficiency and multi-pass overhead.
            0 = auto (DEFAULT_BLOCK_SIZE entries, ~1 MB).
            None = disabled (no blocking).
            int > 0 = explicit size in number of float32 entries.
        prepare: Private (p3alpha, rp3beta): `prepare(m1, m2)` gets the call's
            canonical copies before the data checks, changes their values in
            place, and returns custom m2 depop weights (or None).

    Returns:
        A sparse matrix of shape (n_rows, n_cols) in the specified format,
        containing the top-k similarity scores: at most k entries per row, rows
        sorted by column, canonical (`has_canonical_format`).

    Raises:
        ValueError: invalid shapes or parameters (see `_check_args`),
            a negative block_size, more than 2^31 - 1 non-zeros in an input, input
            values not finite in float32, a1 != 1 or bayesian_shrink on signed input
            (values < 0, not binary), or l1 != 0 with t1, t2 whose tversky
            term can be <= 0 on weighted input (see `_check_score_rules`), values
            out of float32 range for a score computed from pre-scaled inputs
            (see `_prescale`).
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
            Params(a1, l1, l2, l3, t1, t2, stabilized_shrink, bayesian_shrink, threshold),
            NormParams(c1, c2, p1, p2, additive_shrink, weight_depop_matrix1, weight_depop_matrix2, binary),
            filter_cols, target_cols, format_output, num_threads, block_size, prepare,
        )
    except BaseException:
        if bar is not None:
            bar.close('Failed')
        raise
    if bar is not None:
        bar.close('Done')
    return res


def _s_plus(
    bar, matrix1, matrix2, targets, k, params: Params, norm_params: NormParams,
    filter_cols, target_cols, format_output, num_threads, block_size, prepare,
):
    m1 = _canonical_csr(matrix1, np.float32, num_threads)
    m2 = _canonical_csr(matrix2, np.float32, num_threads)
    if prepare is not None and (w := prepare(m1, m2)) is not None:
        norm_params = norm_params._replace(depop2=w)
    n_output_rows, n_output_cols = m1.shape[0], m2.shape[1]
    _check_data(m1, m2, params, norm_params)
    m1_data, m2_data = (np.ones(m.nnz, dtype=np.float32) if norm_params.binary else m.data
                        for m in (m1, m2))
    m1_arrays = Csr(m1_data, m1.indices, m1.indptr, m1.shape[1])
    m2_arrays = Csr(m2_data, m2.indices, m2.indptr, n_output_cols)

    if block_size == 0:  # auto; None (no blocking) and int > 0 go as they are
        block_size = DEFAULT_BLOCK_SIZE

    norms = _norms(m1_arrays, m2_arrays, n_output_cols, params, norm_params)
    params, norms = _depop_as_cosine(params, norms)
    if _product_form(params):
        _prescale(m1_arrays, m2_arrays, params.l2, norms)
        params, norms = params._replace(l2=0), Norms()  # plain dot of the scaled inputs: DOT mode
    filt, targ, m2_arrays = _selectors(filter_cols, target_cols, m2_arrays, n_output_cols, num_threads)
    m2_arrays, norms, filt, targ, new_to_old = _popularity_layout(
        m2_arrays, norms, filt, targ, n_output_cols, num_threads)

    if bar is not None:
        bar.set_description('Computing')
    kernel_start = time.perf_counter()
    cols, vals, counts = _core.compute_similarities(
        targets=targets, m1=m1_arrays, m2=m2_arrays, norms=norms,
        params=params,
        k=k, filter=filt, target=targ, col_map=new_to_old,
        num_threads=num_threads, block_size=block_size, progress=bar,
    )
    kernel_end = time.perf_counter()

    if bar is not None:
        bar.set_description(f'Building {format_output} matrix')
    out = _build_output(cols, vals, counts, targets, (n_output_rows, n_output_cols), format_output)
    _last_timings.update(kernel=kernel_end - kernel_start, output=time.perf_counter() - kernel_end)
    return out


# ---- `_s_plus` steps. CSR arrays travel as `Csr`, matrix selectors as
# (indptr, indices) or None, norms as `Norms`. ----

def _norms(m1_arrays, m2_arrays, n_cols, params: Params, norm_params: NormParams):
    """
    The normalization terms the weights use (x per m1 row, y per m2 column); the
    others stay None. float32, for the kernel and `_prescale` alike.
    """
    l1, l2, l3 = params.l1, params.l2, params.l3
    c1, c2, p1, p2 = norm_params.c1, norm_params.c2, norm_params.p1, norm_params.p2
    additive_shrink = norm_params.additive_shrink
    weight_depop_matrix1, weight_depop_matrix2 = norm_params.depop1, norm_params.depop2
    m1_data, m1_indptr, m2_data, m2_indices = m1_arrays.data, m1_arrays.indptr, m2_arrays.data, m2_arrays.indices
    norms = Norms()
    if l1 != 0 or l2 != 0:
        m1_sq = _row_sums(np.square(m1_data), m1_indptr)
        m2_sq = _col_sums(np.square(m2_data), m2_indices, n_cols)
    if l1 != 0:
        norms = norms._replace(x_tversky=m1_sq, y_tversky=m2_sq)
    if l2 != 0:
        f32 = np.float32
        norms = norms._replace(x_cosine=np.power(m1_sq + f32(additive_shrink), f32(c1), dtype=f32),
                               y_cosine=np.power(m2_sq + f32(additive_shrink), f32(c2), dtype=f32))
    if l3 != 0:
        # 'sum' popularity = sum of |x| (identical on non-negative input)
        norms = norms._replace(
            x_depop=_depop_weights(weight_depop_matrix1, lambda: _row_sums(np.abs(m1_data), m1_indptr),
                                   np.float32(p1), len(m1_indptr) - 1),
            y_depop=_depop_weights(weight_depop_matrix2, lambda: _col_sums(np.abs(m2_data), m2_indices, n_cols),
                                   np.float32(p2), n_cols))
    return norms


def _product_form(p: Params) -> bool:
    """
    The score is xy / (l X[row] Y[col]): one cosine or one depop term (an
    additive shrink sits inside the norms), no power, no other shrink. Then it
    is the plain dot product of m1 rows scaled by 1 / (l X) and m2 columns by
    1 / Y (`_prescale`).
    """
    return (p.a1 == 1 and p.l1 == 0 and (p.l2 != 0) != (p.l3 != 0)
            and p.stabilized_shrink == 0 and p.bayesian_shrink == 0)


def _prescale(m1_arrays, m2_arrays, l, norms) -> None:
    """
    For `_product_form` (after `_depop_as_cosine`, so the cosine slots): scales
    m1 rows by 1 / (l X) and m2 columns by 1 / Y, in place (the call's own
    copies), so the kernel can run DOT; 0 where a norm is 0 (an empty
    row/column: the kernel scores 0 there too). A scaled value may underflow
    to 0.0: the kernel scores it 0, never a result.

    Raises ValueError on values out of float32 range for the score: a scale 0
    or inf on a row/column with entries (its sum of squares or popularity
    underflows or overflows, e.g. data x 1e-25 or x 1e19), or a scaled value
    that overflows; every m1 row is checked, `target_rows` or not (scaled
    whole). The kernel paths keep float32 norms too, without the check.
    """
    m1_data, m1_indptr, m2_data, m2_indices = m1_arrays.data, m1_arrays.indptr, m2_arrays.data, m2_arrays.indices
    X, Y = norms.x_cosine, norms.y_cosine
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        sx = np.divide(1, np.float32(l) * X, out=np.zeros_like(X), where=X != 0)
        sy = np.divide(1, Y, out=np.zeros_like(Y), where=Y != 0)
        m1_data *= np.repeat(sx, np.diff(m1_indptr))
        m2_data *= sy[m2_indices]
    used_x, used_y = np.diff(m1_indptr) > 0, np.bincount(m2_indices, minlength=len(Y)) > 0
    ok = (all(np.isfinite(a).all() and a.all() for a in (sx[used_x], sy[used_y]))
          and np.isfinite(m1_data).all() and np.isfinite(m2_data).all())
    if not ok:
        raise ValueError('input values out of float32 range for this similarity (a norm or a '
                         'scaled value underflows or overflows): rescale the input')


def _depop_as_cosine(params: Params, norms):
    """
    Depop-only weights (rp3beta) through the cosine slots: the kernel's COSINE
    mode instead of FULL, same bits. Both compute the term as `l * (x * y)`,
    and FULL's den `0 + 0 + dp + shrink` adds only zeros to it (`score` in
    kernel.rs); den == 0 gives 0 in both. A bayesian shrink stays FULL.
    """
    if params.l1 == 0 and params.l2 == 0 and params.l3 != 0:
        return params._replace(l2=params.l3, l3=0), norms._replace(
            x_cosine=norms.x_depop, y_cosine=norms.y_depop, x_depop=None, y_depop=None)
    return params, norms


def _selectors(filter_cols, target_cols, m2_arrays, n_cols, num_threads):
    """
    (filter, target) matrix selectors for the kernel (`_matrix_selector`).
    List selectors drop their columns from m2 here (returned m2): the target
    list (or all columns) minus the filter list (ids validated in range).
    """
    filt = _matrix_selector(filter_cols, num_threads)
    targ = _matrix_selector(target_cols, num_threads)
    if _is_list(filter_cols) or _is_list(target_cols):
        keep = np.ones(n_cols, dtype=bool)
        if _is_list(target_cols):
            keep = np.zeros(n_cols, dtype=bool)
            keep[np.asarray(target_cols)] = True
        if _is_list(filter_cols):
            keep[np.asarray(filter_cols)] = False
        m2_arrays = _keep_columns(m2_arrays, keep)
    return filt, targ, m2_arrays


def _popularity_layout(m2_arrays, norms, filt, targ, n_cols, num_threads):
    """
    Popularity order (default, `_COLUMN_ORDER`): renumber m2 columns by
    descending count, on every call, blocked or not. Popular columns share the
    first, cache-hot blocks, and the kernel's tie rule (lower column wins) means
    the same thing on every path: ties go to the more popular column, then to
    the lower original id (stable sort); 'original' keeps the ids (one path:
    an identity remap). The kernel writes original ids (`new_to_old`).
    Blocking needs sorted m2 rows (binary search): canonical m2 is, and a
    renumbering re-sorts.
    """
    if _COLUMN_ORDER == 'popularity':
        new_to_old = np.argsort(-np.bincount(m2_arrays.indices, minlength=n_cols), kind='stable').astype(np.int32)
    else:
        new_to_old = np.arange(n_cols, dtype=np.int32)
    old_to_new = np.empty(n_cols, dtype=np.int32)
    old_to_new[new_to_old] = np.arange(n_cols, dtype=np.int32)
    norms = norms._replace(y_tversky=_permuted(norms.y_tversky, new_to_old),
                           y_cosine=_permuted(norms.y_cosine, new_to_old),
                           y_depop=_permuted(norms.y_depop, new_to_old))
    for sel in (filt, targ):  # matrix selectors: the pattern only
        if sel is not None:
            _core.remap_sort_rows(indptr=sel[0], indices=sel[1], data=None, forward=old_to_new,
                                  num_threads=num_threads)
    # ponytail: unblocked calls re-sort too (one Rust pass does both); a
    # remap-only pass would skip the sort if it ever shows in a profile
    _core.remap_sort_rows(indptr=m2_arrays.indptr, indices=m2_arrays.indices, data=m2_arrays.data,
                          forward=old_to_new, num_threads=num_threads)
    return m2_arrays, norms, filt, targ, new_to_old


def _permuted(y, new_to_old):
    """A per-column norm in the new column order (None: unused)."""
    return None if y is None else y[new_to_old]


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
