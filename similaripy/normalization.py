"""Normalizations and weightings of sparse matrices (row/column norms,
TF-IDF, BM25, binarize, time and rank decay). Also at the top level
(`sim.bm25`, as in 0.x)."""

from math import e

import numpy as np
import scipy.sparse as sps

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import SparseInput, _canonical_csr

__all__ = ["binarize", "bm25", "bm25plus", "normalize", "rank_decay", "tfidf", "time_decay"]

# ---- Valid mode strings ----
_NORMALIZATIONS = ("l1", "l2", "max")
_TF_MODES = ("binary", "raw", "sqrt", "freq", "log")
_IDF_MODES = ("unary", "base", "smooth", "prob", "bm25")
_SHAPES = ("exponential", "power", "linear", "step")


# ---- Private helpers ----


def _canonical(X: SparseInput, axis: int = 1) -> sps.csr_array:
    """
    The call's own canonical CSR copy of X (axis 1) or X.T (axis 0), see
    `_canonical_csr`: float32/float64 values kept, other value types -> float32;
    int32 indices.

    Raises:
        ValueError: If axis is not 0 or 1, or X has more than 2^31 - 1 entries.
        TypeError: If X is not a sparse matrix.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if not sps.issparse(X):
        raise TypeError("X must be a sparse matrix")
    dtype = X.dtype if X.dtype in (np.float32, np.float64) else np.float32
    return _canonical_csr(X.T if axis == 0 else X, dtype, 0)


def _transformed(X: SparseInput, axis: int, op, name: str | None = None, **kwargs) -> sps.csr_array:
    """
    The canonical copy of X with `op` (an in-place `_core` normalization) applied
    to its rows (axis 1) or columns (axis 0); `name`: a term weighting, whose
    input must be non-negative.
    """
    X = _canonical(X, axis)
    _check_finite(X)
    if name is not None:
        _check_non_negative(X, name)
    op(data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape, **kwargs)
    # ty 0.0.90 + scipy-stubs mis-resolve csr_array.T (shape-generic overloads)
    return X.T.tocsr() if axis == 0 else X  # ty: ignore[invalid-attribute-access]


def _check_finite(X: sps.csr_array) -> None:
    """NaN/inf would spread through a row's norm or idf, silently (as the
    similarities, checked on the canonical copy)."""
    if not np.isfinite(X.data).all():
        raise ValueError("X values must be finite")


def _check_non_negative(X: sps.csr_array, name: str) -> None:
    """Term weightings need counts: a negative value would give a NaN log or a
    negative document length (signed input). Checked on the canonical data,
    so +1/-1 duplicates sum to 0 and pass."""
    if (X.data < 0).any():
        raise ValueError(
            f"{name} needs non-negative values (term frequencies); signed input (values < 0) is not supported"
        )


def _validate_modes(tf_mode: str, idf_mode: str) -> None:
    """
    Validate TF and IDF mode strings.

    Raises:
        ValueError: If tf_mode or idf_mode is not a recognized mode.
    """
    if tf_mode not in _TF_MODES:
        raise ValueError(f"tf_mode must be one of {_TF_MODES}, got '{tf_mode}'")
    if idf_mode not in _IDF_MODES:
        raise ValueError(f"idf_mode must be one of {_IDF_MODES}, got '{idf_mode}'")


# ---- Public API ----


def normalize(
    X: sps.sparray | sps.spmatrix,
    norm: str = "l2",
    axis: int = 1,
) -> sps.csr_array:
    """
    Normalize a sparse matrix along rows or columns using L1, L2, or max-norm.

    Signed input is fine: l1 divides by the sum of |x|, l2 by sqrt(sum of x^2),
    max by the largest |x|. Rows of zeros (empty after canonicalization) stay
    unchanged.

    Args:
        X: Input sparse matrix.
        norm: Normalization method ('l1', 'l2', or 'max').
        axis: Normalize rows (1) or columns (0).

    Returns:
        Normalized CSR matrix.

    Raises:
        ValueError: on a NaN or inf value.
    """
    if norm not in _NORMALIZATIONS:
        raise ValueError(f"norm must be one of {_NORMALIZATIONS}, got '{norm}'")
    return _transformed(X, axis, _core.normalize, norm=norm)


def binarize(X: sps.sparray | sps.spmatrix) -> sps.csr_array:
    """
    Set every non-zero entry to 1.0 (set semantics): duplicates are summed
    first, then explicit zeros dropped. A negative value is an interaction
    too: it becomes 1.0.

    Same values as `binary=True` in the similarity functions, as a separate
    step: binarize only one of the two matrices, or binarize before another
    normalization.

    Args:
        X: Input sparse matrix.

    Returns:
        Binarized CSR matrix.

    Raises:
        ValueError: on a NaN or inf value (checked before the values become 1).
    """
    X = _canonical(X)
    _check_finite(X)  # before the values become 1
    X.data[:] = 1
    return X


def bm25(
    X: sps.sparray | sps.spmatrix,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    logbase: float = e,
    tf_mode: str = "raw",
    idf_mode: str = "bm25",
) -> sps.csr_array:
    """
    Apply BM25 normalization to a sparse matrix.

    Args:
        X: Input sparse matrix; values >= 0 (term frequencies), checked after
            summing duplicates.
        axis: Normalize rows (1) or columns (0).
        k1: Term saturation parameter.
        b: Length normalization parameter.
        logbase: Logarithm base.
        tf_mode: Term frequency mode ('raw', 'log', 'sqrt', etc.).
        idf_mode: Inverse document frequency mode ('bm25', 'smooth', etc.).

    Returns:
        BM25-normalized CSR matrix.

    Raises:
        ValueError: on a negative, NaN or inf value.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(
        X, axis, _core.bm25plus, "bm25", k1=k1, b=b, delta=0.0, tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase
    )


def bm25plus(
    X: sps.sparray | sps.spmatrix,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    logbase: float = e,
    tf_mode: str = "raw",
    idf_mode: str = "bm25",
) -> sps.csr_array:
    """
    Apply BM25+ normalization to a sparse matrix.

    Args:
        X: Input sparse matrix; values >= 0 (term frequencies), checked after
            summing duplicates.
        axis: Normalize rows (1) or columns (0).
        k1: Term saturation parameter.
        b: Length normalization parameter.
        delta: BM25+ boosting parameter.
        logbase: Logarithm base.
        tf_mode: Term frequency mode.
        idf_mode: Inverse document frequency mode.

    Returns:
        BM25+ normalized CSR matrix.

    Raises:
        ValueError: on a negative, NaN or inf value.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(
        X,
        axis,
        _core.bm25plus,
        "bm25plus",
        k1=k1,
        b=b,
        delta=delta,
        tf_mode=tf_mode,
        idf_mode=idf_mode,
        logbase=logbase,
    )


def tfidf(
    X: sps.sparray | sps.spmatrix,
    axis: int = 1,
    logbase: float = e,
    tf_mode: str = "sqrt",
    idf_mode: str = "smooth",
) -> sps.csr_array:
    """
    Apply TF-IDF normalization to a sparse matrix.

    Args:
        X: Input sparse matrix; values >= 0 (term frequencies), checked after
            summing duplicates.
        axis: Normalize rows (1) or columns (0).
        logbase: Logarithm base.
        tf_mode: Term frequency mode.
        idf_mode: Inverse document frequency mode.

    Returns:
        TF-IDF normalized CSR matrix.

    Raises:
        ValueError: on a negative, NaN or inf value.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(X, axis, _core.tfidf, "tfidf", tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)


# ---- Time and rank decay ----


def _scale(scale) -> float:
    """A positive number (np.inf = no decay); durations as seconds."""
    if isinstance(scale, np.timedelta64):
        scale = scale / np.timedelta64(1, "s")
    elif hasattr(scale, "total_seconds"):  # datetime.timedelta, pd.Timedelta
        scale = scale.total_seconds()
    scale = float(scale)
    if not scale > 0:  # also NaN
        raise ValueError(f"scale must be > 0 (np.inf = no decay), got {scale}")
    return scale


def _weights(age: np.ndarray, shape: str, scale: float, exponent) -> np.ndarray:
    """The weight of each age (float64): 1 at age 0, 0.5 at age = scale for
    exponential and power, 0 from age = scale for linear and step."""
    if shape not in _SHAPES:
        raise ValueError(f"shape must be one of {_SHAPES}, got {shape!r}")
    if exponent is not None and shape != "power":
        raise ValueError(f"exponent applies to shape='power' only, got shape={shape!r}")
    x = age / scale  # scale = inf: 0, every weight 1
    if shape == "exponential":
        return np.exp2(-x)
    if shape == "power":
        p = 1.0 if exponent is None else float(exponent)
        if not p > 0:
            raise ValueError(f"exponent must be > 0, got {exponent}")
        return (1 + np.expm1(np.log(2) / p) * x) ** -p
    if shape == "linear":
        return np.maximum(1 - x, 0)
    return (x < 1).astype(np.float64)


def _timestamps_of(X: SparseInput, T) -> tuple[sps.csr_array, sps.csr_array]:
    """X canonical, T as given (explicit zeros are timestamps), both checked."""
    X = _canonical(X)
    _check_finite(X)
    if not sps.issparse(T):
        raise TypeError("T must be a sparse matrix (the T from from_interactions)")
    T = sps.csr_array(T)
    if T.shape != X.shape:
        raise ValueError(f"T must have X's shape {X.shape}, got {T.shape}")
    if not T.has_canonical_format:
        raise ValueError("T must be canonical (sorted rows, no duplicates): use the T from from_interactions")
    if not np.isfinite(T.data).all():
        raise ValueError("T values must be finite")
    return X, T


def _decayed(X: sps.csr_array, T: sps.csr_array, w: np.ndarray, min_weight: float) -> sps.csr_array:
    """X times the weight of the same entry in T (w: one per T entry); entries
    with weight 0 or < min_weight dropped. X's entries must be in T."""
    if np.array_equal(X.indptr, T.indptr) and np.array_equal(X.indices, T.indices):
        return _scaled(X, w, min_weight)  # the common case (X = the URM, bm25 of it): no matching
    n_cols = X.shape[1]
    t_key = np.repeat(np.arange(T.shape[0], dtype=np.int64), np.diff(T.indptr)) * n_cols + T.indices
    x_key = np.repeat(np.arange(X.shape[0], dtype=np.int64), np.diff(X.indptr)) * n_cols + X.indices
    pos = np.minimum(np.searchsorted(t_key, x_key), max(t_key.size - 1, 0))
    if x_key.size and (t_key.size == 0 or (t_key[pos] != x_key).any()):
        raise ValueError("X has entries T doesn't (T must come with X's interactions, e.g. from from_interactions)")
    return _scaled(X, w[pos] if x_key.size else w[:0], min_weight)


def _scaled(X: sps.csr_array, w: np.ndarray, min_weight: float) -> sps.csr_array:
    """X's values times w (one per X entry), in place; weight 0 or < min_weight dropped."""
    X.data[:] = X.data * w
    X.data[(w == 0) | (w < min_weight)] = 0
    X.eliminate_zeros()  # also an entry whose product underflows
    return X


def time_decay(
    X: sps.sparray | sps.spmatrix,
    T: sps.sparray | sps.spmatrix,
    *,
    shape: str = "exponential",
    scale,
    exponent: float | None = None,
    reference: str | float = "user",
    min_weight: float = 0.0,
) -> sps.csr_array:
    """
    Weight each interaction by its age in time: X * weight(reference - t).

    Shapes (weight 1 at age 0): "exponential" 0.5^(age / scale) (scale =
    half-life); "power" (1 + (2^(1/p) - 1) age / scale)^-p (also 0.5 at age
    = scale, a longer tail for a smaller p = `exponent`, default 1);
    "linear" max(0, 1 - age / scale) (0 from age = scale); "step" 1 if age
    < scale else 0 (a window).

    Args:
        X: user-item matrix (e.g. the URM, or bm25 of it): its entries must
            be in T. Decays chain: `rank_decay(time_decay(URM, T, ...), T, ...)`.
        T: timestamps, the T of `from_interactions` (same users x items).
        shape: "exponential", "power", "linear" or "step".
        scale: in T's units (seconds for datetimes), or a duration
            (`np.timedelta64`, `datetime.timedelta`, `pd.Timedelta`); required
            (no default fits every dataset). `np.inf` = no decay, e.g. the
            baseline of a parameter search.
        exponent: the power-law p (> 0), shape "power" only.
        reference: "user" (each user's latest timestamp: an inactive user
            isn't pushed down as a whole), "global" (T's latest), or a time
            (e.g. the training cutoff, in T's units).
        min_weight: entries with a smaller weight are dropped (the near-zero
            tail; caps a history's length).

    Returns:
        CSR matrix, X's values times the weights, zero-weight entries dropped.

    Raises:
        ValueError: on an event after `reference` (age < 0: data leaking
            from after the cutoff), X entries missing from T, `exponent`
            with a shape other than "power", a bad scale.
    """
    X, T = _timestamps_of(X, T)
    scale = _scale(scale)
    t = T.data
    if reference == "user":
        counts = np.diff(T.indptr)
        starts = T.indptr[:-1][counts > 0]
        ref = np.repeat(np.maximum.reduceat(t, starts), counts[counts > 0]) if t.size else t
    elif reference == "global":
        ref = t.max() if t.size else 0.0
    elif isinstance(reference, str):
        raise ValueError(f"reference must be 'user', 'global' or a time, got {reference!r}")
    else:
        ref = float(reference)
    age = ref - t
    if (age < 0).any():
        raise ValueError(f"{int((age < 0).sum())} timestamps are after reference {reference!r}")
    return _decayed(X, T, _weights(age, shape, scale, exponent), min_weight)


def rank_decay(
    X: sps.sparray | sps.spmatrix,
    T: sps.sparray | sps.spmatrix,
    *,
    shape: str = "exponential",
    scale: float,
    exponent: float | None = None,
    min_weight: float = 0.0,
) -> sps.csr_array:
    """
    Weight each interaction by its recency rank in the user's row: X *
    weight(rank), rank 0 = the latest item (equal timestamps: lower column
    first), counted over T's whole row (distinct items, whatever X keeps).

    Shapes as `time_decay`, `scale` in items: e.g. `shape="step",
    scale=200` keeps each user's 200 latest items, `shape="exponential",
    scale=50` halves the weight every 50 items.

    Args:
        X: user-item matrix, its entries in T (decays chain).
        T: timestamps, the T of `from_interactions`.
        shape: "exponential", "power", "linear" or "step".
        scale: in items (> 0, `np.inf` = no decay); required.
        exponent: the power-law p (> 0), shape "power" only.
        min_weight: entries with a smaller weight are dropped.

    Returns:
        CSR matrix, X's values times the weights, zero-weight entries dropped.

    Raises:
        ValueError: on X entries missing from T, `exponent` with a shape
            other than "power", a bad scale.
    """
    X, T = _timestamps_of(X, T)
    if isinstance(scale, np.timedelta64) or hasattr(scale, "total_seconds"):
        raise TypeError("rank_decay's scale is a number of items, not a duration (time_decay takes durations)")
    scale = _scale(scale)
    rank = _core.recency_rank(t=np.ascontiguousarray(T.data, dtype=np.float64), indptr=T.indptr.astype(np.int32))
    return _decayed(X, T, _weights(rank, shape, scale, exponent), min_weight)
