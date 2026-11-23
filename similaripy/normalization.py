"""Normalizations and weightings of sparse matrices (row/column norms,
TF-IDF, BM25, binarize)."""

from math import e, isfinite
from typing import Literal

import numpy as np
import scipy.sparse as sps

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import Csr, SparseInput, _canonical_csr, _indptr_dtype

__all__ = ["binarize", "bm25", "bm25plus", "normalize", "tfidf"]

# ---- Valid mode strings ----
_NORMALIZATIONS = ("l1", "l2", "max")
_TF_MODES = ("binary", "raw", "sqrt", "freq", "log")
_IDF_MODES = ("unary", "base", "smooth", "prob", "bm25", "bm25_robertson")


# ---- Private helpers ----


def _canonical(X: SparseInput) -> Csr:
    """
    The call's own canonical CSR copy of X (`_canonical_csr`): float32 values,
    int32 indices.

    Raises:
        TypeError: If X is not a sparse matrix.
    """
    if not sps.issparse(X):
        raise TypeError("X must be a sparse matrix")
    return _canonical_csr(X, 0)


def _transformed(X: SparseInput, axis: int, op, name: str | None = None, **kwargs) -> sps.csr_array:
    """
    The canonical copy of X with `op` (an in-place `_core` normalization) applied
    to its rows (axis 1) or columns (axis 0, column statistics in the core: no
    transposes); `name`: a term weighting, whose input must be non-negative.

    Raises:
        ValueError: If axis is not 0 or 1.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    X = _canonical(X)
    _check_finite(X)
    if name is not None:
        _check_non_negative(X, name)
    op(data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape, axis=axis, **kwargs)
    return _to_scipy(X)


def _to_scipy(X: Csr) -> sps.csr_array:
    """The canonical arrays as a scipy CSR, indptr by `_indptr_dtype` (past
    2^31 - 1 entries scipy makes the indices int64 too)."""
    indptr = X.indptr.astype(_indptr_dtype(len(X.data)), copy=False)
    out = sps.csr_array((X.data, X.indices, indptr), shape=X.shape)
    out.has_canonical_format = True  # sorted, summed: skips scipy's re-checks
    return out


def _check_finite(X: Csr) -> None:
    """NaN/inf would spread through a row's norm or idf, silently (as the
    similarities, checked on the canonical copy)."""
    if not np.isfinite(X.data).all():
        raise ValueError("X values must be finite")


def _check_non_negative(X: Csr, name: str) -> None:
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


def _validate_weighting(logbase: float, k1: float = 0.0, b: float = 0.0, delta: float = 0.0) -> None:
    """
    The values that give silently wrong weights: logbase <= 1 flips every log's
    sign (1 divides by 0), k1 < 0 or b outside [0, 1] can make the BM25
    denominator <= 0, delta < 0 can make a weight negative.

    Raises:
        ValueError: on any of them, or a NaN / inf.
    """
    if not (isfinite(logbase) and logbase > 1):
        raise ValueError(f"logbase must be finite and > 1, got {logbase}")
    if not (isfinite(k1) and k1 >= 0):
        raise ValueError(f"k1 must be finite and >= 0, got {k1}")
    if not 0 <= b <= 1:
        raise ValueError(f"b must be in [0, 1], got {b}")
    if not (isfinite(delta) and delta >= 0):
        raise ValueError(f"delta must be finite and >= 0, got {delta}")


# ---- Public API ----


def normalize(
    X: sps.sparray | sps.spmatrix,
    *,
    norm: Literal["l1", "l2", "max"] = "l2",
    axis: int = 1,
) -> sps.csr_array:
    """
    Normalize a sparse matrix along rows or columns using L1, L2, or max-norm.

    Signed input is fine; rows of zeros (empty after canonicalization) stay
    unchanged.

    Args:
        X: sparse matrix; negative values are fine.
        norm: "l1" (sum of |x|), "l2" (sqrt of the sum of x^2) or "max"
            (the largest |x|).
        axis: 1 = normalize each row (default), 0 = each column.

    Returns:
        A new float32 CSR matrix (duplicates summed, explicit zeros dropped);
            X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a NaN or inf value, or an invalid `norm` or `axis`.
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
        X: sparse matrix; negative values count as interactions too.

    Returns:
        A new float32 CSR matrix (duplicates summed, explicit zeros dropped);
            X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a NaN or inf value (checked before the values become 1).
    """
    C = _canonical(X)
    _check_finite(C)  # before the values become 1
    C.data[:] = 1
    return _to_scipy(C)


def bm25(
    X: sps.sparray | sps.spmatrix,
    *,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    logbase: float = e,
    tf_mode: Literal["binary", "raw", "sqrt", "freq", "log"] = "raw",
    idf_mode: Literal["unary", "base", "smooth", "prob", "bm25", "bm25_robertson"] = "bm25",
) -> sps.csr_array:
    """
    Apply BM25 normalization to a sparse matrix.

    Each entry becomes `idf * tf / (tf + k1 * (1 - b + b * len / avg_len))`, the
    Lucene / Elasticsearch / bm25s formula: positive with the default
    `idf_mode`, and without Robertson's `(k1 + 1)` factor, a constant that
    changes no ranking.

    Args:
        X: sparse matrix of term frequencies (e.g. the URM): values >= 0,
            checked after summing duplicates.
        axis: 1 = rows are the documents (default), 0 = columns.
        k1: term-frequency saturation, >= 0: higher saturates later.
        b: document-length normalization, in [0, 1] (0 = none).
        logbase: base of the IDF logarithm, > 1.
        tf_mode: term-frequency transform: "binary", "raw", "sqrt", "freq" or
            "log" (see the TF modes in the Normalizations guide).
        idf_mode: inverse document frequency: "unary", "base", "smooth", "prob",
            "bm25" (always > 0) or "bm25_robertson" (negative for terms in more
            than half the documents); see the IDF modes in the Normalizations guide.

    Returns:
        A new float32 CSR matrix (duplicates summed, explicit zeros dropped);
            X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a negative, NaN or inf value, an invalid `axis`,
            `tf_mode` or `idf_mode`, or a `logbase` <= 1, `k1` < 0 or `b`
            outside [0, 1].
    """
    _validate_modes(tf_mode, idf_mode)
    _validate_weighting(logbase, k1, b)
    return _transformed(
        X, axis, _core.bm25plus, "bm25", k1=k1, b=b, delta=0.0, tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase
    )


def bm25plus(
    X: sps.sparray | sps.spmatrix,
    *,
    axis: int = 1,
    k1: float = 1.2,
    b: float = 0.75,
    delta: float = 1.0,
    logbase: float = e,
    tf_mode: Literal["binary", "raw", "sqrt", "freq", "log"] = "raw",
    idf_mode: Literal["unary", "base", "smooth", "prob", "bm25", "bm25_robertson"] = "bm25",
) -> sps.csr_array:
    """
    Apply BM25+ normalization to a sparse matrix.

    Each entry becomes `idf * (tf / (tf + k1 * (1 - b + b * len / avg_len)) + delta)`:
    `bm25` plus `delta * idf`, with the same TF part as `bm25` (no `(k1 + 1)`
    factor): to rank like the paper's BM25+ with delta d, pass `delta=d / (k1 + 1)`.

    Args:
        X: sparse matrix of term frequencies (e.g. the URM): values >= 0,
            checked after summing duplicates.
        axis: 1 = rows are the documents (default), 0 = columns.
        k1: term-frequency saturation, >= 0: higher saturates later.
        b: document-length normalization, in [0, 1] (0 = none).
        delta: BM25+ lower bound added to every term weight, >= 0 (> 0 keeps rare
            terms in long documents from vanishing).
        logbase: base of the IDF logarithm, > 1.
        tf_mode: term-frequency transform: "binary", "raw", "sqrt", "freq" or
            "log" (see the TF modes in the Normalizations guide).
        idf_mode: inverse document frequency: "unary", "base", "smooth", "prob",
            "bm25" (always > 0) or "bm25_robertson" (negative for terms in more
            than half the documents); see the IDF modes in the Normalizations guide.

    Returns:
        A new float32 CSR matrix (duplicates summed, explicit zeros dropped);
            X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a negative, NaN or inf value, an invalid `axis`,
            `tf_mode` or `idf_mode`, or a `logbase` <= 1, `k1` < 0, `b`
            outside [0, 1] or `delta` < 0.
    """
    _validate_modes(tf_mode, idf_mode)
    _validate_weighting(logbase, k1, b, delta)
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
    *,
    axis: int = 1,
    logbase: float = e,
    tf_mode: Literal["binary", "raw", "sqrt", "freq", "log"] = "sqrt",
    idf_mode: Literal["unary", "base", "smooth", "prob", "bm25", "bm25_robertson"] = "smooth",
) -> sps.csr_array:
    """
    Apply TF-IDF normalization to a sparse matrix.

    Args:
        X: sparse matrix of term frequencies (e.g. the URM): values >= 0,
            checked after summing duplicates.
        axis: 1 = rows are the documents (default), 0 = columns.
        logbase: base of the IDF logarithm, > 1.
        tf_mode: term-frequency transform: "binary", "raw", "sqrt", "freq" or
            "log" (see the TF modes in the Normalizations guide).
        idf_mode: inverse document frequency: "unary", "base", "smooth", "prob",
            "bm25" (always > 0) or "bm25_robertson" (negative for terms in more
            than half the documents); see the IDF modes in the Normalizations guide.

    Returns:
        A new float32 CSR matrix (duplicates summed, explicit zeros dropped);
            X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a negative, NaN or inf value, an invalid `axis`,
            `tf_mode` or `idf_mode`, or a `logbase` <= 1.
    """
    _validate_modes(tf_mode, idf_mode)
    _validate_weighting(logbase)
    return _transformed(X, axis, _core.tfidf, "tfidf", tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)
