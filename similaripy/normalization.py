"""Normalizations and weightings of sparse matrices (row/column norms,
TF-IDF, BM25, binarize). Also at the top level (`sim.bm25`, as in 0.x)."""

from math import e
from typing import Literal

import numpy as np
import scipy.sparse as sps

from . import _core  # ty: ignore[unresolved-import]  (compiled, no stub)
from ._s_plus import SparseInput, _canonical_csr

__all__ = ["binarize", "bm25", "bm25plus", "normalize", "tfidf"]

# ---- Valid mode strings ----
_NORMALIZATIONS = ("l1", "l2", "max")
_TF_MODES = ("binary", "raw", "sqrt", "freq", "log")
_IDF_MODES = ("unary", "base", "smooth", "prob", "bm25")


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
        A new CSR matrix (duplicates summed, explicit zeros dropped, the input's
            float type kept); X never changes.

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
        A new CSR matrix (duplicates summed, explicit zeros dropped, the input's
            float type kept); X never changes.

    Raises:
        TypeError: on a non-sparse X.
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
    tf_mode: Literal["binary", "raw", "sqrt", "freq", "log"] = "raw",
    idf_mode: Literal["unary", "base", "smooth", "prob", "bm25"] = "bm25",
) -> sps.csr_array:
    """
    Apply BM25 normalization to a sparse matrix.

    Args:
        X: sparse matrix of term frequencies (e.g. the URM): values >= 0,
            checked after summing duplicates.
        axis: 1 = rows are the documents (default), 0 = columns.
        k1: term-frequency saturation: higher saturates later.
        b: document-length normalization, in [0, 1] (0 = none).
        logbase: base of the IDF logarithm.
        tf_mode: term-frequency transform: "binary", "raw", "sqrt", "freq" or
            "log" (see the TF modes in the Normalizations guide).
        idf_mode: inverse document frequency: "unary", "base", "smooth", "prob"
            or "bm25" (see the IDF modes in the Normalizations guide).

    Returns:
        A new CSR matrix (duplicates summed, explicit zeros dropped, the input's
            float type kept); X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a negative, NaN or inf value, or an invalid `axis`,
            `tf_mode` or `idf_mode`.
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
    tf_mode: Literal["binary", "raw", "sqrt", "freq", "log"] = "raw",
    idf_mode: Literal["unary", "base", "smooth", "prob", "bm25"] = "bm25",
) -> sps.csr_array:
    """
    Apply BM25+ normalization to a sparse matrix.

    Args:
        X: sparse matrix of term frequencies (e.g. the URM): values >= 0,
            checked after summing duplicates.
        axis: 1 = rows are the documents (default), 0 = columns.
        k1: term-frequency saturation: higher saturates later.
        b: document-length normalization, in [0, 1] (0 = none).
        delta: BM25+ lower bound added to every term weight (> 0 keeps rare
            terms in long documents from vanishing).
        logbase: base of the IDF logarithm.
        tf_mode: term-frequency transform: "binary", "raw", "sqrt", "freq" or
            "log" (see the TF modes in the Normalizations guide).
        idf_mode: inverse document frequency: "unary", "base", "smooth", "prob"
            or "bm25" (see the IDF modes in the Normalizations guide).

    Returns:
        A new CSR matrix (duplicates summed, explicit zeros dropped, the input's
            float type kept); X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a negative, NaN or inf value, or an invalid `axis`,
            `tf_mode` or `idf_mode`.
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
    tf_mode: Literal["binary", "raw", "sqrt", "freq", "log"] = "sqrt",
    idf_mode: Literal["unary", "base", "smooth", "prob", "bm25"] = "smooth",
) -> sps.csr_array:
    """
    Apply TF-IDF normalization to a sparse matrix.

    Args:
        X: sparse matrix of term frequencies (e.g. the URM): values >= 0,
            checked after summing duplicates.
        axis: 1 = rows are the documents (default), 0 = columns.
        logbase: base of the IDF logarithm.
        tf_mode: term-frequency transform: "binary", "raw", "sqrt", "freq" or
            "log" (see the TF modes in the Normalizations guide).
        idf_mode: inverse document frequency: "unary", "base", "smooth", "prob"
            or "bm25" (see the IDF modes in the Normalizations guide).

    Returns:
        A new CSR matrix (duplicates summed, explicit zeros dropped, the input's
            float type kept); X never changes.

    Raises:
        TypeError: on a non-sparse X.
        ValueError: on a negative, NaN or inf value, or an invalid `axis`,
            `tf_mode` or `idf_mode`.
    """
    _validate_modes(tf_mode, idf_mode)
    return _transformed(X, axis, _core.tfidf, "tfidf", tf_mode=tf_mode, idf_mode=idf_mode, logbase=logbase)
